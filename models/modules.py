import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from torchvision import models
from torch import optim
from torch.distributions import Normal, ContinuousBernoulli
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from models import igebm


class LikelihoodDecoder(nn.Module):
    def log_likelihood(self, x, z):
        decoder_out = self.forward_(z)
        # from pudb import set_trace; set_trace()
        if self.likelihood_type == 'isotropic_gaussian':
            # equivalent to using Euclidean distance
            D = torch.prod(torch.tensor(x.shape[1:]))
            sig = torch.tensor(1, dtype=torch.float32)
            const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
            loglik = const - 0.5 * ((x - decoder_out)**2).view((x.shape[0], -1)).sum(dim=1) / (sig ** 2)
        elif self.likelihood_type == 'diagonal_gaussian':
            original_dim = x.shape[1]
            D = torch.prod(torch.tensor(x.shape[1:])).cuda()
            mu = decoder_out[:, :original_dim]
            sig = torch.exp(decoder_out[:, original_dim:]) + 0.5
            # sig = torch.ones_like(x).cuda() * 0.01
            const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig.view(x.shape[0], -1)).sum(dim=1)
            loglik = const - 0.5 * (((mu - x) ** 2) / (sig ** 2)).view((x.shape[0], -1)).sum(dim=1)
            # normal = Normal(mu, sig)
            # loglik_per_pixel = normal.log_prob(x)
            # loglik = loglik_per_pixel.view((x.shape[0], -1)).sum(dim=1)
        elif self.likelihood_type == 'bernoulli':
            pass
        else:
            raise ValueError(f'Undefined likelihood_type: {self.likelihood}')
        return loglik

    def modify_forward_output(self, out):
        out_chan = out.shape[1]
        if self.likelihood_type == 'diagonal_gaussian':
            return out[:, :out_chan // 2]
        else:
            return out


class IsotropicGaussian(nn.Module):
    """Isotripic Gaussian density function paramerized by a neural net.
    standard deviation is a free scalar parameter"""
    def __init__(self, net, sigma=1., sigma_trainable=False, error_normalize=True, deterministic=False):
        super().__init__()
        self.net = net
        self.sigma_trainable = sigma_trainable
        self.error_normalize = error_normalize
        self.deterministic = deterministic
        if sigma_trainable:
            # self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
            self.register_parameter('sigma', nn.Parameter(torch.tensor(sigma, dtype=torch.float)))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float))

    def log_likelihood(self, x, z):
        decoder_out = self.net(z)
        if self.deterministic:
            return - ((x - decoder_out)**2).view((x.shape[0], -1)).sum(dim=1) 
        else:
            D = torch.prod(torch.tensor(x.shape[1:]))
            # sig = torch.tensor(1, dtype=torch.float32)
            sig = self.sigma
            const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
            loglik = const - 0.5 * ((x - decoder_out)**2).view((x.shape[0], -1)).sum(dim=1) / (sig ** 2)
            return loglik

    def error(self, x, x_hat):
        if not self.error_normalize:
            return (((x - x_hat) / self.sigma) ** 2).view(len(x), -1).sum(-1)
        else:
            return ((x - x_hat) ** 2).view(len(x), -1).mean(-1)

    def forward(self, z):
        """returns reconstruction"""
        return self.net(z)

    def sample(self, z):
        if self.deterministic:
            return self.mean(z)
        else:
            x_hat = self.net(z)
            return x_hat + torch.randn_like(x_hat) * self.sigma

    def mean(self, z):
        return self.net(z)

    def max_log_likelihood(self, x):
        if self.deterministic:
            return torch.tensor(0., dtype=torch.float, device=x.device)
        else:
            D = torch.prod(torch.tensor(x.shape[1:]))
            sig = self.sigma
            const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
            return const


class CB(nn.Module):
    """continuous Bernoulli distribution layer for modeling images in a real domain """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def log_likelihood(self, x, z):
        cb = self(z)
        return cb.log_prob(x).view(len(x), -1).sum(dim=1)

    def forward(self, z):
        cb = ContinuousBernoulli(probs=self.net(z))
        return cb

    def sample(self, z):
        cb = self(z)
        return cb.rsample()

class IsotropicLaplace(nn.Module):
    """Isotropic Laplace density function -- equivalent to using L1 error """
    def __init__(self, net, sigma=0.1, sigma_trainable=False):
        super().__init__()
        self.net = net
        self.sigma_trainable = sigma_trainable
        if sigma_trainable:
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float))

    def log_likelihood(self, x, z):
        # decoder_out = self.net(z)
        # D = torch.prod(torch.tensor(x.shape[1:]))
        # sig = torch.tensor(1, dtype=torch.float32)
        # const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
        # loglik = const - 0.5 * (torch.abs(x - decoder_out)).view((x.shape[0], -1)).sum(dim=1) / (sig ** 2)
        # return loglik
        raise NotImplementedError

    def error(self, x, x_hat):
        if self.sigma_trainable:
            return ((torch.abs(x - x_hat) / self.sigma)).view(len(x), -1).sum(-1)
        else:
            return (torch.abs(x - x_hat)).view(len(x), -1).mean(-1)

    def forward(self, z):
        """returns reconstruction"""
        return self.net(z)

    def sample(self, z):
        # x_hat = self.net(z) 
        # return x_hat + torch.randn_like(x_hat) * self.sigma
        raise NotImplementedError


# Simple ConvNet / DeconvNet pair
class ConvNet1(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=8, out_activation='linear'):
        """nh: determines the numbers of conv filters"""
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, 8, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, bias=True)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, bias=True)
        self.conv5 = nn.Conv2d(64, out_chan, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max2(x)
        x = self.conv5(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x


class DeConvNet1(LikelihoodDecoder):
    def __init__(self, in_chan=1, out_chan=1, nh=8):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet1, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_chan, nh * 16, kernel_size=3, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=3, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        return x


# ConvNet desigend for 28x28 input
class ConvNet2(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=8, activation='relu', out_activation='linear', use_spectral_norm=False, learn_out_scale=False):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, out_chan, kernel_size=4, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.activation = activation
        self.out_activation = get_activation(out_activation)
        if learn_out_scale:
            self.out_scale = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=True)
        else:
            self.out_scale = None

        if use_spectral_norm:
            self.conv1 = P.spectral_norm(self.conv1)
            self.conv2 = P.spectral_norm(self.conv2)
            self.conv3 = P.spectral_norm(self.conv3)
            self.conv4 = P.spectral_norm(self.conv4)
            self.conv5 = P.spectral_norm(self.conv5)

    def forward(self, x):
        x = self.conv1(x)
        x = get_activation_F(x, self.activation)
        x = self.conv2(x)
        x = get_activation_F(x, self.activation)
        x = self.max1(x)
        x = self.conv3(x)
        x = get_activation_F(x, self.activation)
        x = self.conv4(x)
        x = get_activation_F(x, self.activation)
        x = self.max2(x)
        x = self.conv5(x)
        if self.out_scale is not None:
            x = self.out_scale(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x


class ConvNet2FC(nn.Module):
    """additional 1x1 conv layer at the top"""
    def __init__(self, in_chan=1, out_chan=64, nh=8, nh_mlp=512, activation='relu', out_activation='linear', use_spectral_norm=False, learn_out_scale=False):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2FC, self).__init__()
        conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv5 = nn.Conv2d(nh * 16, nh_mlp, kernel_size=4, bias=True)
        conv6 = nn.Conv2d(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        out_activation = get_activation(out_activation)
        # assert activation == 'relu'

        if use_spectral_norm:
            conv1 = P.spectral_norm(conv1)
            conv2 = P.spectral_norm(conv2)
            conv3 = P.spectral_norm(conv3)
            conv4 = P.spectral_norm(conv4)
            conv5 = P.spectral_norm(conv5)

        layers = [conv1,
                  get_activation(activation),
                  conv2,
                  get_activation(activation),
                  max1,
                  conv3,
                  get_activation(activation),
                  conv4,
                  get_activation(activation),
                  max2,
                  conv5,
                  get_activation(activation),
                  conv6,]
        if learn_out_scale:
            layers.append(nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=True))
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvNet2BN(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=8, out_activation='linear'):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2BN, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.bn1 = nn.BatchNorm2d(nh * 4)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.bn2 = nn.BatchNorm2d(nh * 8)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.bn3 = nn.BatchNorm2d(nh * 8)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.bn4 = nn.BatchNorm2d(nh * 16)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, out_chan, kernel_size=4, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.max2(x)
        x = self.conv5(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def parameters_wo_bn(self):
        l_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        return itertools.chain(*[layer.parameters() for layer in l_layers])


class ConvNet2Att(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=8, resdim=1024, n_res=1, out_activation='linear', ver='V2'):
        """nh: determines the numbers of conv filters"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, out_chan, kernel_size=4, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation

        self.resdim = resdim
        self.n_res = n_res
        self.ver = ver
        self.l_block = []
        self.l_att = []
        for i_res in range(n_res):
            if ver == 'V2':
                self.l_block.append(ResBlockEBAEV2(out_chan, resdim))
                self.l_att.append(ResBlockEBAEV2(out_chan, resdim))
            else:
                raise NotImplementedError

        layers = [self.conv1,
                  nn.ReLU(),
                  self.conv2,
                  nn.ReLU(),
                  self.max1,
                  self.conv3,
                  nn.ReLU(),
                  self.conv4,
                  nn.ReLU(),
                  self.max2,
                  self.conv5]

        if self.out_activation == 'tanh':
            layers.append(nn.Tanh())
        elif self.out_activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.resblock = nn.Sequential(*self.l_block)
        self.att = nn.Sequential(*self.l_att)

    def forward(self, x):
        x = self.net(x)
        res_out = self.resblock(x)
        att_out = self.att(x)
        return x + torch.sigmoid(att_out) * res_out


class DeConvNet2(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation='linear',
                 use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet2, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_chan, nh * 16, kernel_size=4, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=3, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation) 

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)
            self.conv4 = spectral_norm(self.conv4)
            self.conv5 = spectral_norm(self.conv5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x


class DeConvNet2BN(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation='linear'):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet2BN, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_chan, nh * 16, kernel_size=4, bias=True)
        self.bn1 = nn.BatchNorm2d(nh * 16)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.bn2 = nn.BatchNorm2d(nh * 8)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.bn3 = nn.BatchNorm2d(nh * 8)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.bn4 = nn.BatchNorm2d(nh * 4)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=3, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def parameters_wo_bn(self):
        l_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        return itertools.chain(*[layer.parameters() for layer in l_layers])


# ConvNet desigend for 28x28 input
# add 1x1 conv
class ConvNet2p(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=8, out_activation='linear'):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2p, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, out_chan * 2, kernel_size=4, bias=True)
        self.conv6 = nn.Conv2d(out_chan * 2, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max2(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        if self.out_activation == 'tanh':
            x = torch.tanh(x)
        elif self.out_activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x


'''
ConvNet for CIFAR10, following architecture in (Ghosh et al., 2019)
but excluding batch normalization
'''
class ConvNet3(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=32, out_activation='linear', activation='relu'):
        """nh: determines the numbers of conv filters"""
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=4, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=4, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=4, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=2, bias=True, stride=2)
        self.fc1 = nn.Conv2d(nh * 32, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan

        layers = [self.conv1,
                  get_activation(activation),
                  self.conv2,
                  get_activation(activation),
                  self.conv3,
                  get_activation(activation),
                  self.conv4,
                  get_activation(activation),
                  self.fc1]
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvNet3FC(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=32, nh_mlp=1024, out_activation='linear', activation='relu', use_spectral_norm=False, learn_out_scale=False):
        """nh: determines the numbers of conv filters"""
        super(ConvNet3FC, self).__init__()
        conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=4, bias=True, stride=2)
        conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=4, bias=True, stride=2)
        conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=4, bias=True, stride=2)
        conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=2, bias=True, stride=2)
        fc1 = nn.Conv2d(nh * 32, nh_mlp, kernel_size=1, bias=True)
        fc2 = nn.Conv2d(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan

        if use_spectral_norm:
            conv1 = P.spectral_norm(conv1)
            conv2 = P.spectral_norm(conv2)
            conv3 = P.spectral_norm(conv3)
            conv4 = P.spectral_norm(conv4)

        layers = [conv1,
                  get_activation(activation),
                  conv2,
                  get_activation(activation),
                  conv3,
                  get_activation(activation),
                  conv4,
                  get_activation(activation),
                  fc1,
                  get_activation(activation),
                  fc2]
        if learn_out_scale:
            layers.append(nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=True))

        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvNet3FCBN(nn.Module):
    """ with Batch Normalization """
    def __init__(self, in_chan=1, out_chan=64, nh=32, nh_mlp=1024, out_activation='linear', activation='relu', encoding_range=None):
        """nh: determines the numbers of conv filters"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=4, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=4, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=4, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=2, bias=True, stride=2)
        self.fc1 = nn.Conv2d(nh * 32, nh_mlp, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.encoding_range = encoding_range
        
        layers = [self.conv1,
                  nn.BatchNorm2d(nh*4),
                  get_activation(activation),
                  self.conv2,
                  nn.BatchNorm2d(nh*8),
                  get_activation(activation),
                  self.conv3,
                  nn.BatchNorm2d(nh*16),
                  get_activation(activation),
                  self.conv4,
                  nn.BatchNorm2d(nh*32),
                  get_activation(activation),
                  self.fc1,
                  nn.BatchNorm2d(nh_mlp),
                  get_activation(activation),
                  self.fc2]
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.encoding_range is not None:
            half_chan = int(x.shape[1] / 2)
            out = self.net(x)
            out[:half_chan] = self.encoding_range * torch.tanh(out[:half_chan])
            return out
        else:
            return self.net(x)


class ConvNet3MLP(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=32, l_nh_mlp=(1024,), out_activation='linear', activation='relu'):
        """nh: determines the numbers of conv filters"""
        super(ConvNet3MLP, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=4, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=4, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=4, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=2, bias=True, stride=2)
        self.l_nh_mlp = l_nh_mlp
        self.l_fc = []
        prev_out = nh * 32
        for nh_mlp in l_nh_mlp:
            fc = nn.Conv2d(prev_out, nh_mlp, kernel_size=1, bias=True)
            self.l_fc.append(fc)
            prev_out = nh_mlp
        fc = nn.Conv2d(prev_out, out_chan, kernel_size=1, bias=True)
        self.l_fc.append(fc)

        self.in_chan, self.out_chan = in_chan, out_chan

        layers = [self.conv1,
                  get_activation(activation),
                  self.conv2,
                  get_activation(activation),
                  self.conv3,
                  get_activation(activation),
                  self.conv4,
                  ]
        for fc in self.l_fc:
            layers.append(get_activation(activation))
            layers.append(fc)

        layers.append(get_activation(out_activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvNet64(nn.Module):
    """ConvNet architecture for CelebA64 following Ghosh et al., 2019"""
    def __init__(self, in_chan=3, out_chan=64, nh=32, out_activation='linear', activation='relu',
                 num_groups=None, use_bn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=5, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=5, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=5, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=5, bias=True, stride=2)
        self.fc1 = nn.Conv2d(nh * 32, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.num_groups = num_groups
        self.use_bn = use_bn

        layers = []
        layers.append(self.conv1)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 4))
        layers.append(get_activation(activation))
        layers.append(self.conv2)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 8))
        layers.append(get_activation(activation))
        layers.append(self.conv3)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 16))
        layers.append(get_activation(activation))
        layers.append(self.conv4)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 32))
        layers.append(get_activation(activation))
        layers.append(self.fc1)
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_norm_layer(self, num_channels):
        if self.num_groups is not None:
            return nn.GroupNorm(num_groups=self.num_groups, num_channels=num_channels)
        elif self.use_bn:
            return nn.BatchNorm2d(num_channels)


class DeConvNet64(nn.Module):
    """ConvNet architecture for CelebA64 following Ghosh et al., 2019"""
    def __init__(self, in_chan=64, out_chan=3, nh=32, out_activation='linear', activation='relu',
                 num_groups=None, use_bn=False):
        super().__init__()
        self.fc1 = nn.ConvTranspose2d(in_chan, nh * 32, kernel_size=8, bias=True)
        self.conv1 = nn.ConvTranspose2d(nh * 32, nh * 16, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.num_groups = num_groups
        self.use_bn = use_bn

        layers = []
        layers.append(self.fc1)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 32))
        layers.append(get_activation(activation))
        layers.append(self.conv1)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 16))
        layers.append(get_activation(activation))
        layers.append(self.conv2)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 8))
        layers.append(get_activation(activation))
        layers.append(self.conv3)
        if num_groups is not None:
            layers.append(self.get_norm_layer(num_channels=nh * 4))
        layers.append(get_activation(activation))
        layers.append(self.conv4)
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_norm_layer(self, num_channels):
        if self.num_groups is not None:
            return nn.GroupNorm(num_groups=self.num_groups, num_channels=num_channels)
        elif self.use_bn:
            return nn.BatchNorm2d(num_channels)



class ResBlockEBAE(nn.Module):
    """Residual connection for fully connected layer"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ResBlockEBAEV2(nn.Module):
    """Residual connection for fully connected layer"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.hidden_dim = hidden_dim
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ResBlockEBAEV3(nn.Module):
    """Residual connection for fully connected layer"""
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return x + self.block(x) * self.att(x)



class ConvNet3Res(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=32, resdim=1024, n_res=1, out_activation='linear', activation='relu', ver='V1'):
        """nh: determines the numbers of conv filters"""
        super(ConvNet3Res, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=4, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=4, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=4, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=2, bias=True, stride=2)
        self.fc1 = nn.Conv2d(nh * 32, resdim, kernel_size=1, bias=True)  # before resblocks
        self.fc2 = nn.Conv2d(resdim, out_chan, kernel_size=1, bias=True) # after resblocks
        self.resdim = resdim
        self.n_res = n_res
        self.ver = ver
        self.l_res = []
        for i_res in range(n_res):
            if ver == 'V1':
                self.l_res.append(ResBlockEBAE(out_chan))
            elif ver == 'V2':
                self.l_res.append(ResBlockEBAEV2(out_chan))
            elif ver == 'V3':
                self.l_res.append(ResBlockEBAEV3(resdim))

        self.in_chan, self.out_chan = in_chan, out_chan

        layers = [self.conv1,
                  get_activation(activation),
                  self.conv2,
                  get_activation(activation),
                  self.conv3,
                  get_activation(activation),
                  self.conv4,
                  get_activation(activation),
                  self.fc1,
                  ]
        for res in self.l_res:
            layers.append(res)
        layers.append(act())
        layers.append(self.fc2)
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvNet3Att(nn.Module):
    def __init__(self, in_chan=1, out_chan=64, nh=32, resdim=1024, n_res=1, out_activation='linear', activation='relu', ver='V1'):
        """nh: determines the numbers of conv filters"""
        super(ConvNet3Att, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=4, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=4, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=4, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=2, bias=True, stride=2)
        self.fc1 = nn.Conv2d(nh * 32, out_chan, kernel_size=1, bias=True)  # before resblocks
        # self.fc2 = nn.Conv2d(resdim, out_chan, kernel_size=1, bias=True) # after resblocks
        self.resdim = resdim
        self.n_res = n_res
        self.ver = ver
        self.l_block = []
        self.l_att = []
        for i_res in range(n_res):
            if ver == 'V1':
                self.l_block.append(ResBlockEBAE(resdim))
                raise NotImplementedError
            elif ver == 'V2':
                self.l_block.append(ResBlockEBAEV2(out_chan, resdim))
                self.l_att.append(ResBlockEBAEV2(out_chan, resdim))
            elif ver == 'V3':
                self.l_res.append(ResBlockEBAEV3(resdim))
                self.l_att.append(ResBlockEBAEV3(out_chan, resdim))
                raise NotImplementedError

        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'leakyrelu':
            def act():
                return nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise ValueError

        layers = [self.conv1,
                  act(),
                  self.conv2,
                  act(),
                  self.conv3,
                  act(),
                  self.conv4,
                  act(),
                  self.fc1,
                  ]
        # for res in self.l_res:
        #     layers.append(res)
        # layers.append(act())
        # layers.append(self.fc2)

        if self.out_activation == 'tanh':
            layers.append(nn.Tanh())
        elif self.out_activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.resblock = nn.Sequential(*self.l_block)
        self.att = nn.Sequential(*self.l_att)

    def forward(self, x):
        x = self.net(x)
        res_out = self.resblock(x)
        att_out = self.att(x)
        return x + torch.sigmoid(att_out) * res_out


class ConvMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        if out_dim is None:
            out_dim = dim

        self.block = nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1))

    def forward(self, x):
        return self.block(x)


class ConvNet3AttV2(nn.Module):
    """ Spherical latent space. non-residual blocks. scalar attention mask """
    def __init__(self, in_chan=1, out_chan=64, nh=32, resdim=1024, out_activation='linear', activation='relu',
                 spherical=True):
        """nh: determines the numbers of conv filters"""
        super(ConvNet3AttV2, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=4, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=4, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=4, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=2, bias=True, stride=2)
        self.fc1 = nn.Conv2d(nh * 32, out_chan, kernel_size=1, bias=True)  # before resblocks
        # self.fc2 = nn.Conv2d(resdim, out_chan, kernel_size=1, bias=True) # after resblocks
        self.resdim = resdim
        self.l_block = []
        self.l_att = []
        self.l_block.append(ConvMLPBlock(out_chan, hidden_dim=resdim))
        self.l_att.append(ConvMLPBlock(out_chan, hidden_dim=resdim, out_dim=1))
        self.spherical = spherical

        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'leakyrelu':
            def act():
                return nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise ValueError

        layers = [self.conv1,
                  act(),
                  self.conv2,
                  act(),
                  self.conv3,
                  act(),
                  self.conv4,
                  act(),
                  self.fc1,
                  ]

        if self.out_activation == 'tanh':
            layers.append(nn.Tanh())
        elif self.out_activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.resblock = nn.Sequential(*self.l_block)
        self.att = nn.Sequential(*self.l_att)

    def forward(self, x):
        x = self.net(x)
        x = self.normalize(x)
        res_out = self.resblock(x)
        att_out = self.att(x)
        return x + torch.sigmoid(att_out) * res_out

    def forward_wo_att(self, x):
        return self.net(x)

    def normalize(self, z):
        """normalize to unit length"""
        if self.spherical:
            if len(z.shape) == 4:
                z = z / z.view(len(z), -1).norm(dim=-1)[:, None, None, None]
            else:
                z = z / z.view(len(z), -1).norm(dim=1, keepdim=True)
            return z
        else:
            return z






class DeConvNet3(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=32, out_activation='linear',
                 activation='relu', num_groups=None):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet3, self).__init__()
        self.num_groups = num_groups
        self.fc1 = nn.ConvTranspose2d(in_chan, nh * 32, kernel_size=8, bias=True)
        self.conv1 = nn.ConvTranspose2d(nh * 32, nh * 16, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan

        layers = [self.fc1,] 
        layers += [] if self.num_groups is None else [self.get_norm_layer(nh*32)]
        layers += [get_activation(activation), self.conv1,]
        layers += [] if self.num_groups is None else [self.get_norm_layer(nh*16)]
        layers += [get_activation(activation), self.conv2,]
        layers += [] if self.num_groups is None else [self.get_norm_layer(nh*8)]
        layers += [get_activation(activation), self.conv3] 
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_norm_layer(self, num_channels):
        if self.num_groups is not None:
            return nn.GroupNorm(num_groups=self.num_groups, num_channels=num_channels)
        # elif self.use_bn:
        #     return nn.BatchNorm2d(num_channels)
        else:
            return None




class IGEBMEncoder(nn.Module):
    """Neural Network used in IGEBM"""
    def __init__(self, in_chan=3, out_chan=1, n_class=None, use_spectral_norm=False, keepdim=True,
                 out_activation='linear', avg_pool_dim=1):
        super().__init__()
        self.keepdim = keepdim
        self.use_spectral_norm = use_spectral_norm
        self.avg_pool_dim = avg_pool_dim

        if use_spectral_norm:
            self.conv1 = spectral_norm(nn.Conv2d(in_chan, 128, 3, padding=1), std=1)
        else:
            self.conv1 = nn.Conv2d(in_chan, 128, 3, padding=1)

        self.blocks = nn.ModuleList(
            [
                igebm.ResBlock(128, 128, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(128, 128, n_class, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(128, 256, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(256, 256, n_class, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(256, 256, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                igebm.ResBlock(256, 256, n_class, use_spectral_norm=use_spectral_norm),
            ]
        )

        if keepdim:
            self.linear = nn.Conv2d(256, out_chan, 1, 1, 0)
        else:
            self.linear = nn.Linear(256, out_chan)
        if use_spectral_norm:
            self.linear = spectral_norm(self.linear)

        self.out_activation = get_activation(out_activation)
        self.pre_activation = None

    def forward(self, input, class_id=None):
        out = self.conv1(input)

        out = F.leaky_relu(out, negative_slope=0.2)

        for block in self.blocks:
            out = block(out, class_id)

        out = F.relu(out)
        if self.keepdim:
            out = F.adaptive_avg_pool2d(out, (self.avg_pool_dim, self.avg_pool_dim))
        else:
            out = out.view(out.shape[0], out.shape[1], -1).sum(2)

        out = self.linear(out)
        self.pre_activation = out
        if self.out_activation is not None:
            out = self.out_activation(out)

        return out


class IGEBMEncoderV2(nn.Module):
    """Neural Network used in IGEBM
    replace spectral norm implementation
    add learn_out_scaling"""
    def __init__(self, in_chan=3, out_chan=1, n_class=None, use_spectral_norm=False, keepdim=True,
                 out_activation='linear', avg_pool_dim=1, learn_out_scale=False, nh=128):
        super().__init__()
        self.keepdim = keepdim
        self.use_spectral_norm = use_spectral_norm
        self.avg_pool_dim = avg_pool_dim

        if use_spectral_norm:
            self.conv1 = P.spectral_norm(nn.Conv2d(in_chan, nh, 3, padding=1))
        else:
            self.conv1 = nn.Conv2d(in_chan, nh, 3, padding=1)

        self.blocks = nn.ModuleList(
            [
                igebm.ResBlockV2(nh, nh, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                igebm.ResBlockV2(nh, nh, n_class, use_spectral_norm=use_spectral_norm),
                igebm.ResBlockV2(nh, nh*2, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                igebm.ResBlockV2(nh*2, nh*2, n_class, use_spectral_norm=use_spectral_norm),
                igebm.ResBlockV2(nh*2, nh*2, n_class, downsample=True, use_spectral_norm=use_spectral_norm),
                igebm.ResBlockV2(nh*2, nh*2, n_class, use_spectral_norm=use_spectral_norm),
            ]
        )

        if keepdim:
            self.linear = nn.Conv2d(nh*2, out_chan, 1, 1, 0)
        else:
            self.linear = nn.Linear(nh*2, out_chan)

        self.out_activation = get_activation(out_activation)
        self.pre_activation = None
        self.learn_out_scale = learn_out_scale
        if learn_out_scale:
            self.out_scale = nn.Linear(1, 1, bias=True)

    def forward(self, input, class_id=None):
        out = self.conv1(input)

        out = F.leaky_relu(out, negative_slope=0.2)

        for block in self.blocks:
            out = block(out, class_id)

        out = F.relu(out)
        if self.keepdim:
            out = F.adaptive_avg_pool2d(out, (self.avg_pool_dim, self.avg_pool_dim))
        else:
            out = out.view(out.shape[0], out.shape[1], -1).sum(2)

        out = self.linear(out)
        if self.learn_out_scale:
            out = self.out_scale(out)
        self.pre_activation = out
        if self.out_activation is not None:
            out = self.out_activation(out)

        return out



class ResBlock(nn.Module):
    """neural network architecture used in VQVAE code:
       https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VQVAEEncoder32_8(nn.Module):
    """neural network architecture used in VQVAE code:
       https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py"""
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 4, 2, 1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
            nn.Conv2d(out_chan, out_chan, 4, 2, 1),
            ResBlock(out_chan),
            ResBlock(out_chan),
        )

    def forward(self, x):
        return self.encoder(x)


class VQVAEDecoder32_8(nn.Module):
    """neural network architecture used in VQVAE code
       https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py"""
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.decoder = nn.Sequential(
            ResBlock(in_chan),
            ResBlock(in_chan),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_chan, in_chan, 4, 2, 1),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_chan, out_chan, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


class VQVAEEncoder32_4(nn.Module):
    """neural network architecture used in VQVAE code:
       https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py"""
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 4, 2, 1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
            nn.Conv2d(out_chan, out_chan, 4, 2, 1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
            nn.Conv2d(out_chan, out_chan, 4, 2, 1),
            ResBlock(out_chan),
            ResBlock(out_chan),
        )

    def forward(self, x):
        return self.encoder(x)


class VQVAEDecoder32_4(nn.Module):
    """neural network architecture used in VQVAE code
       https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py"""
    def __init__(self, in_chan, out_chan, out_activation='tanh'):
        super().__init__()
        l_layers = [ResBlock(in_chan),
            ResBlock(in_chan),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_chan, in_chan, 4, 2, 1),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_chan, out_chan, 4, 2, 1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_chan, out_chan, 4, 2, 1)]
        if out_activation is not None:
            l_layers.append(get_activation(out_activation))

        self.decoder = nn.Sequential(*l_layers)

    def forward(self, x):
        return self.decoder(x)


class SphericalActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if len(x.shape) == 4:
            x = x / x.view(len(x), -1).norm(dim=-1)[:, None, None, None]
        else:
            x = x / x.view(len(x), -1).norm(dim=1, keepdim=True)
        return x


class SphericalActivationV2(nn.Module):
    """Each spatial dimension is a hypersphere"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x / x.norm(dim=1, p=2, keepdim=True)
        return x


# Fully Connected Network
def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'spherical':
        return SphericalActivation()
    elif s_act == 'sphericalV2':
        return SphericalActivationV2()
    elif s_act == 'swish':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')


def get_activation_F(x, s_act):
    """functional version of get_activation"""
    if s_act == 'relu':
        return F.relu(x, inplace=True)
    elif s_act == 'sigmoid':
        return F.sigmoid(x)
    elif s_act == 'softplus':
        return F.softplus(x)
    elif s_act == 'linear':
        return x
    elif s_act == 'tanh':
        return F.tanh(x)
    elif s_act == 'leakyrelu':
        return F.leaky_relu(x, 0.2, inplace=True)
    elif s_act == 'softmax':
        return F.softmax(x, dim=1)
    elif s_act == 'spherical':
        return x / x.norm(dim=1, p=2, keepdim=True)
    elif s_act == 'sphericalV2':
        return x / x.norm(dim=1, p=2, keepdim=True)
    elif s_act == 'swish':
        return F.silu(x)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')


class FCNet(nn.Module):
    """fully-connected network"""
    def __init__(self,
            in_dim,
            out_dim,
            l_hidden=(50,),
            activation='sigmoid',
            out_activation='linear',
            use_spectral_norm=False,
            flatten_input=False,
            batch_norm=False,
            out_batch_norm=False,
            learn_out_scale=False,
            bias=True):
        super().__init__()
        l_neurons = tuple(l_hidden) + (out_dim,)
        if isinstance(activation, str):
            activation = (activation,) * len(l_hidden)
        activation = tuple(activation) + (out_activation,)

        l_layer = []
        prev_dim = in_dim
        for i_layer, (n_hidden, act) in enumerate(zip(l_neurons, activation)):
            if use_spectral_norm and i_layer < len(l_neurons) - 1:  # don't apply SN to the last layer
                l_layer.append(P.spectral_norm(nn.Linear(prev_dim, n_hidden)))
            else:
                l_layer.append(nn.Linear(prev_dim, n_hidden, bias=bias))
            if batch_norm:
                if out_batch_norm:
                    l_layer.append(nn.BatchNorm1d(num_features=n_hidden))
                else:
                    if i_layer < len(l_neurons) - 1:  # don't apply BN to the last layer
                        l_layer.append(nn.BatchNorm1d(num_features=n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        # add learnable scaling operation at the end
        if learn_out_scale:
            l_layer.append(nn.Linear(1, 1, bias=True))

        self.net = nn.Sequential(*l_layer)
        self.in_dim = in_dim
        self.out_shape = (out_dim,)
        self.flatten_input = flatten_input

    def forward(self, x):
        if self.flatten_input and len(x.shape) == 4:
            x = x.view(len(x), -1)
        return self.net(x)


class ConvMLP(nn.Module):
    def __init__(self, in_dim, out_dim, l_hidden=(50,), activation='sigmoid', out_activation='linear',
                 spatial_dim=1, fusion_at=0, use_spectral_norm=False, learn_out_scale=False):
        """
        a MLP-like feed-forward NN.
        spatial_dim: the size of input image
        fusion_at: the index of fusion layer
        """
        super().__init__()
        l_neurons = tuple(l_hidden) + (out_dim,)
        activation = (activation,) * len(l_hidden)
        activation = tuple(activation) + (out_activation,)

        l_layer = []
        prev_dim = in_dim
        for i_layer, (n_hidden, act) in enumerate(zip(l_neurons, activation)):
            if i_layer == fusion_at:
                if use_spectral_norm:
                    layer = P.spectral_norm(nn.Conv2d(prev_dim, n_hidden, spatial_dim, bias=True))
                else:
                    layer = nn.Conv2d(prev_dim, n_hidden, spatial_dim, bias=True)
            else:
                if use_spectral_norm and i_layer < len(l_neurons) - 1:  # don't apply SN to the last layer
                    layer = P.spectral_norm(nn.Conv2d(prev_dim, n_hidden, 1, bias=True))
                else:
                    layer = nn.Conv2d(prev_dim, n_hidden, 1, bias=True)
            l_layer.append(layer)
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        # add learnable scaling operation at the end
        if learn_out_scale:
            l_layer.append(nn.Linear(1, 1, bias=True))

        self.net = nn.Sequential(*l_layer)
        self.in_dim = in_dim

    def forward(self, x):
        return self.net(x)


class FCResNet(nn.Module):
    """FullyConnected Residual Network
    Input - Linear - (ResBlock * K) - Linear - Output"""
    def __init__(self, in_dim, out_dim, res_dim, n_res_hidden=100, n_resblock=2,
                 out_activation='linear', use_spectral_norm=False, flatten_input=False):
        super().__init__()
        self.flatten_input = flatten_input
        l_layer = []
        block = nn.Linear(in_dim, res_dim)
        if use_spectral_norm:
            block = spectral_norm(block)
        l_layer.append(block)

        for i_resblock in range(n_resblock):
            block = FCResBlock(res_dim, n_res_hidden, use_spectral_norm=use_spectral_norm)
            l_layer.append(block)
        l_layer.append(nn.ReLU())

        block = nn.Linear(res_dim, out_dim)
        if use_spectral_norm:
            block = spectral_norm(block)
        l_layer.append(block)
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            l_layer.append(out_activation)
        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        if self.flatten_input and len(x.shape) == 4:
            x = x.view(len(x), -1)
        return self.net(x)


class FCResBlock(nn.Module):
    def __init__(self, res_dim, n_res_hidden, use_spectral_norm=False):
        super().__init__()
        if use_spectral_norm:
            self.net = nn.Sequential(nn.ReLU(),
                                     spectral_norm(nn.Linear(res_dim, n_res_hidden)),
                                     nn.ReLU(),
                                     spectral_norm(nn.Linear(n_res_hidden, res_dim)))
        else:
            self.net = nn.Sequential(nn.ReLU(),
                                     nn.Linear(res_dim, n_res_hidden),
                                     nn.ReLU(),
                                     nn.Linear(n_res_hidden, res_dim))

    def forward(self, x):
        return x + self.net(x)


class ResNet1x1(nn.Module):
    """ResNet with 1x1 convolution"""
    def __init__(self, in_dim, out_dim, res_dim, n_res_hidden=100, n_resblock=2, out_activation='linear',
                 activation='relu', use_spectral_norm=False):
        super().__init__()
        l_layer = []
        block = nn.Conv2d(in_dim, res_dim, 1, bias=True)
        if use_spectral_norm:
            block = spectral_norm(block)
        l_layer.append(block)

        for i_resblock in range(n_resblock):
            block = ResBlock1x1(res_dim, n_res_hidden, use_spectral_norm=use_spectral_norm, activation=activation)
            l_layer.append(block)
        # l_layer.append(nn.ReLU())
        l_layer.append(get_activation(activation))

        block = nn.Conv2d(res_dim, out_dim, 1, bias=True)
        if use_spectral_norm:
            block = spectral_norm(block)
        l_layer.append(block)
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            l_layer.append(out_activation)
        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)


class ResBlock1x1(nn.Module):
    def __init__(self, res_dim, n_res_hidden, use_spectral_norm=False, activation='relu'):
        super().__init__()
        if use_spectral_norm:
            self.net = nn.Sequential(get_activation(activation),
                                     spectral_norm(nn.Conv2d(res_dim, n_res_hidden, 1, bias=True)),
                                     get_activation(activation),
                                     spectral_norm(nn.Conv2d(n_res_hidden, res_dim, 1, bias=True)))
        else:
            self.net = nn.Sequential(get_activation(activation),
                                     nn.Conv2d(res_dim, n_res_hidden, 1, bias=True),
                                     get_activation(activation),
                                     nn.Conv2d(n_res_hidden, res_dim, 1, bias=True))

    def forward(self, x):
        return x + self.net(x)


class ConvDenoiser(ConvMLP):
    def __init__(self, in_dim, out_dim, sig, l_hidden=(50,), activation='sigmoid', out_activation='linear',
                 likelihood_type='isotropic_gaussian'):
        super(ConvDenoiser, self).__init__(in_dim, out_dim, l_hidden=l_hidden, activation=activation,
                                           out_activation=out_activation, likelihood_type=likelihood_type)
        self.sig = sig

    def denoise(self, x, stepsize=1.0):
        return self(x)
        # score = (self(x) - x) / (self.sig ** 2)
        # return x + score * stepsize

    def train_step(self, x, opt):
        opt.zero_grad()
        noise = torch.randn_like(x)
        recon = self(x + self.sig * noise)

        loss = torch.mean((recon - x) ** 2)
        loss.backward()
        opt.step()
        return {'loss': loss}

    def add_noise(self, x):
        noise = torch.randn_like(x)
        return x + self.sig * noise


# DCGAN encoder / decoder pair
class DCGANDecoder(LikelihoodDecoder):
    def __init__(self, in_chan, out_chan, ngf=64, likelihood_type='isotropic_gaussian', bias=True):
        super().__init__()
        self.likelihood_type = likelihood_type
        nz = in_chan
        nc = out_chan
        self.in_chan, self.out_chan = in_chan, out_chan
        self.bias = bias
        self.main = [
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=True),
            nn.ConvTranspose2d(     nz, ngf * 8, 2, 1, 0, bias=bias),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=bias),
            # nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward_(self, x):
        for layer in self.main:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_(x)
        x = self.modify_forward_output(x)
        return x


class DCGANDecoderNoBN(LikelihoodDecoder):
    def __init__(self, in_chan, out_chan, ngf=64, likelihood_type='isotropic_gaussian', bias=True, out_activation='linear'):
        super().__init__()
        self.likelihood_type = likelihood_type
        nz = in_chan
        nc = out_chan
        self.in_chan, self.out_chan = in_chan, out_chan
        self.bias = bias
        self.main = [
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=True),
            nn.ConvTranspose2d(     nz, ngf * 8, 2, 1, 0, bias=bias),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=bias),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=bias),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=bias),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=bias),
            # nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ]

        if out_activation == 'sigmoid':
            self.main.append(nn.Sigmoid())
        elif out_activation == 'tanh':
            self.main.append(nn.Tanh())

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward_(self, x):
        for layer in self.main:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_(x)
        x = self.modify_forward_output(x)
        return x


class DCGANEncoder(nn.Module):
    def __init__(self, in_chan, out_chan, ndf=64, out_activation='linear', bias=True):
        nc = in_chan
        nz = out_chan
        self.in_chan, self.out_chan = in_chan, out_chan
        super(DCGANEncoder, self).__init__()
        self.bias = bias
        self.main = [
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=True),
            nn.Conv2d(ndf * 8, nz, 2, 1, 0, bias=bias),
        ]
        self.out_activation = out_activation
        if self.out_activation == 'tanh':
            self.main.append(nn.Tanh())

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


# modified DCGAN encoder / decoder pair
class DCGANDecoder2(nn.Module):
    """smaller model"""
    def __init__(self, in_chan, out_chan, ngf=64):
        nz = in_chan
        nc = out_chan
        self.in_chan, self.out_chan = in_chan, out_chan
        super(DCGANDecoder2, self).__init__()
        self.main = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 4, 5, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=True),
            # nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class DCGANEncoder2(nn.Module):
    def __init__(self, in_chan, out_chan, ndf=64, nh1=64):
        """
        Smaller model 
        nh1: the number of 1x1 feature maps"""
        nc = in_chan
        nz = out_chan
        self.in_chan, self.out_chan = in_chan, out_chan
        super(DCGANEncoder2, self).__init__()
        self.main = [
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 5 x 5
            nn.Conv2d(ndf * 4, nh1, 5, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nh1, nz, 1, 1, 0, bias=True),
        ]

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class DCGANDecoder_Resnet(nn.Module):
    def __init__(self, in_chan, out_chan, ngf=64):
        nz = in_chan
        nc = out_chan
        self.in_chan, self.out_chan = in_chan, out_chan
        super(DCGANDecoder_Resnet, self).__init__()
        self.main = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x



class ConvClassifier(nn.Module):
    def __init__(self, nz, h_dim=128, wasserstein=False):
        super(ConvClassifier, self).__init__()
        if not wasserstein:
            self.main = [
                nn.Conv2d(nz, h_dim, 1),
                nn.ReLU(),
                nn.Conv2d(h_dim, 1, 1),
                nn.Sigmoid(),
                # nn.AdaptiveMaxPool2d(1)
                # nn.AdaptiveAvgPool2d(1)
            ]
        else:
            self.main = [
                nn.Conv2d(nz, h_dim, 1),
                nn.ReLU(),
                nn.Conv2d(h_dim, 1, 1),
                # nn.AdaptiveMaxPool2d(1)
                # nn.AdaptiveAvgPool2d(1)
            ]

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x    
