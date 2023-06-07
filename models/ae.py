"""
ae.py
=====
Autoencoders
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.autograd as autograd
from torchvision.utils import make_grid 
from optimizers import _get_optimizer_instance
from optimizers import get_optimizer


class GlobalConvNet(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(GlobalConvNet, self).__init__()
        self.in_chan, self.out_chan = in_chan, out_chan
        self.main = [
            nn.Conv2d(in_chan, in_chan, 4, 4, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_chan, out_chan, 4, 4, bias=True),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d(1),
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class FullyClassifier(nn.Module):
    def __init__(self, in_chan, h_dim=128):
        super(FullyClassifier, self).__init__()
        self.main = [
            nn.Conv2d(in_chan, h_dim, 1),
            nn.ReLU(),
            nn.Conv2d(h_dim, 1, 1),
            nn.Sigmoid(),
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


def get_dagmm(gpu=True, n_chans=1, backbone='cnn', **kwargs):
    z_dim = kwargs.get('z_dim', 5)
    n_gmm = kwargs.get('n_gmm', 5)
    lambda_energy = kwargs.get('lambda_energy', 0.1)
    lambda_cov_diag = kwargs.get('lambda_cov_diag', 0.005)
    grad_clip = kwargs.get('grad_clip', True)
    rec_cosine = kwargs.get('rec_cosine', True)
    rec_euclidean = kwargs.get('rec_euclidean', True)
    likelihood_type = kwargs.get('likelihood_type', 'isotropic_gaussian')

    if backbone == 'cnn':
        # encoder = ConvNet(in_chan=n_chans, out_chan=z_dim)
        # decoder = DeConvNet(in_chan=z_dim, out_chan=n_chans)
        pass
    elif backbone == 'cnn2':
        nh = kwargs.get('nh', 8)
        n_hidden = kwargs.get('n_hidden', 1024)
        out_activation = kwargs.get('out_activation', None)
        encoder = ConvNet2(in_chan=n_chans, out_chan=z_dim, nh=nh, out_activation=out_activation)
        decoder = DeConvNet2(in_chan=z_dim, out_chan=n_chans, nh=nh, likelihood_type=likelihood_type)
        latent_dim = z_dim
        print(rec_cosine, rec_euclidean)
        if rec_cosine:
            latent_dim += 1
        if rec_euclidean:
            latent_dim += 1

        estimator = FCNet(in_dim=latent_dim, out_dim=n_gmm, l_hidden=(n_hidden,), activation='sigmoid',
                          out_activation='softmax')
    elif backbone == 'cnn3':
        nh = kwargs.get('nh', 8)
        # encoder = ConvNet3(in_chan=n_chans, out_chan=z_dim, nh=nh)
        # decoder = DeConvNet3(in_chan=z_dim, out_chan=n_chans, nh=nh)
    elif backbone == 'dcgan':
        ndf = kwargs.get('ndf', 64)
        ngf = kwargs.get('ngf', 64)
        # encoder = DCGANEncoder(in_chan=n_chans, out_chan=z_dim, ndf=ndf)
        # decoder = DCGANDecoder(in_chan=z_dim, out_chan=n_chans, ngf=ngf)
    elif backbone == 'dcgan2':
        pass
        # encoder = DCGANEncoder2(in_chan=n_chans, out_chan=z_dim)
        # decoder = DCGANDecoder2(in_chan=z_dim, out_chan=n_chans)
    elif backbone == 'FC':  # fully connected
        n_hidden = kwargs.get('n_hidden', 1024)
        # encoder = FCNet(in_dim=n_chans, out_dim=z_dim, l_hidden=(n_hidden,), activation='relu', out_activation='tanh')
        # decoder = FCNet(in_dim=z_dim, out_dim=n_chans, l_hidden=(n_hidden,), activation='relu', out_activation='linear')
    else:
        raise ValueError(f'Invalid argument backbone: {backbone}')

    return DaGMM(encoder, decoder, estimator, gpu=gpu, lambda_energy=lambda_energy, lambda_cov_diag=lambda_cov_diag,
                 grad_clip=grad_clip, rec_cosine=rec_cosine, rec_euclidean=rec_euclidean)


class AE(nn.Module):
    """autoencoder"""
    def __init__(self, encoder, decoder):
        """
        encoder, decoder : neural networks
        """
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.own_optimizer = False

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        z = self.encoder(x)
        return z

    def predict(self, x):
        """one-class anomaly prediction"""
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return predict

    def predict_and_reconstruct(self, x):
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            recon_err = self.decoder.error(x, recon)
        else:
            recon_err = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return recon_err, recon

    def validation_step(self, x, **kwargs):
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        loss = predict.mean()

        if kwargs.get('show_image', True) and len(x.shape) == 4:
            x_img = make_grid(x.detach().cpu(), nrow=10, value_range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, value_range=(0, 1))
        else:
            x_img, recon_img = None, None
        return {'loss': loss.item(), 'predict': predict, 'reconstruction': recon,
                'input@': x_img, 'recon@': recon_img}

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        recon_error = self.predict(x)
        loss = recon_error.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {'loss': loss.item()}

    def reconstruct(self, x):
        return self(x)

    def sample(self, N, z_shape=None, device='cpu'):
        if z_shape is None:
            z_shape = self.encoder.out_shape

        rand_z = torch.rand(N, *z_shape).to(device) * 2 - 1
        sample_x = self.decoder(rand_z)
        return sample_x


class KernelEntropyAE(AE):
    def __init__(self, encoder, decoder, reg=0.0, h=0.5):
        super(KernelEntropyAE, self).__init__(encoder, decoder)
        self.reg = reg
        self.h = h

    def train_step(self, x, optimizer):
        optimizer.zero_grad()

        z = self.encoder(x)
        recon = self.decoder(z)
        recon_loss = ((recon - x) ** 2).mean()
        entropy = self.entropy(z)

        loss = recon_loss - self.reg * entropy
        loss.backward()
        optimizer.step()
        return {'loss': recon_loss, 'entropy': entropy}

    def entropy(self, x):
        """kernel biased estimate of entropy"""
        # x: (n_batch, n_chan, 1, 1)
        x_ = torch.squeeze(torch.squeeze(x, dim=3), dim=2)
        D = x_.shape[1]
        pdist = self.pdist(x_)
        K = torch.exp(-pdist / 2 / self.h**2) / ((np.sqrt(2 * np.pi) * (self.h)) ** D)
        return - torch.mean(torch.log(K.mean(dim=1)))

    def cdist(self, X, Z):
        """pairwise squared euclidean distance"""
        t1 = torch.diag(X.mm(X.t()))[:, None]
        t2 = torch.diag(Z.mm(Z.t()))[:, None]
        t3 = X.mm(Z.t())
        return torch.mm(t1, torch.ones_like(t2.t())) + torch.mm(torch.ones_like(t1), t2.t()) - t3 * 2

    def pdist(self, X):
        """pairwise squared euclidean distance"""
        t3 = X.mm(X.t())
        sq = torch.diag(t3)[:, None]
        t1 = torch.mm(sq, torch.ones_like(sq.t()))
        return t1 + t1.t() - t3 * 2


class DAE(AE):
    """denoising autoencoder"""
    def __init__(self, encoder, decoder, sig=0.0, noise_type='gaussian'):
        super(DAE, self).__init__(encoder, decoder)
        self.sig = sig
        self.noise_type = noise_type

    def train_step(self, x, optimizer, y=None):
        optimizer.zero_grad()
        if self.noise_type == 'gaussian':
            noise = torch.randn(*x.shape, dtype=torch.float32)
            noise = noise.to(x.device)
            recon = self(x + self.sig * noise)
        elif self.noise_type == 'saltnpepper':
            x = self.salt_and_pepper(x)
            recon = self(x)
        else:
            raise ValueError(f'Invalid noise_type: {self.noise_type}')

        loss = torch.mean((recon - x) ** 2)
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}

    def salt_and_pepper(self, img):
        """salt and pepper noise for mnist"""
        # for salt and pepper noise, sig is probability of occurance of noise pixels.
        img = img.copy()
        prob = self.sig
        rnd = torch.random.rand(*img.shape).to(img.device)
        img[rnd < prob / 2] = 0.
        img[rnd > 1 - prob / 2] = 1.
        return img


class NNProjNet(AE):
    """Nearest Neighbor Projection Network"""
    def __init__(self, encoder, decoder, k=1, width=3, sig=0.1):
        super(NNProjNet, self).__init__(encoder, decoder)
        self.k = k
        self.width = width
        self.sig = sig

    def sample_background(self, x):
        r = torch.rand_like(x)
        return (r - 0.5) * self.width

    def train_step(self, x, optimizer):
        optimizer.zero_grad()
        U = self.sample_background(x)
        if len(x.shape) == 4:
            x = x.squeeze(3).squeeze(2)
            U = U.squeeze(3).squeeze(2)

        with torch.no_grad():
            dist = self.pdist(U, x)
            sort_idx = torch.argsort(dist, dim=1)

        l_loss = []
        d_proj_loss = {}
        if self.k == 0:
            d_proj_loss = {'k0': 0}
        for i_k in range(self.k):
            target_x = x[sort_idx[:, i_k]]
            recon = self(U)
            proj_loss = torch.mean((recon - target_x) ** 2)
            d_proj_loss['k' + str(i_k)] = proj_loss.item()
            l_loss.append(proj_loss)

        if self.sig is not None:
            if self.sig == 0:
                recon = self(x)

            else:
                noise = torch.randn_like(x)
                recon = self(x + self.sig * noise)
            denoise_loss = torch.mean((recon - x) ** 2)
            l_loss.append(denoise_loss)
            denoise_loss_ = denoise_loss.item()
        else:
            denoise_loss_ = 0

        loss = torch.mean(torch.stack(l_loss))
        loss.backward()
        optimizer.step()
        d_result = {'loss': loss.item(), 'denoise': denoise_loss_}
        d_result.update(d_proj_loss)
        return d_result

    def pdist(self, X, Y):
        t1 = (X ** 2).sum(dim=1)[:, None]
        t2 = (Y ** 2).sum(dim=1)[None, :]
        t3 = X.mm(Y.t())
        return t1 + t2 - 2 * t3


class VAE(AE):
    def __init__(self, encoder, decoder, n_sample=1, use_mean=False, pred_method='recon', sigma_trainable=False):
        super(VAE, self).__init__(encoder, IsotropicGaussian(decoder, sigma=1, sigma_trainable=sigma_trainable))
        self.n_sample = n_sample  # the number of samples to generate for anomaly detection
        self.use_mean = use_mean  # if True, does not sample from posterior distribution
        self.pred_method = pred_method  # which anomaly score to use
        self.z_shape = None
        
    def forward(self, x):
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        return self.decoder(z_sample)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        if self.use_mean:
            return mu
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + torch.exp(log_sig) * eps

    # def sample_marginal_latent(self, z_shape):
    #     return torch.randn(z_shape)

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x) | p(z))"""
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu_sq = mu ** 2
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        # return 0.5 * torch.mean(kl.view(len(kl), -1), dim=1)
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer, y=None, clip_grad=None):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        nll = - self.decoder.log_likelihood(x, z_sample)

        kl_loss = self.kl_loss(z)
        loss = nll + kl_loss
        loss = loss.mean()
        nll = nll.mean()

        loss.backward()
        optimizer.step()
        return {'loss': nll.item(), 'vae/kl_loss_': kl_loss.mean(), 'vae/sigma_': self.decoder.sigma.item()}

    def predict(self, x):
        """one-class anomaly prediction using the metric specified by self.anomaly_score"""
        if self.pred_method == 'recon':
            return self.reconstruction_probability(x)
        elif self.pred_method == 'lik':
            return  - self.marginal_likelihood(x)  # negative log likelihood
        else:
            raise ValueError(f'{self.pred_method} should be recon or lik')

    def validation_step(self, x, y=None, **kwargs):
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        recon = self.decoder(z_sample)
        loss = torch.mean((recon - x) ** 2)
        predict = - self.decoder.log_likelihood(x, z_sample)
        
        if kwargs.get('show_image', True):
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None
            
        return {'loss': loss.item(), 'predict': predict, 'reconstruction': recon,
                'input@': x_img, 'recon@': recon_img}

    def reconstruction_probability(self, x):
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)

    def marginal_likelihood(self, x, n_sample=None):
        """marginal likelihood from importance sampling
        log P(X) = log \int P(X|Z) * P(Z)/Q(Z|X) * Q(Z|X) dZ"""
        if n_sample is None:
            n_sample = self.n_sample

        # check z shape
        with torch.no_grad():
            z = self.encoder(x)

            l_score = []
            for i in range(n_sample):
                z_sample = self.sample_latent(z)
                log_recon = self.decoder.log_likelihood(x, z_sample)
                log_prior = self.log_prior(z_sample)
                log_posterior = self.log_posterior(z, z_sample)
                l_score.append(log_recon + log_prior - log_posterior)
        score = torch.stack(l_score)
        logN = torch.log(torch.tensor(n_sample, dtype=torch.float, device=x.device))
        return torch.logsumexp(score, dim=0) - logN

    def marginal_likelihood_naive(self, x, n_sample=None):
        if n_sample is None:
            n_sample = self.n_sample

        # check z shape
        z_dummy = self.encoder(x[[0]])
        z = torch.zeros(len(x), *list(z_dummy.shape[1:]), dtype=torch.float).to(x.device)

        l_score = []
        for i in range(n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        score = torch.stack(l_score)
        return - torch.logsumexp(-score, dim=0)

    def elbo(self, x):
        l_score = []
        z = self.encoder(x)
        for i in range(self.n_sample):
            z_sample = self.sample_latent(z)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            kl_loss = self.kl_loss(z)
            score = recon_loss + kl_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)

    def log_posterior(self, z, z_sample):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]

        log_p = torch.distributions.Normal(mu, torch.exp(log_sig)).log_prob(z_sample)
        log_p = log_p.view(len(z), -1).sum(-1)
        return log_p

    def log_prior(self, z_sample):
        log_p = torch.distributions.Normal(torch.zeros_like(z_sample), torch.ones_like(z_sample)).log_prob(z_sample)
        log_p = log_p.view(len(z_sample), -1).sum(-1)
        return log_p

    def posterior_entropy(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        D = mu.shape[1]
        pi = torch.tensor(np.pi, dtype=torch.float32).to(z.device)
        term1 = D / 2
        term2 = D / 2 * torch.log(2 * pi)
        term3 = log_sig.view(len(log_sig), -1).sum(dim=-1)
        return term1 + term2 + term3

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        dummy_z = self.sample_latent(dummy_z)
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    def sample_z(self, n_sample, device):
        z_shape = (n_sample,) + self.z_shape
        return torch.randn(z_shape, device=device, dtype=torch.float)

    def sample(self, n_sample, device):
        z = self.sample_z(n_sample, device)
        return {'sample_x': self.decoder.sample(z)}



class VAE_ConstEnt(VAE):
    def __init__(self, encoder, decoder, n_sample=1, use_mean=False, sig=None):
        super(VAE_ConstEnt, self).__init__(encoder, decoder)
        self.n_sample = n_sample  # the number of samples to generate for anomaly detection
        self.use_mean = use_mean  # if True, does not sample from posterior distribution
        if sig is None:
            self.register_parameter('sig', nn.Parameter(torch.tensor(1, dtype=torch.float)))
        else:
            self.register_buffer('sig', torch.tensor(sig, dtype=torch.float))

    def sample_latent(self, z):
        mu = z
        std = self.sig
        # half_chan = int(z.shape[1] / 2)
        # mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        if self.use_mean:
            return mu
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + std * eps

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians"""
        mu = z
        mu_sq = mu ** 2
        sig_sq = self.sig ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer):
        d_result = super().train_step(x, optimizer)
        d_result['sig'] = self.sig
        return d_result

    def log_posterior(self, z, z_sample):
        mu = z
        log_p = torch.distributions.Normal(mu, self.sig).log_prob(z_sample)
        log_p = log_p.view(len(z), -1).sum(-1)
        return log_p


class VAE_FLOW(VAE):
    def __init__(self, encoder, decoder, flow, n_sample=1, use_mean=False, n_kl_sample=1):
        super(VAE_FLOW, self).__init__(encoder, decoder, n_sample=n_sample, use_mean=use_mean)
        self.flow = flow
        self.n_kl_sample = n_kl_sample

    def kl_loss(self, z):
        l_kl = []
        for i in range(self.n_kl_sample):
            z_sample = self.sample_latent(z)
            log_qz = self.log_posterior(z, z_sample)
            log_pz = self.flow.log_likelihood(z_sample)
            l_kl.append(log_qz - log_pz)
        return torch.stack(l_kl).mean(dim=0)

    def log_prior(self, z_sample):
        return self.flow.log_likelihood(z_sample)


class VAE_SR(VAE):
    """VAE with state reification"""
    def __init__(self, encoder, decoder, denoiser, n_sample=1, use_mean=False):
        super(VAE_SR, self).__init__(encoder, decoder, n_sample=n_sample, use_mean=use_mean)
        self.denoiser = denoiser
        self.own_optimizer = True

    def forward(self, x, denoise=True):
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        z_sample = self.denoiser.denoise(z_sample)
        return self.decoder(z_sample)

    def train_step(self, x, d_opt):
        opt_1, opt_2 = d_opt['dgm'], d_opt['denoiser']

        """Main generative model training"""
        opt_1.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        z_sample = self.denoiser.add_noise(z_sample)
        z_sample = self.denoiser.denoise(z_sample)
        nll = - self.decoder.log_likelihood(x, z_sample).mean()

        kl_loss = self.kl_loss(z)
        loss = nll + kl_loss

        loss.backward()
        opt_1.step()

        """Denoiser training"""
        z_sample = self.sample_latent(z).detach()
        denoise_loss = self.denoiser.train_step(z_sample, opt_2)
        return {'loss': nll.item(), 'kl_loss': kl_loss.item(),
                'denoise_loss': denoise_loss['loss'].item()}

    def get_optimizer(self, opt_cfg):
        # l_dgm_param = list(self.encoder.parameters()) + list(self.decoder.parameters())
        l_dgm_param = self.parameters()
        opt_1 = get_optimizer(opt_cfg['dgm'], l_dgm_param)
        opt_2 = get_optimizer(opt_cfg['denoiser'], self.denoiser.parameters())
        return {'dgm': opt_1, 'denoiser': opt_2}

    def predict(self, x):
        """one-class anomaly prediction"""
        l_score = []
        for i in range(self.n_sample):
            z = self.encoder(x)
            z_sample = self.sample_latent(z)
            # z_sample = self.denoiser.denoise(z_sample)
            recon_loss = - self.decoder.log_likelihood(x, z_sample)
            score = recon_loss
            l_score.append(score)
        return torch.stack(l_score).mean(dim=0)
