"""
network architecture that takes time as input
"""
import torch
from models.DiffusionRecoveryLikelihood.modules import (
    FCNet_temb,
    get_timestep_embedding,
)
from models.DiffusionRecoveryLikelihood.modules import WideResNet_temb2
from models.modules import get_activation, get_activation_F, ConvMLP, FCNet, FCResNet
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P


def process_single_t(x, t):
    if isinstance(t, int) or len(t.shape) == 0:
        t = torch.ones([x.shape[0]], dtype=torch.long, device=x.device) * t
    return t


def get_net_temb(in_dim, out_dim, **kwargs):
    arch = kwargs.pop("arch").lower()
    if arch == "fcnet_temb":
        return FCNet_temb(in_dim=in_dim, out_dim=out_dim, **kwargs)
    elif arch == "convnet2fc_temb":
        return ConvNet2FC_temb(in_chan=in_dim, out_chan=out_dim, **kwargs)
    elif arch == "wideresnet_temb2":
        return WideResNet_temb2(in_channels=in_dim, **kwargs)
    else:
        raise NotImplementedError


class ConvNet2FC_temb(nn.Module):
    """additional 1x1 conv layer at the top
    time step positional embedding"""

    def __init__(
        self,
        in_chan=1,
        out_chan=64,
        nh=8,
        nh_mlp=512,
        out_activation="linear",
        use_spectral_norm=False,
        temb_ch=128,
        final_temb="prod",
    ):
        """nh: determines the numbers of conv filters"""
        super().__init__()
        self.temb_ch = temb_ch
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, nh_mlp, kernel_size=4, bias=True)
        self.conv6 = nn.Conv2d(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation)

        if use_spectral_norm:
            self.conv1 = P.spectral_norm(self.conv1)
            self.conv2 = P.spectral_norm(self.conv2)
            self.conv3 = P.spectral_norm(self.conv3)
            self.conv4 = P.spectral_norm(self.conv4)
            self.conv5 = P.spectral_norm(self.conv5)

        self.dense2 = nn.Linear(temb_ch * 4, nh * 8)
        self.dense3 = nn.Linear(temb_ch * 4, nh * 8)
        self.dense4 = nn.Linear(temb_ch * 4, nh * 16)
        self.dense5 = nn.Linear(temb_ch * 4, nh_mlp)

        # timestep embedding
        self.temb_dense_0 = P.spectral_norm(
            nn.Linear(in_features=self.temb_ch, out_features=self.temb_ch * 4)
        )
        self.temb_dense_1 = P.spectral_norm(
            nn.Linear(in_features=self.temb_ch * 4, out_features=self.temb_ch * 4)
        )
        self.temb_dense_2 = nn.Linear(
            in_features=self.temb_ch * 4, out_features=out_chan
        )
        self.final_temb = final_temb
        assert final_temb is None or final_temb in {"prod", "sum"}

    def forward(self, x, t):
        nonlinearity = F.leaky_relu
        temb = get_timestep_embedding(t, self.temb_ch)
        temb = self.temb_dense_0(temb)
        temb = self.temb_dense_1(nonlinearity(temb))
        temb_final = self.temb_dense_2(nonlinearity(temb))
        assert temb.shape == (len(x), self.temb_ch * 4)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x) + self.dense2(temb)[:, :, None, None]
        x = F.relu(x)
        x = self.max1(x)
        x = self.conv3(x) + self.dense3(temb)[:, :, None, None]
        x = F.relu(x)
        x = self.conv4(x) + self.dense4(temb)[:, :, None, None]
        x = F.relu(x)
        x = self.max2(x)
        x = self.conv5(x) + self.dense5(temb)[:, :, None, None]
        x = F.relu(x)
        if self.final_temb == "prod":
            x = self.conv6(x) * temb_final[:, :, None, None]
        elif self.final_temb == "sum":
            x = self.conv6(x) + temb_final[:, :, None, None]
        elif self.final_temb is None:
            x = self.conv6(x)

        if self.out_activation is not None:
            x = self.out_activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        n_class=None,
        downsample=False,
        use_spectral_norm=True,
        activation="relu",
        normalization=None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            3,
            padding=1,
            bias=False if n_class is not None else True,
        )

        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            3,
            padding=1,
            bias=False if n_class is not None else True,
        )

        if use_spectral_norm:
            self.conv1 = P.spectral_norm(self.conv1)
            self.conv2 = P.spectral_norm(self.conv2)

        self.class_embed = None

        if n_class is not None:
            class_embed = nn.Embedding(n_class, out_channel * 2 * 2)
            class_embed.weight.data[:, : out_channel * 2] = 1
            class_embed.weight.data[:, out_channel * 2 :] = 0

            self.class_embed = class_embed

        self.skip = None

        if in_channel != out_channel or downsample:
            if use_spectral_norm:
                self.skip = nn.Sequential(
                    P.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, bias=False))
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 1, bias=False)
                )

        self.downsample = downsample
        self.activation = activation
        self.normalization = normalization

    def forward(self, input, class_id=None):
        out = input

        out = self.conv1(out)

        if self.class_embed is not None:
            embed = self.class_embed(class_id).view(input.shape[0], -1, 1, 1)
            weight1, weight2, bias1, bias2 = embed.chunk(4, 1)
            out = weight1 * out + bias1

        out = get_activation_F(out, self.activation)

        out = self.conv2(out)

        if self.class_embed is not None:
            out = weight2 * out + bias2

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = get_activation_F(out, self.activation)

        return out


class ResNet(nn.Module):
    """Neural Network used in IGEBM
    replace spectral norm implementation
    add learn_out_scaling"""

    def __init__(
        self,
        in_chan=3,
        out_chan=1,
        ch=128,
        n_class=None,
        use_spectral_norm=False,
        keepdim=True,
        activation="leakyrelu",
        normalize=None,
        out_activation="linear",
        avg_pool_dim=1,
        learn_out_scale=False,
        return_feature=False,
    ):
        super().__init__()
        self.keepdim = keepdim
        self.use_spectral_norm = use_spectral_norm
        self.avg_pool_dim = avg_pool_dim
        self.return_feature = return_feature  # only return feature not energy

        if use_spectral_norm:
            self.conv1 = P.spectral_norm(nn.Conv2d(in_chan, ch, 3, padding=1))
        else:
            self.conv1 = nn.Conv2d(in_chan, ch, 3, padding=1)

        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    ch,
                    ch,
                    n_class,
                    downsample=True,
                    use_spectral_norm=use_spectral_norm,
                    activation=activation,
                    normalization=normalize,
                ),
                ResBlock(
                    ch,
                    ch,
                    n_class,
                    use_spectral_norm=use_spectral_norm,
                    activation=activation,
                    normalization=normalize,
                ),
                ResBlock(
                    ch,
                    ch * 2,
                    n_class,
                    downsample=True,
                    use_spectral_norm=use_spectral_norm,
                    activation=activation,
                    normalization=normalize,
                ),
                ResBlock(
                    ch * 2,
                    ch * 2,
                    n_class,
                    use_spectral_norm=use_spectral_norm,
                    activation=activation,
                    normalization=normalize,
                ),
                ResBlock(
                    ch * 2,
                    ch * 2,
                    n_class,
                    downsample=True,
                    use_spectral_norm=use_spectral_norm,
                    activation=activation,
                    normalization=normalize,
                ),
                ResBlock(
                    ch * 2,
                    ch * 2,
                    n_class,
                    use_spectral_norm=use_spectral_norm,
                    activation=activation,
                    normalization=normalize,
                ),
            ]
        )

        if keepdim and not return_feature:
            self.linear = nn.Conv2d(ch * 2, out_chan, 1, 1, 0)
        elif return_feature:
            self.linear = None
        else:
            self.linear = nn.Linear(ch * 2, out_chan)

        self.out_activation = get_activation(out_activation)
        self.pre_activation = None
        self.learn_out_scale = learn_out_scale
        if learn_out_scale and not return_feature:
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

        if self.return_feature:
            return out

        out = self.linear(out)
        if self.learn_out_scale:
            out = self.out_scale(out)
        self.pre_activation = out
        if self.out_activation is not None:
            out = self.out_activation(out)

        return out


class ResNetMultiScale(nn.Module):
    def __init__(
        self,
        in_chan=3,
        out_chan=1,
        ch=128,
        n_class=None,
        use_spectral_norm=False,
        keepdim=True,
        activation="leakyrelu",
        normalize=None,
        out_activation="linear",
        avg_pool_dim=1,
        learn_out_scale=False,
    ):
        super().__init__()
        params = {"in_chan": in_chan, "out_chan": out_chan, "ch": ch, "n_class": n_class, "use_spectral_norm": use_spectral_norm, "keepdim": keepdim, "activation": activation, "normalize": normalize, "out_activation": out_activation, "avg_pool_dim": avg_pool_dim, "learn_out_scale": learn_out_scale}
        self.resnet32 = ResNet(return_feature=True, **params)
        self.resnet16 = ResNet(return_feature=True, **params)
        self.resnet8 = ResNet(return_feature=True, **params)
        self.mlp = ConvMLP(ch * 6, out_chan, l_hidden=(2048, 128), activation=activation, out_activation='linear',
                           spatial_dim=1, fusion_at=0, use_spectral_norm=False)

        self.learn_out_scale = learn_out_scale
        if learn_out_scale:
            self.out_scale = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        x16 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x8 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        feature = torch.cat([self.resnet32(x), self.resnet16(x16), self.resnet8(x8)], dim=1)
        out = self.mlp(feature)
        if self.learn_out_scale:
            out = self.out_scale(out)
        return out


class DCASEEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, l_hidden, frame_embed_dim, res_dim, n_res_hidden, n_resblock, out_activation='linear',
                 activation='relu', n_frame=5):
        super().__init__()
        self.n_frame = n_frame
        frame_dim = in_dim // n_frame

        self.frame_encoder = FCNet(in_dim=frame_dim, out_dim=frame_embed_dim, l_hidden=l_hidden, activation=activation,
                                   out_activation='linear')
        self.resnet = FCResNet(in_dim=frame_embed_dim*5, out_dim=out_dim, res_dim=res_dim, n_res_hidden=n_res_hidden,
                               out_activation=out_activation, n_resblock=n_resblock)

    def forward(self, x):
        batched_x = torch.cat(torch.chunk(x, self.n_frame, dim=1), dim=0)
        batched_xx = self.frame_encoder(batched_x)
        xx = torch.cat(torch.chunk(batched_xx, self.n_frame, dim=0), dim=1)
        z = self.resnet(xx)
        return z


class DCASEDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, l_hidden, frame_embed_dim, res_dim, n_res_hidden, n_resblock, out_activation='linear',
                 activation='relu', n_frame=5):
        super().__init__()
        self.n_frame = n_frame
        frame_dim = out_dim // n_frame

        self.resnet = FCResNet(in_dim=in_dim, out_dim=frame_embed_dim*5, res_dim=res_dim, n_res_hidden=n_res_hidden,
                               out_activation=out_activation, n_resblock=n_resblock)

        self.frame_decoder = FCNet(in_dim=frame_embed_dim, out_dim=frame_dim, l_hidden=l_hidden, activation=activation,
                                   out_activation='linear')

    def forward(self, z):
        zz = self.resnet(z)
        batched_zz = torch.cat(torch.chunk(zz, self.n_frame, dim=1), dim=0)
        batched_x = self.frame_decoder(batched_zz)
        x = torch.cat(torch.chunk(batched_x, self.n_frame, dim=0), dim=1)
        return x
