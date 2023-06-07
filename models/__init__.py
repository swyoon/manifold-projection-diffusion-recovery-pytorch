import os
from omegaconf import OmegaConf
import copy
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from loader import get_dataloader
from augmentations import get_composed_augmentations

from models.ae import (
    AE,
    VAE,
)
from models.mcmc import get_sampler
from models.modules import (
    ConvNet2,
    DeConvNet2,
    FCNet,
    FCResNet,
    ResNet1x1,
    ConvDenoiser,
    DCGANEncoder,
    DCGANDecoder,
    DCGANDecoderNoBN,
    ConvNet3,
    DeConvNet3,
    ConvNet2p,
    ConvNet2FC,
    ConvNet3FC,
    ConvNet3FCBN,
    ConvNet3MLP,
    ConvNet3Att,
    ConvNet3AttV2,
    ConvMLP,
    ConvNet2Att,
    IGEBMEncoder,
    IGEBMEncoderV2,
    ConvNet64,
    DeConvNet64,
)
from models.modules_sngan import Generator as SNGANGeneratorBN
from models.modules_sngan import GeneratorNoBN as SNGANGeneratorNoBN
from models.modules_sngan import GeneratorNoBN64 as SNGANGeneratorNoBN64
from models.modules_sngan import GeneratorGN as SNGANGeneratorGN
from models.energybased import EnergyBasedModel
from models.igebm import IGEBM

def get_net(in_dim, out_dim, **kwargs):
    arch = kwargs.pop("arch")
    nh = kwargs.get("nh", 8)
    out_activation = kwargs.get("out_activation", "linear")

    if arch == "conv2":
        kwargs.pop('nh')
        kwargs.pop('out_activation')
        net = ConvNet2(in_chan=in_dim,
                out_chan=out_dim,
                nh=nh,
                out_activation=out_activation,
                **kwargs)
    elif arch == "conv2p":
        net = ConvNet2p(in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation)
    elif arch == "conv2fc":
        kwargs.pop('nh')
        net = ConvNet2FC(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            **kwargs
        )
    elif arch == "conv2bn":
        from models.modules import ConvNet2BN
        net = ConvNet2BN(in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation)

    elif arch == "conv2att":
        resdim = kwargs["resdim"]
        n_res = kwargs["n_res"]
        ver = kwargs["ver"]
        net = ConvNet2Att(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            resdim=resdim,
            n_res=n_res,
            ver=ver,
            out_activation=out_activation,
        )

    elif arch == "deconv2":
        net = DeConvNet2(
            in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation
        )
    elif arch == "conv64":
        num_groups = kwargs.get("num_groups", None)
        use_bn = kwargs.get("use_bn", False)
        net = ConvNet64(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            out_activation=out_activation,
            num_groups=num_groups,
            use_bn=use_bn,
        )
    elif arch == "deconv64":
        num_groups = kwargs.get("num_groups", None)
        use_bn = kwargs.get("use_bn", False)
        net = DeConvNet64(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            out_activation=out_activation,
            num_groups=num_groups,
            use_bn=use_bn,
        )
    elif arch == "conv3":
        net = ConvNet3(in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation)
    elif arch == "conv3fc":
        kwargs.pop('nh')
        net = ConvNet3FC(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            **kwargs
        )
    elif arch == "conv3fcbn":
        nh_mlp = kwargs["nh_mlp"]
        encoding_range = kwargs.get("encoding_range", None)
        net = ConvNet3FCBN(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            nh_mlp=nh_mlp,
            out_activation=out_activation,
            encoding_range=encoding_range
        )

    elif arch == "conv3mlp":
        l_nh_mlp = kwargs["l_nh_mlp"]
        activation = kwargs["activation"]
        net = ConvNet3MLP(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            l_nh_mlp=l_nh_mlp,
            out_activation=out_activation,
            activation=activation,
        )
    elif arch == "conv3att":
        resdim = kwargs["resdim"]
        n_res = kwargs["n_res"]
        ver = kwargs["ver"]
        activation = kwargs["activation"]
        net = ConvNet3Att(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            resdim=resdim,
            n_res=n_res,
            ver=ver,
            out_activation=out_activation,
            activation=activation,
        )
    elif arch == "conv3attV2":
        resdim = kwargs["resdim"]
        activation = kwargs["activation"]
        spherical = kwargs["spherical"]
        net = ConvNet3AttV2(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            resdim=resdim,
            out_activation=out_activation,
            activation=activation,
            spherical=spherical,
        )

    elif arch == "deconv3":
        num_groups = kwargs.get("num_groups", None)
        net = DeConvNet3(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            out_activation=out_activation,
            num_groups=num_groups,
        )
    elif arch == "fc":
        net = FCNet(
            in_dim=in_dim,
            out_dim=out_dim,
            **kwargs
        )
    elif arch == "fcres":
        net = FCResNet(
                in_dim=in_dim, out_dim=out_dim,
                res_dim=kwargs['resdim'],
                n_res_hidden=kwargs['n_res_hidden'],
                n_resblock=kwargs['n_resblock'],
                out_activation=out_activation,
                use_spectral_norm=kwargs.get('use_spectral_norm', False),
                flatten_input=kwargs.get('flatten_input', False)
                )
    elif arch == "res1x1":
        net = ResNet1x1(in_dim=in_dim, out_dim=out_dim,
                res_dim=kwargs['resdim'],
                n_res_hidden=kwargs['n_res_hidden'],
                n_resblock=kwargs['n_resblock'],
                out_activation=out_activation,
                use_spectral_norm=kwargs.get('use_spectral_norm', False))

    elif arch == "convmlp":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        spatial_dim = kwargs.get('spatial_dim', 1)
        fusion_at = kwargs.get('fusion_at', 0)
        net = ConvMLP(in_dim=in_dim, out_dim=out_dim,
                      l_hidden=l_hidden, activation=activation,
                      out_activation=out_activation,
                      spatial_dim=spatial_dim, fusion_at=fusion_at)
    elif arch == "convdenoiser":
        sig = kwargs["sig"]
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = ConvDenoiser(
            in_dim=in_dim,
            out_dim=out_dim,
            sig=sig,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif arch == "dcgan_encoder":
        bias = kwargs.get("bias", True)
        print(f"DCGAN encoder bias: {bias}")
        net = DCGANEncoder(in_chan=in_dim, out_chan=out_dim, bias=bias)
    elif arch == "dcgan_decoder":
        bias = kwargs.get("bias", True)
        print(f"DCGAN decoder bias: {bias}")
        net = DCGANDecoder(in_chan=in_dim, out_chan=out_dim, bias=bias)
    elif arch == "dcgan_decoder_nobn":
        bias = kwargs.get("bias", True)
        print(f"DCGAN decoder bias: {bias}")
        out_activation = kwargs["out_activation"]
        net = DCGANDecoderNoBN(
            in_chan=in_dim, out_chan=out_dim, bias=bias, out_activation=out_activation
        )
    elif arch == "vqvae_encoder_32_8":
        net = VQVAEEncoder32_8(in_chan=in_dim, out_chan=out_dim)
    elif arch == "vqvae_decoder_32_8":
        net = VQVAEDecoder32_8(in_chan=in_dim, out_chan=out_dim)
    elif arch == "vqvae_encoder_32_4":
        net = VQVAEEncoder32_4(in_chan=in_dim, out_chan=out_dim)
    elif arch == "vqvae_decoder_32_4":
        net = VQVAEDecoder32_4(in_chan=in_dim, out_chan=out_dim, out_activation=kwargs['out_activation'])
    elif arch == "IGEBM":
        net = IGEBM(in_chan=in_dim)
    elif arch == "GPND_E":
        for_mnist = kwargs.get("mnist", False)
        net = GPND_net.Encoder(out_dim, channels=in_dim, mnist=for_mnist)
    elif arch == "GPND_G":
        for_mnist = kwargs.get("mnist", False)
        if for_mnist:
            net = GPND_net.GeneratorMNIST(in_dim, channels=out_dim)
        else:
            net = GPND_net.Generator(in_dim, channels=out_dim)
    elif arch == "GPND_D":
        for_mnist = kwargs.get("mnist", False)
        net = GPND_net.Discriminator(channels=in_dim, mnist=for_mnist)
    elif arch == "GPND_ZD":
        net = GPND_net.ZDiscriminator(in_dim)
    elif arch == "IGEBMEncoder":
        use_spectral_norm = kwargs.get("user_spectral_norm", False)
        keepdim = kwargs.get("keepdim", True)
        out_activation = kwargs.get('out_activation', 'linear')
        net = IGEBMEncoder(
            in_chan=in_dim,
            out_chan=out_dim,
            n_class=None,
            use_spectral_norm=use_spectral_norm,
            keepdim=keepdim,
            out_activation=out_activation,
            avg_pool_dim=kwargs.get('avg_pool_dim', 1)
        )
    elif arch == "IGEBMEncoderV2":
        net = IGEBMEncoderV2(in_chan=in_dim, out_chan=out_dim, n_class=None,
                **kwargs)
    elif arch == "JEMWideResNet":
        depth = kwargs.get("depth", 28)
        width = kwargs.get("width", 10)
        dropout_rate = kwargs.get("dropout_rate", 0.0)
        net = Wide_ResNet(
            depth=depth, widen_factor=width, norm=None, dropout_rate=dropout_rate
        )

    elif arch == "MDDenseNet":
        depth = kwargs.get("depth", 100)
        net = MDDenseNet(depth, out_dim)

    elif arch == "MDResNet34":
        num_classes = out_dim
        net = MDResNet34(num_classes)

    elif arch == "OdinDenseNet":
        depth = kwargs.get("depth", 100)
        net = OdinDenseNet(depth, out_dim, in_dim)

    elif arch == "OdinWideResNet":
        depth = kwargs.get("depth", 28)
        net = OdinWideResNet(depth, out_dim, in_dim)

    elif arch == "sngan_generator_bn":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        net = SNGANGeneratorBN(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
        )
    elif arch == "sngan_generator_nobn":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        net = SNGANGeneratorNoBN(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
        )
    elif arch == "sngan_generator_nobn64":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        net = SNGANGeneratorNoBN64(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
        )
    elif arch == "sngan_generator_gn":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        num_groups = kwargs["num_groups"]
        net = SNGANGeneratorGN(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
            num_groups=num_groups,
            spatial_dim=kwargs.get('spatial_dim', 1)
        )
    elif arch == "wideresnet_temb":
        from models.DiffusionRecoveryLikelihood.modules import WideResNet_temb2
        kwargs.pop('arch')
        net = WideResNet_temb2(in_channels=in_dim, **kwargs)
    elif arch == "avgpool32":
        from models.modules import AvgPoolNet32
        net = AvgPoolNet32(in_chan=in_dim, out_chan=out_dim, **kwargs)
    elif arch == "multiscale_resnet":
        from models.mpdr import ResNetMultiScale
        net = ResNetMultiScale(in_chan=in_dim, out_chan=out_dim, **kwargs)
    elif arch == "dcase_encoder":
        from models.mpdr import DCASEEncoder
        net = DCASEEncoder(in_dim=in_dim, out_dim=out_dim, **kwargs)
    elif arch == "dcase_decoder":
        from models.mpdr import DCASEDecoder
        net = DCASEDecoder(in_dim=in_dim, out_dim=out_dim, **kwargs)
    elif arch == "conv8x8":
        from models.modules import ConvNet8x8
        net = ConvNet8x8(in_chan=in_dim, out_chan=out_dim, **kwargs)
    elif arch == "deconv8x8":
        from models.modules import DeConvNet8x8
        net = DeConvNet8x8(in_chan=in_dim, out_chan=out_dim, **kwargs)
    elif arch == "ae_mpdr_v5":
        net = get_ae_mpdr_v5(**kwargs)
    elif arch == "identity":
        from models.modules import Identity
        net = Identity()

    return net


def get_ae(**model_cfg):
    arch = model_cfg.pop('arch')
    x_dim = model_cfg.pop("x_dim")
    z_dim = model_cfg.pop("z_dim")
    enc_cfg = model_cfg.pop('encoder')
    dec_cfg = model_cfg.pop('decoder')

    if arch == "ae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = AE(encoder, decoder)
    elif arch == "dae":
        sig = model_cfg["sig"]
        noise_type = model_cfg["noise_type"]
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = DAE(encoder, decoder, sig=sig, noise_type=noise_type)
    elif arch == "wae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = WAE(encoder, decoder, **model_cfg)
    elif arch == "em_ae_v1":
        K = model_cfg["K"]
        l_ae = []
        for k in range(K):
            encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
            decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
            ae_ = AE(encoder, decoder)
            l_ae.append(ae_)
        ae = EM_AE_V1(l_ae, **model_cfg["options"])
    elif arch == "vae":
        sigma_trainable = model_cfg.get("sigma_trainable", False)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = VAE(encoder, decoder, **model_cfg)
    elif arch == "vqvae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = VQVAE(encoder, decoder, z_dim, **model_cfg)
    return ae


def get_ae_supervised(**model_cfg):
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]
    y_dim = model_cfg["y_dim"]

    if model_cfg["arch"] == "ae_supervised":
        recon_weight = model_cfg["recon_weight"]
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        predictor = get_net(in_dim=z_dim, out_dim=y_dim, **model_cfg["predictor"])
        ae = SupervisedAE(encoder, decoder, predictor, recon_weight=recon_weight)
    return ae


def get_vae(**model_cfg):
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]
    if model_cfg["arch"] == "vae_cent":
        encoder_out_dim = z_dim
    else:
        encoder_out_dim = z_dim * 2

    encoder = get_net(in_dim=x_dim, out_dim=encoder_out_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
    n_sample = model_cfg.get("n_sample", 1)
    pred_method = model_cfg.get("pred_method", "recon")

    if model_cfg["arch"] == "vae":
        ae = VAE(encoder, decoder, n_sample=n_sample, pred_method=pred_method)
    elif model_cfg["arch"] == "vae_sr":
        denoiser = get_net(in_dim=z_dim, out_dim=z_dim, **model_cfg["denoiser"])
        ae = VAE_SR(encoder, decoder, denoiser)
    elif model_cfg["arch"] == "vae_flow":
        flow_cfg = model_cfg["flow"]
        flow = get_glow(**flow_cfg)
        n_kl_sample = model_cfg.get("n_kl_sample", 1)
        ae = VAE_FLOW(encoder, decoder, flow, n_sample=n_sample, n_kl_sample=n_kl_sample)
    elif model_cfg["arch"] == "vae_proj":
        sample_proj = model_cfg.get("sample_proj", False)
        ae = VAE_PROJ(encoder, decoder, n_sample=n_sample, sample_proj=sample_proj)
    elif model_cfg["arch"] == "vae_cent":
        sig = model_cfg.get("sig", None)
        ae = VAE_ConstEnt(encoder, decoder, n_sample=n_sample, sig=sig)
    return ae


def get_contrastive(**kwargs):
    model_cfg = copy.deepcopy(kwargs["model"])
    x_dim = model_cfg["x_dim"]
    if model_cfg["arch"] == "contrastive_multi":
        x_dim += 1

    net = get_net(in_dim=x_dim, out_dim=1, kwargs=model_cfg["net"])

    if model_cfg["arch"] == "contrastive":
        sigma = model_cfg["sigma"]
        uniform_jitter = model_cfg.get("uniform_jitter", False)
        model = Contrastive(net, sigma=sigma, uniform_jitter=uniform_jitter)
    elif model_cfg["arch"] == "contrastive_v2":
        sigma_1 = model_cfg["sigma_1"]
        sigma_2 = model_cfg["sigma_2"]
        model = ContrastiveV2(net, sigma_1=sigma_1, sigma_2=sigma_2)
    elif model_cfg["arch"] == "contrastive_multi":
        l_sigma = model_cfg["l_sigma"]
        sigma_0 = model_cfg["sigma_0"]
        model = ContrastiveMulti(net, l_sigma=l_sigma, sigma_0=sigma_0)
    return model


def get_glow(**model_cfg):
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    x_dim = model_cfg.pop("x_dim")
    x_size = model_cfg.pop("x_size")
    glow = GlowV2(image_shape=[x_size, x_size, x_dim], **model_cfg)
    return glow


def get_glow_y0ast(**model_cfg):
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    x_dim = model_cfg.pop("x_dim")
    x_size = model_cfg.pop("x_size")
    glow = Glow_y0ast(image_shape=[x_size, x_size, x_dim], **model_cfg)
    return glow


def get_bnaf(**kwargs):
    model_cfg = copy.deepcopy(kwargs["model"])
    model_cfg.pop("arch")
    in_dim = model_cfg["in_dim"]
    model_cfg.pop("in_dim")
    bnaf = BNAF_uniform(data_dim=in_dim, **model_cfg)
    return bnaf


def get_ebm(**model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    in_dim = model_cfg["x_dim"]
    model_cfg.pop("x_dim")
    net = get_net(in_dim=in_dim, out_dim=1, **model_cfg["net"])
    model_cfg.pop("net")
    return EnergyBasedModel(net, **model_cfg)


def get_gatedpixelcnn(**model_cfg):
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    model = GatedPixelCNN(n_classes=0, **model_cfg)
    return model


def get_ebae(**model_cfg):
    arch = model_cfg.pop("arch")
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]

    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])

    if arch == "ebae":
        ae = EBAE_V1(encoder, decoder, **model_cfg["ebae"])
    elif arch == "ebae_v2":
        perturb = get_net(in_dim=z_dim, out_dim=z_dim, **model_cfg["perturb"])
        att = get_net(in_dim=z_dim, out_dim=1, **model_cfg["att"])
        ae = EBAE_V2(encoder, decoder, perturb, att, **model_cfg["ebae"])
    elif arch == "ebae_v3":
        ae = EBAE_V3(encoder, decoder, **model_cfg["ebae"])
    else:
        raise ValueError(f"{arch}")
    return ae


def get_nae(**model_cfg):
    arch = model_cfg.pop("arch")
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]

    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])

    if arch == "nae":
        ae = NAE(encoder, decoder, **model_cfg["nae"])
    else:
        raise ValueError(f"{arch}")
    return ae


def get_nae_cl(**model_cfg):
    arch = model_cfg.pop('arch')
    sampling = model_cfg.pop('sampling')
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']

    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
    if arch == 'nae_cl' and sampling == 'omi':
        sampler_z = get_sampler(**model_cfg['sampler_z'])
        sampler_x = get_sampler(**model_cfg['sampler_x'])
        spatial_dim = model_cfg['decoder'].get('spatial_dim', 1)
        varnet = get_net(in_dim=z_dim * spatial_dim * spatial_dim, out_dim=1, **model_cfg["varnet"])
        nae = NAE_CL_OMI(encoder, decoder, varnet, sampler_z, sampler_x, **model_cfg['nae'])
    elif arch == 'nae_clx' and sampling == 'omi':
        sampler_z = get_sampler(**model_cfg['sampler_z'])
        sampler_x = get_sampler(**model_cfg['sampler_x'])
        varnet = get_net(in_dim=x_dim, out_dim=1, **model_cfg["varnet"])
        nae = NAE_CLX_OMI(encoder, decoder, varnet, sampler_z, sampler_x, **model_cfg['nae'])
    elif arch == 'nae_clzx' and sampling == 'omi':
        sampler_z = get_sampler(**model_cfg['sampler_z'])
        sampler_x = get_sampler(**model_cfg['sampler_x'])
        varnetz = get_net(in_dim=z_dim, out_dim=1, **model_cfg["varnetz"])
        varnetx = get_net(in_dim=x_dim, out_dim=1, **model_cfg["varnetx"])
        nae = NAE_CLZX_OMI(encoder, decoder, varnetz, varnetx, sampler_z, sampler_x, **model_cfg['nae'])

    elif arch == 'nae_l2' and sampling == 'omi':
        sampler_z = get_sampler(**model_cfg['sampler_z'])
        sampler_x = get_sampler(**model_cfg['sampler_x'])
        nae = NAE_L2_OMI(encoder, decoder, sampler_z, sampler_x, **model_cfg['nae'])
    else:
        raise ValueError(f'Invalid sampling: {sampling}')
    return nae


def get_ffebm(**model_cfg):
    from models.nae import FFEBMV2
    model_cfg = copy.deepcopy(model_cfg)
    arch = model_cfg.pop('arch')
    x_dim = model_cfg.pop('x_dim')
    net_cfg = model_cfg.pop('net')
    sampler_cfg = model_cfg.pop('sampler_x')

    net = get_net(in_dim=x_dim, out_dim=1, **net_cfg)
    sampler_x = get_sampler(**sampler_cfg)
    model = FFEBMV2(net, sampler_x, **model_cfg)
    return model


def get_omdrl(**model_cfg):
    from models.mpdr import NAE_OnManifoldDRL 
    from models.mpdr import get_net_temb
    arch = model_cfg.pop('arch')
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg['encoder'])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg['decoder'])
    net_x = get_net(in_dim=x_dim, out_dim=1, **model_cfg['net_x'])
    net_z = get_net_temb(in_dim=z_dim, out_dim=1, **model_cfg['net_z'])

    return NAE_OnManifoldDRL(encoder, decoder, net_x=net_x, net_z=net_z, **model_cfg['omdrl'])


def get_mpdr(**model_cfg):
    from models.mpdr import MPDR, OMPD_V2
    from models.mpdr import get_net_temb
    arch = model_cfg.pop('arch')
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg['encoder'])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg['decoder'])
    net_x = get_net_temb(in_dim=x_dim, out_dim=1, **model_cfg['net_x'])
    net_z = get_net_temb(in_dim=z_dim, out_dim=1, **model_cfg['net_z'])

    if arch == 'mpdr':
        return MPDR(encoder, decoder, net_x=net_x, net_z=net_z, z_dim=z_dim,
                **model_cfg['mpdr'])
    elif arch == 'ompd_v2':
        return OMPD_V2(encoder, decoder, net_x=net_x, net_z=net_z, **model_cfg['ompd'])


def get_mpdr_v4(**model_cfg):
    from models.mpdr.v4_joint import MPDR_Joint, AE
    from models.mpdr import get_net_temb
    model_cfg = copy.deepcopy(model_cfg)
    arch = model_cfg.pop('arch')
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    ae_cfg = model_cfg['ae']
    if 'encoder' in ae_cfg:
        encoder_cfg = ae_cfg.pop('encoder')
        decoder_cfg = ae_cfg.pop('decoder')
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **encoder_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **decoder_cfg)
        ae = AE(encoder, decoder, **ae_cfg)
    elif 'arch' in ae_cfg:
        if ae_cfg['arch'] == 'vaebm':
            from models.vaebm.vaebm import AE
            ae_cfg.pop('arch')
            ae = AE(**ae_cfg)

    if model_cfg.get('net_z', None) is not None:
        net_z = get_net(in_dim=z_dim, out_dim=1, **model_cfg['net_z'])
    else:
        net_z = None
    net_x = get_net(in_dim=x_dim, out_dim=1, **model_cfg['net_x'])
    if model_cfg.get('net_s', None) is not None:
        net_s = get_net_temb(in_dim=z_dim, out_dim=1, **model_cfg['net_s'])
    else:
        net_s = None

    return MPDR_Joint(ae, net_x=net_x, net_z=net_z, net_s=net_s, z_dim=z_dim,
            **model_cfg['mpdr'])

def get_mpdr_v5(**model_cfg):
    from models.mpdr.mpdr import MPDR_Single, AE
    model_cfg = copy.deepcopy(model_cfg)
    arch = model_cfg.pop('arch')
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    ae_cfg = model_cfg['ae']
    if 'encoder' in ae_cfg:
        encoder_cfg = ae_cfg.pop('encoder')
        decoder_cfg = ae_cfg.pop('decoder')
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **encoder_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **decoder_cfg)
        ae = AE(encoder, decoder, net_b=None, **ae_cfg)
    elif 'arch' in ae_cfg:
        if ae_cfg['arch'] == 'vaebm':
            from models.vaebm.vaebm import AE
            ae_cfg.pop('arch')
            ae = AE(**ae_cfg)
        elif ae_cfg['arch'] == 'ldm_ae':
            ae_cfg.pop('arch')
            from models.mpdr.mpdr import LDM_AE
            ae = LDM_AE(**ae_cfg)

    net_x_cfg = model_cfg['net_x']
    if net_x_cfg['arch'] == 'ae':
        net_x = get_ae_mpdr_v5(**net_x_cfg)
        # net_x_cfg.pop('arch')
        # encoder_cfg = net_x_cfg.pop('encoder')
        # decoder_cfg = net_x_cfg.pop('decoder')
        # encoder = get_net(in_dim=x_dim, out_dim=z_dim, **encoder_cfg)
        # decoder = get_net(in_dim=z_dim, out_dim=x_dim, **decoder_cfg)
        # net_b_cfg = net_x_cfg.pop('net_b', None)
        # if net_b_cfg is not None:
        #     net_b = get_net(in_dim=z_dim, out_dim=1, **net_b_cfg)
        #     for m in net_b.modules():
        #         if isinstance(m, nn.Conv2d):
        #             m.weight.data.normal_(0, 0.001)
        #             m.bias.data.zero_()
        # else:
        #     net_b = None
        # net_x = AE(encoder, decoder, net_b=net_b, **net_x_cfg)
    else:
        net_x = get_net(in_dim=x_dim, out_dim=1, **net_x_cfg)

    return MPDR_Single(ae, net_x=net_x,z_dim=z_dim,
            **model_cfg['mpdr'])

def get_mpdr_v5_ensemble(**model_cfg):
    from models.mpdr.mpdr import MPDR_Ensemble, AE
    model_cfg = copy.deepcopy(model_cfg)
    arch = model_cfg.pop('arch')
    x_dim = model_cfg['x_dim']
    ae_cfg = model_cfg['ae']
    l_ae = [get_ae_mpdr_v5(**each_ae_cfg) for each_ae_cfg in ae_cfg]
    net_x_cfg = model_cfg['net_x']
    if net_x_cfg['arch'] == 'ae':
        net_x = get_ae_mpdr_v5(**net_x_cfg)
    elif net_x_cfg['arch'] == 'idnn':
        net_x = get_idnn(**net_x_cfg)
    else:
        net_x = get_net(in_dim=x_dim, out_dim=1, **net_x_cfg)

    return MPDR_Ensemble(l_ae, net_x=net_x, **model_cfg['mpdr'])


def get_ae_mpdr_v5(**model_cfg):
    from models.mpdr.mpdr import AE
    model_cfg = copy.deepcopy(model_cfg)
    arch = model_cfg.pop('arch')
    x_dim = model_cfg.pop('x_dim')
    z_dim = model_cfg.pop('z_dim')

    encoder_cfg = model_cfg.pop('encoder')
    decoder_cfg = model_cfg.pop('decoder')
    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **encoder_cfg)
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **decoder_cfg)
    ae = AE(encoder, decoder, net_b=None, **model_cfg)
    return ae


def get_latent_ae(**model_cfg):
    from models.mpdr.latent import LatentAE
    from models.mpdr.mpdr import AE
    model_cfg = copy.deepcopy(model_cfg)
    arch = model_cfg.pop('arch')
    backbone_name = model_cfg['backbone_name']
    if backbone_name == 'vit_base_patch16_224':
        x_dim = 768  # dim of the representation of the backbone
    ae_cfg = model_cfg.pop('ae')
    if ae_cfg is not None:
        z_dim = model_cfg.pop('z_dim')  # dim of the latent space of the autoencoder
        encoder_cfg = ae_cfg.pop('encoder')
        decoder_cfg = ae_cfg.pop('decoder')
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **encoder_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **decoder_cfg)
        ae = AE(encoder, decoder, net_b=None, **ae_cfg)
    else:
        ae = None
    return LatentAE(ae, 
            **model_cfg)


def get_latent_md(**model_cfg):
    from models.mpdr.md import Mahalanobis
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.pop('arch')
    return Mahalanobis(**model_cfg)


def get_drl(**model_cfg):
    from models.DiffusionRecoveryLikelihood import v1
    from models.DiffusionRecoveryLikelihood.modules import WideResNet_temb2
    from models.mpdr import get_net_temb
    x_dim = model_cfg.pop('x_dim')
    # net = WideResNet_temb2(**model_cfg['net'])
    net = get_net_temb(in_dim=x_dim, out_dim=1, **model_cfg['net'])
    drl = v1.DiffusionRecoveryLikelihood(net, **model_cfg['drl'])
    return drl


def get_idnn(**model_cfg):
    from models.mpdr import IDNN
    model_cfg = copy.deepcopy(model_cfg)
    arch = model_cfg.pop('arch')
    x_dim = model_cfg.pop('x_dim')
    z_dim = model_cfg.pop('z_dim')
    enc_cfg = model_cfg.pop('encoder')
    dec_cfg = model_cfg.pop('decoder')

    # IDNN specific...
    # interpolate dim (open ended range)
    interp_dim_start = model_cfg.pop('interp_dim_start')
    interp_dim_end = model_cfg.pop('interp_dim_end')
    interp_dim = interp_dim_end - interp_dim_start
    # context dim
    c_dim = x_dim - interp_dim
    contextnet = get_net(in_dim=c_dim, out_dim=z_dim, **enc_cfg)
    if 'n_component' in model_cfg:
        from models.mpdr.idnn import GroupMADE
        n_component = model_cfg.pop('n_component')
        decoder = get_net(in_dim=z_dim, out_dim=interp_dim * n_component * 3, **dec_cfg)
        idnn = GroupMADE(
            contextnet=contextnet,
            decoder=decoder,
            interp_dim_start=interp_dim_start,
            interp_dim_end=interp_dim_end,
            n_component=n_component,
            **model_cfg,
        )
        return idnn

    else:
        decoder = get_net(in_dim=z_dim, out_dim=interp_dim, **dec_cfg)
        idnn = IDNN(
            contextnet=contextnet,
            decoder=decoder,
            interp_dim_start=interp_dim_start,
            interp_dim_end=interp_dim_end,
            **model_cfg,
        )
        return idnn


def get_idnn_mpdr(**model_cfg):
    """
        Description: Retrieves conditional version
        of MPDR for IDNN-type models.
    """
    from models.mpdr.mpdr import AE, MPDR_Single
    from models.mpdr import IDNN, IDNN_Plus_AE

    model_cfg = copy.deepcopy(model_cfg)
    # common configs
    arch = model_cfg.pop('arch')
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    ae_cfg = model_cfg['ae']
    # idnn specific

    # build autoencoder
    encoder_cfg = ae_cfg.pop('encoder')
    decoder_cfg = ae_cfg.pop('decoder')
    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **encoder_cfg)
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **decoder_cfg)
    ae = AE(encoder, decoder, **ae_cfg)

    net_x_cfg = model_cfg['net_x']
    if net_x_cfg.arch == 'idnn':
        # build idnn which serves as net_x in MPDR
        net_x = get_idnn(**net_x_cfg)
    elif net_x_cfg.arch == 'idnn_plus_ae':
        # autoencoder
        ae_cfg = net_x_cfg.pop('ae')
        encoder_cfg = ae_cfg.pop('encoder')
        decoder_cfg = ae_cfg.pop('decoder')
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **encoder_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **decoder_cfg)
        ae_ = AE(encoder, decoder, **ae_cfg)

        # idnn
        idnn_cfg = net_x_cfg.pop('idnn')
        idnn = get_idnn(**idnn_cfg)
        net_x = IDNN_Plus_AE(idnn, ae_)

    else:
        # feed-forward network for energy function
        net_x = get_net(in_dim=x_dim, out_dim=1, **net_x_cfg)

    # return mpdr
    return MPDR_Single(
        ae=ae,  # idnn,
        net_x=net_x,
        z_dim=z_dim,
        **model_cfg['mpdr'],
    )

def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    model = _get_model_instance(name)
    model = model(**model_dict, **kwargs)
    return model


def _get_model_instance(name):
    try:
        return {
            "ae": get_ae,
            "dae": get_ae,
            "vae": get_ae,
            "igebm": get_ebm,
            "ffebm": get_ffebm,
            "diffusion_recovery_likelihood": get_drl,
            "omdrl": get_omdrl,
            "mpdr": get_mpdr,
            "ompd_v2": get_mpdr,
            "mpdr_ensemble": get_mpdr_v5_ensemble,
            "mpdr_joint": get_mpdr_v4,
            "mpdr_single": get_mpdr_v5,
            "idnn": get_idnn,
            "idnn_mpdr": get_idnn_mpdr,
            "ae_mpdr_v5": get_ae_mpdr_v5,
            "latent_ae": get_latent_ae,
            "latent_md": get_latent_md,
        }[name]
    except:
        raise ("Model {} not available".format(name))

