import os

import pytest
import torch
from models import get_model
from models.modules import ConvNet2, DeConvNet2, FCNet
from models.mpdr.idnn import IDNN
from models.mpdr.modules import ConvNet2FC_temb, ResNetMultiScale
from models.mpdr.utils import orthogonalize_batch
from omegaconf import OmegaConf
from torch.optim import Adam


def test_orthogonalize_batch():
    B = 5
    D = 3
    v = torch.diag(torch.ones(D - 1), diagonal=1) + torch.diag(
        torch.ones(D - 1), diagonal=-1
    )
    print(v)
    p = torch.eye(D)
    vv = orthogonalize_batch(v, p)
    assert torch.allclose(vv, v)


def test_load_model_configs():
    cfg = OmegaConf.load("configs_mpdr/mnist_ho_mpdr/z32_r.yml")
    model = get_model(cfg["model"])

    cfg = OmegaConf.load("configs_mpdr/mnist_ho_mpdr/z32_s.yml")
    model = get_model(cfg["model"])


def test_resnet_multiscale():
    x = torch.rand(2, 3, 32, 32, dtype=torch.float)
    net = ResNetMultiScale(
        in_chan=3,
        out_chan=1,
        ch=32,
        use_spectral_norm=True,
        keepdim=True,
        activation="leakyrelu",
        normalize=None,
        out_activation="linear",
        avg_pool_dim=1,
        learn_out_scale=True,
    )
    out = net(x)
    assert out.shape == (2, 1, 1, 1)


def test_idnn():
    zdim = 5
    xdim = 10
    interp_dim_start = 2
    interp_dim_end = 4
    contextdim = xdim - (interp_dim_end - interp_dim_start)
    outdim = interp_dim_end - interp_dim_start
    contextnet = FCNet(in_dim=contextdim, out_dim=zdim, l_hidden=(128,))
    decoder = FCNet(in_dim=zdim, out_dim=outdim, l_hidden=(128,))
    model = IDNN(
        contextnet,
        decoder,
        interp_dim_start=interp_dim_start,
        interp_dim_end=interp_dim_end,
    )

    x = torch.rand(2, 10)
    r = model.recon_error(x)
    opt = Adam(model.parameters(), lr=1e-3)
    model.train_step(x, opt)


def test_v5_mpdr_single():
    from models.mpdr.mpdr import AE, MPDR_Single

    x = torch.randn(2, 10)
    z_dim = 3
    sampling_s = "gaussian"

    # networks
    encoder = FCNet(in_dim=10, out_dim=z_dim, l_hidden=(128,))
    decoder = FCNet(in_dim=z_dim, out_dim=10, l_hidden=(128,))
    ae1 = AE(encoder, decoder, tau=0.1)
    net_x = FCNet(in_dim=10, out_dim=1, l_hidden=(128,))

    mpdr = MPDR_Single(
        ae=ae1,
        net_x=net_x,
        sampling_x="langevin",
        mcmc_n_step_x=2,
        mcmc_stepsize_x=0.01,
        mcmc_noise_x=1,
        proj_mode="uniform",
        proj_noise_start=0.05,
        proj_noise_end=0.2,
        gamma_vx=1.0,
        z_dim=z_dim,
        mcmc_n_step_omi=2,
        mcmc_stepsize_omi=0.01,
        mcmc_noise_omi=0.1,
        mh_omi=True,
    )

    mpdr.predict(x)
    opt = Adam(mpdr.net_x.parameters(), lr=1e-3)
    mpdr.train_step(x, opt, mode="off")

    opt = Adam(mpdr.ae.parameters(), lr=1e-3)
    mpdr.train_step_ae(x, opt)


def test_conditional_mpdr_v5():
    from models.mpdr.mpdr import AE, MPDR_Single
    from models.mpdr.idnn import FCNet_Conditional_Concat

    x = torch.randn(2, 10)
    y = torch.randn(2, 10)
    z_dim = 3

    # networks
    encoder = FCNet(in_dim=10, out_dim=z_dim, l_hidden=(128,))
    decoder = FCNet(in_dim=z_dim, out_dim=10, l_hidden=(128,))
    ae1 = AE(encoder, decoder)
    net_x = FCNet_Conditional_Concat(in_dim=20, out_dim=1, l_hidden=(128,))

    mpdr = MPDR_Single(
        ae=ae1,
        net_x=net_x,
        sampling_x="langevin",
        mcmc_n_step_x=2,
        mcmc_stepsize_x=0.01,
        mcmc_noise_x=1,
        proj_mode="uniform",
        proj_noise_start=0.05,
        proj_noise_end=0.2,
        gamma_vx=1.0,
        z_dim=z_dim,
        mcmc_n_step_omi=2,
        mcmc_stepsize_omi=0.01,
        mcmc_noise_omi=0.1,
        mh_omi=True,
        conditional=True,
    )

    mpdr.predict(x, y=y)
    opt = Adam(mpdr.net_x.parameters(), lr=1e-3)
    mpdr.train_step(x, opt, mode="off", y=y)

    opt = Adam(mpdr.ae.parameters(), lr=1e-3)
    mpdr.train_step_ae(x, opt)


def test_conditional_mpdr_idnn():
    from models.mpdr.mpdr import MPDR_Single, AE
    from models.mpdr.idnn import IDNN

    x_dim = 10
    z_dim = 3
    x = torch.randn(2, x_dim)
    interp_dim_start = 2
    interp_dim_end = 4
    interp_dim = interp_dim_end - interp_dim_start

    # networks
    encoder = FCNet(in_dim=x_dim, out_dim=z_dim, l_hidden=(128,))
    decoder = FCNet(in_dim=z_dim, out_dim=x_dim, l_hidden=(128,))
    ae1 = AE(encoder, decoder)
    contextnet = FCNet(in_dim=x_dim - interp_dim, out_dim=z_dim, l_hidden=(128,))
    decoder = FCNet(in_dim=z_dim, out_dim=interp_dim, l_hidden=(128,))
    net_x = IDNN(
        contextnet,
        decoder,
        interp_dim_start=interp_dim_start,
        interp_dim_end=interp_dim_end,
    )

    mpdr = MPDR_Single(
        ae=ae1,
        net_x=net_x,
        sampling_x="langevin",
        mcmc_n_step_x=2,
        mcmc_stepsize_x=0.01,
        mcmc_noise_x=1,
        proj_mode="uniform",
        proj_noise_start=0.05,
        proj_noise_end=0.2,
        gamma_vx=None,
        gamma_neg_recon=1.0,
        z_dim=z_dim,
        mcmc_n_step_omi=2,
        mcmc_stepsize_omi=0.01,
        mcmc_noise_omi=0.1,
        mh_omi=True,
        conditional=False,
        use_recon_error=False,
    )

    mpdr.predict(x)
    opt = Adam(mpdr.net_x.parameters(), lr=1e-3)
    mpdr.train_step(x, opt, mode="off")

    opt = Adam(mpdr.ae.parameters(), lr=1e-3)
    mpdr.train_step_ae(x, opt)


def test_mpdr_ensemble():
    from models.mpdr.mpdr import AE, MPDR_Ensemble

    x = torch.randn(2, 10)
    z_dim = 3
    sampling_s = "gaussian"

    # networks
    encoder = FCNet(in_dim=10, out_dim=z_dim, l_hidden=(128,))
    decoder = FCNet(in_dim=z_dim, out_dim=10, l_hidden=(128,))
    ae1 = AE(encoder, decoder, tau=0.1)

    encoder = FCNet(in_dim=10, out_dim=z_dim, l_hidden=(128,))
    decoder = FCNet(in_dim=z_dim, out_dim=10, l_hidden=(128,))
    ae2 = AE(encoder, decoder, tau=0.1)

    net_x = FCNet(in_dim=10, out_dim=1, l_hidden=(128,))

    mpdr = MPDR_Ensemble(
        l_ae=[ae1, ae2],
        net_x=net_x,
        sampling_x="langevin",
        mcmc_n_step_x=2,
        mcmc_stepsize_x=0.01,
        mcmc_noise_x=1,
        proj_mode="uniform",
        proj_noise_start=0.05,
        proj_noise_end=0.2,
        gamma_vx=1.0,
        mcmc_n_step_omi=2,
        mcmc_stepsize_omi=0.01,
        mcmc_noise_omi=0.1,
        mh_omi=False,
        improved_cd=True,
    )

    mpdr.predict(x)
    opt = Adam(mpdr.net_x.parameters(), lr=1e-3)
    mpdr.train_step(x, opt, mode="off")
