import pytest
import torch
from torch.optim import Adam
from models.DiffusionRecoveryLikelihood import v1
from models.DiffusionRecoveryLikelihood import v2
from models.DiffusionRecoveryLikelihood.modules import FCNet_temb, WideResNet_temb2
from models.DiffusionRecoveryLikelihood import spherical


def test_diffusion_recovery_2d_v1():
    x = torch.randn(5,2)
    net = FCNet_temb(in_dim=2, out_dim=1)
    drl = v1.DiffusionRecoveryLikelihood(net, 50, beta_schedule='drl', sampling='gaussian')
    opt = Adam(drl.parameters(), lr=1e-4)
    d_train = drl.train_step(x, opt)
    assert 'loss' in d_train
    drl.eval()
    sampled = drl.q_sample_progressive(x)
    assert sampled.shape == (51, 5, 2)


def test_diffusion_recovery_2d_v2():
    x = torch.randn(5,2)
    net = FCNet_temb(in_dim=2, out_dim=1)
    drl = v2.DiffusionRecoveryLikelihood(net, 50, beta_schedule='drl', sampling='gaussian')
    opt = Adam(drl.parameters(), lr=1e-4)
    d_train = drl.train_step(x, opt)
    assert 'loss' in d_train


def test_get_sigma_schedule_v1():
    beta_schedule = 'drl'
    sigmas, a_s = v1.get_sigma_schedule(beta_schedule, beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=50)
    assert sigmas.shape == (51,)
    assert a_s.shape == (51,)


def test_get_sigma_schedule_v2():
    beta_schedule = 'drl'
    sigmas, a_s = v2.get_sigma_schedule(beta_schedule, beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=50)
    assert sigmas.shape == (51,)
    assert a_s.shape == (51,)


@pytest.mark.parametrize("beta_schedule", ['linear', 'constant'])
@pytest.mark.parametrize("sampling", ['gaussian', 'langevin'])
def test_sphere(beta_schedule, sampling):
    x = torch.randn(5, 2)
    x /= x.norm(dim=1, keepdim=True)

    net = FCNet_temb(in_dim=2, out_dim=1)
    drl = spherical.SphericalDiffusionRecoveryLikelihood(net, 50, beta_schedule=beta_schedule, sampling=sampling)
    opt = Adam(drl.parameters(), lr=1e-4)
    d_train = drl.train_step(x, opt)
    assert 'loss' in d_train

    
@pytest.mark.parametrize("img_sz", [32, 64, 128, 256])
def test_paper_network(img_sz):
    x = torch.rand(2,3,img_sz,img_sz)
    net = WideResNet_temb2(img_sz=img_sz)
    y = net(x, 0)
    assert len(y) == len(x)


def test_paper_network_training():
    x = torch.rand(2,3,32,32)
    n_step = 2
    net = WideResNet_temb2()
    drl = v1.DiffusionRecoveryLikelihood(net, n_step, beta_schedule='drl', sampling='langevin')
    opt = Adam(drl.parameters(), lr=1e-4)
    d_train = drl.train_step(x, opt)
    assert 'loss' in d_train
    sampled = drl.p_sample_progressive(x)
    assert sampled.shape == (n_step+1, 2, 3, 32, 32)
 
