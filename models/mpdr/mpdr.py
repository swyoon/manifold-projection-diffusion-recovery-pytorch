"""
Manifold Projection Diffusion Recovery
V5: Leaner implementation
"""
import random
from itertools import chain
from functools import partial
from collections.abc import Iterable

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import make_grid
from omegaconf import OmegaConf

from models.mcmc import sample_langevin_v2, SampleBuffer, sample_langevin_v3
from models.utils import weight_norm
from augmentations import get_composed_augmentations


__all__ = ["AE", "MPDR_Single"]


aug_str = """
          hflip:
            p: 0.5
          randrcrop:
            size: 32
            scale: [0.08, 1.]
            p: 0.2
          cjitter:
            jitter_d: 1.0
            jitter_p: 0.2
          rgray:
            p: 0.2
            """

aug_dict = OmegaConf.create(aug_str)


class AE(nn.Module):
    """autoencoder class"""

    def __init__(
        self,
        encoder,
        decoder,
        *,
        net_b=None,
        spherical=True,
        out_spherical=False,
        l2_norm_reg_enc=None,
        encoding_noise=None,
        eps=1e-6,
        reg_z_norm=None,
        input_noise=None,
        loss="l2",
        tau=0.1,
        perceptual_weight=None,
        condensation_weight=None,
        learn_out_scale=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.net_b = net_b
        self.spherical = spherical
        self.out_spherical = out_spherical
        self.l2_norm_reg_enc = l2_norm_reg_enc
        self.reg_z_norm = reg_z_norm  # for euclidean latent space
        self.encoding_noise = encoding_noise
        self.eps = eps
        self.input_noise = input_noise
        self.loss = loss
        assert loss in {"l2", "l1", "l2_sum"}
        self.tau = tau  # bandwidth of kernel
        self.perceptual_weight = perceptual_weight  # LPIPS loss coef
        self.condensation_weight = condensation_weight

        if perceptual_weight is not None:
            from taming.modules.losses.vqperceptual import LPIPS

            self.perceptual_loss = LPIPS().eval()

        self.learn_out_scale = learn_out_scale
        if learn_out_scale is not None:
            self.register_parameter("out_scale", nn.Parameter(torch.tensor(1.0)))

    def encode(self, x, noise=False, train=False):
        """wrapper for encoder"""
        z = self.encoder(x)
        z_pre = z
        if noise and self.encoding_noise is not None:
            if self.spherical:
                z = self._project(z)
            z = z + torch.randn_like(z) * self.encoding_noise
        if self.spherical:
            z = self._project(z)

        if train:
            return {"z": z, "z_pre": z_pre}
        else:
            return z

    def decode(self, x):
        """wrapper for decoder"""
        if not self.out_spherical:
            return self.decoder(x)
        return self._project(self.decoder(x))

    def _project(self, z):
        """project to unit sphere"""
        return z / (torch.norm(z, dim=1, keepdim=True) + self.eps)

    def recon_error(self, x, noise=False):
        """reconstruction error"""
        z = self.encode(x, noise=noise)
        x_recon = self.decode(z)
        self.z = z
        if self.loss == "l2":
            return ((x - x_recon) ** 2).view(len(x), -1).mean(dim=1)
        elif self.loss == "l2_sum":
            return ((x - x_recon) ** 2).view(len(x), -1).sum(dim=1)
        elif self.loss == "l1":
            return (torch.abs(x - x_recon)).view(len(x), -1).mean(dim=1)

    def forward(self, x):
        recon = self.recon_error(x, noise=False)
        self.state = {"recon": recon}

        if self.net_b is not None:
            b = self.net_b(self.z) ** 2
            self.state["b"] = b
            return recon + b.flatten()
        elif self.learn_out_scale == 'exp':
            return recon * (torch.exp(self.out_scale))
        elif self.learn_out_scale is not None:  # backward compatibility
            return recon * ((self.out_scale)**2)
        else:
            return recon

    def predict(self, x):
        return self(x)

    def project_diffuse(self, x, proj_noise):
        """encode and then apply single step gaussian noise"""
        z = self.encode(x, noise=False)
        z = z + torch.randn_like(z) * proj_noise.view(len(z), *[1] * len(z.shape[1:]))
        if self.spherical:
            z = self._project(z)
        decoded = self.decode(z)
        return decoded, z

    def train_step(self, x, optimizer, clip_grad=None, x_neg=None, mode=None, **kwargs):
        """train autoencoder by minimizing the reconstruction loss"""
        self.train()
        optimizer.zero_grad()
        d_encode = self.encode(x, noise=True, train=True)
        z, z_pre = d_encode["z"], d_encode["z_pre"]
        recon = self.decode(z)
        diff = x - recon

        if self.loss == "l2":
            loss = torch.mean(diff**2)
        elif self.loss == "l1":
            loss = torch.abs(diff).mean()
        elif self.loss == "l2_sum":
            loss = torch.mean((diff ** 2).view(len(x), -1).sum(dim=1))
        recon_error = loss.item()
        d_train = {"loss/recon_error_": recon_error}

        if self.perceptual_weight is not None:
            self.perceptual_loss.eval()
            p_loss = self.perceptual_loss(x * 2 - 1, recon * 2 - 1).mean()
            loss += self.perceptual_weight * p_loss
            d_train["loss/perceptual_loss_"] = p_loss.item()

        encoder_norm, decoder_norm = self.weight_norm()
        if self.l2_norm_reg_enc is not None:
            loss = loss + self.l2_norm_reg_enc * encoder_norm
        d_train["loss/encoder_norm_"] = encoder_norm.item()
        d_train["loss/decoder_norm_"] = decoder_norm.item()

        z_norm = z.view(len(x), -1).norm(dim=1).mean()
        z_pre_sq = (z_pre ** 2).view(len(x), -1).sum(dim=1).mean()
        if self.reg_z_norm is not None:
            loss = loss + self.reg_z_norm * z_pre_sq
        d_train["loss/ae_z_norm_"] = z_norm.item()
        d_train["loss/ae_z_pre_sq_"] = z_pre_sq.item()

        # condensation loss: put z's close to each other
        # pdist = torch.einsum('aijk,bijk->ab', z, z)
        pdist = torch.einsum("a...,b...->ab", z, z)
        pdist = pdist * (1.0 - torch.eye(len(pdist), device=pdist.device))
        d_train["loss/pdist_"] = pdist.mean().item()
        if self.condensation_weight is not None:
            entropy = -torch.logsumexp(pdist / self.tau, dim=1).mean()
            loss = loss + self.condensation_weight * entropy
            d_train["loss/entropy_"] = entropy.item()

        loss.backward()
        optimizer.step()
        d_train["loss/train_loss_ae_"] = loss.item()
        d_train["loss"] = loss.item()

        # for monitoring
        # if len(x.shape) == 4:
        #     img_input = make_grid(x[:16].detach().cpu(), nrow=8, value_range=(0, 1))
        #     img_recon = make_grid(recon[:16].detach().cpu(), nrow=8, value_range=(0, 1))
        #     d_train["ae_train_input@"] = img_input
        #     d_train["ae_train_recon@"] = img_recon

        return d_train

    def validation_step(self, x, y=None):
        """validation step for autoencoder"""
        self.eval()
        recon = self.decode(self.encode(x, noise=False))
        recon_error = self.recon_error(x, noise=False).mean()
        loss = recon_error
        d_result = {"loss/val_recon_error_": recon_error.item()}

        if self.perceptual_weight is not None:
            p_loss = self.perceptual_loss(x * 2 - 1, recon * 2 - 1).mean()
            d_result["loss/val_perceptual_loss_"] = p_loss
            loss += self.perceptual_weight * p_loss

        d_result["loss"] = loss.item()
        # for monitoring the image
        if len(x.shape) == 4 and x.shape[1] == 3:
            img_x = make_grid(x.detach().cpu(), nrow=10, value_range=(0, 1))
            img_recon = make_grid(recon.detach().cpu(), nrow=10, value_range=(0, 1))
            d_result["ae_img_input@"] = img_x
            d_result["ae_img_recon@"] = img_recon
        return d_result

    def weight_norm(self):
        decoder_norm = weight_norm(self.decoder)
        encoder_norm = weight_norm(self.encoder)
        return encoder_norm, decoder_norm

    def reg(self):
        if self.l2_norm_reg_enc is not None:
            encoder_norm, decoder_norm = self.weight_norm()
            return self.l2_norm_reg_enc * encoder_norm
        else:
            return 0


class MPDR_Single(nn.Module):
    """Using ensemble of projection noise and autoencoders"""

    def __init__(
        self,
        ae,
        net_x=None,
        z_dim=None,
        sampling_x="langevin",
        mcmc_n_step_x=None,
        mcmc_stepsize_x=None,
        mcmc_noise_x=None,
        mcmc_bound_x=None,
        mcmc_n_step_omi=None,
        mcmc_stepsize_omi=None,
        mcmc_noise_omi=None,
        mcmc_stepsize_s=None,
        mcmc_noise_s=None,
        mcmc_custom_stepsize=False,
        mcmc_normalize_omi=False,
        temperature=1.0,
        temperature_omi=1.0,
        gamma_vx=None,
        gamma_neg_recon=None,
        gamma_neg_b=None,
        proj_mode="constant",
        proj_noise_start=0.01,
        proj_noise_end=0.01,
        proj_const=1.0,
        proj_const_omi=None,
        proj_dist="sum",
        l2_norm_reg_netx=None,
        energy_method=None,
        eps=1e-6,
        use_net_x=True,
        use_recon_error=False,
        grad_clip_off=None,
        recovery_gaussian_blur=None,
        apply_proj_grad=None,
        replay_ratio=None,
        mh_omi=False,
        mh_x=False,
        return_min=False,
        buffer_match_nearest=False,
        work_on_manifold=False,
        conditional=False,
        custom_netx_reg=False,
    ):

        super().__init__()
        self.ae = ae
        self.net_x = net_x
        self.sampling_x = sampling_x
        self.mcmc_n_step_x = mcmc_n_step_x
        self.mcmc_stepsize_x = mcmc_stepsize_x
        self.mcmc_noise_x = mcmc_noise_x
        self.mcmc_bound_x = mcmc_bound_x
        # on-manifold initialization
        self.mcmc_n_step_omi = mcmc_n_step_omi
        self.mcmc_stepsize_omi = mcmc_stepsize_omi
        self.mcmc_noise_omi = mcmc_noise_omi
        # do not apply stepsize=sigma^2/2
        self.mcmc_custom_stepsize = mcmc_custom_stepsize
        self.mcmc_normalize_omi = mcmc_normalize_omi
        self.temperature = temperature
        self.temperature_omi = temperature_omi
        self.gamma_vx = gamma_vx
        self.gamma_neg_recon = gamma_neg_recon
        self.gamma_neg_b = gamma_neg_b
        self.proj_mode = proj_mode
        self.proj_noise_start = proj_noise_start
        self.proj_noise_end = proj_noise_end
        self.proj_const = proj_const
        self.proj_dist = proj_dist
        self.l2_norm_reg_netx = l2_norm_reg_netx
        self.energy_method = energy_method
        self.eps = eps
        self.z_dim = z_dim
        self.use_net_x = use_net_x
        self.use_recon_error = use_recon_error
        if use_recon_error:
            # register two scalar parameters, a and b
            self.register_parameter("a", nn.Parameter(torch.tensor(1.0)))
            self.register_parameter("b", nn.Parameter(torch.tensor(0.0)))

        self.grad_clip_off = grad_clip_off
        # parameters for gaussian blur of recovery gradient
        # kernel_size, sigma
        self.recovery_gaussian_blur = recovery_gaussian_blur
        self.apply_proj_grad = (
            mcmc_n_step_x if apply_proj_grad is None else apply_proj_grad
        )
        self.proj_const_omi = proj_const_omi
        self.replay_ratio = replay_ratio
        if replay_ratio is not None:
            self.buffer = SampleBuffer(max_samples=10000)
        self.mh_omi = mh_omi
        self.mh_x = mh_x
        self.return_min = return_min
        self.buffer_match_nearest = buffer_match_nearest
        self.work_on_manifold = work_on_manifold
        self.conditional = conditional
        self.custom_netx_reg = custom_netx_reg

    def _project(self, x):
        "TODO: update latent space diffusion"
        return x / (x.norm(dim=1, keepdim=True) + self.eps)

    def encode(self, x):
        return self.ae.encode(x)

    def decode(self, z):
        return self.ae.decode(z)

    def vx(self, x, y=None):
        if self.conditional:
            return self.net_x(x, y).flatten()
        else:
            return self.net_x(x).flatten()

    def forward(self, x, y=None):
        self.eval()
        if self.work_on_manifold:
            with torch.no_grad():
                x = self.decode(self.encode(x))
        return self.energy_x(
            x, y=y, train=False
        )

    def predict(self, x, y=None):
        return self.forward(x, y=y)

    def energy_x(self, x, y=None, train=False):
        vx = self.vx(x, y=y)
        d_out = {"vx": vx}

        if self.energy_method == "exp":
            energy = torch.exp(vx)
        else:
            energy = vx

        if self.use_recon_error:
            recon_error = self.ae.recon_error(x, noise=False)
            # energy = energy + F.relu((self.a**2) * (recon_error - self.b))
            energy = energy + (self.a**2) * (recon_error - self.b)
            d_out["recon_error"] = recon_error

        if hasattr(self.net_x, 'state'):
            # save energy-function specific state for regularization
            d_out = {**d_out, **self.net_x.state}
            # self.net_x.state = None

        if train:
            d_out["energy"] = energy
            return d_out
        else:
            return energy

    def train_step_ae(self, x, optimizer, clip_grad=None):
        """train all autoencoders in the list"""
        return self.ae.train_step(x, optimizer)

    def validation_step_ae(self, x, y=None):
        """validation step for all autoencoders in the list"""
        return self.ae.validation_step(x, y=y)

    def get_proj_noise(self, x):
        n_sample = len(x)
        device = x.device
        if self.proj_mode == "uniform":
            proj_noise = (
                torch.rand(n_sample, device=device)
                * (self.proj_noise_start - self.proj_noise_end)
                + self.proj_noise_end
            )
        elif self.proj_mode == "constant":
            proj_noise = self.proj_noise_start * torch.ones(n_sample, device=device)
        else:
            raise NotImplementedError

        return proj_noise

    def train_step(self, x, optimizer, clip_grad=None, y=None, mode="off"):
        """Off-manifold projection recovery likelihood
        mode: 'off' for off-manifold training energy function
              'on' for on-manifold training for sampling
        """
        # if mode == "off":
        if self.work_on_manifold:
            with torch.no_grad():
                x = self.decode(self.encode(x))
        d_train = self.train_step_off(x, optimizer, y=y)
        # else:
        #     raise ValueError("mode should be off: {}".format(mode))
        return d_train

    def train_step_off(self, x, optimizer, y=None):
        """off-manifold training for learning energy function"""
        # from pudb import set_trace; set_trace()
        self.eval()
        if self.sampling_x == "langevin":
            with torch.no_grad():
                proj_noise = self.get_proj_noise(x)
                x_init_pre, z_obs = self.ae.project_diffuse(x, proj_noise)

            d_omi = self.on_manifold_init_v2(x_init_pre, z_obs, proj_noise, y=y)
            x_init = d_omi["x_init"]

            d_sample = self.p_sample_langevin_off_v2(
                ae=self.ae, x_init=x_init, z_obs=z_obs, proj_noise=proj_noise, y=y
            )
            x_neg = d_sample["sample"]
        elif self.sampling_x is None:
            x_neg = None
            x_init = None
        else:
            raise NotImplementedError
        d_off = {"x_neg": x_neg, "x_init": x_init, "d_sample": d_sample}

        d_train = {**d_off, **d_omi}

        """ for visualization"""
        if len(x.shape) == 4 and x_neg is not None and x.shape[1] in {1, 3}:
            x_neg_off_img = make_grid(x_neg.detach().cpu(), nrow=10, value_range=(0, 1))
            x_img = make_grid(x.detach().cpu(), nrow=10, value_range=(0, 1))
            x_init_img = make_grid(x_init.detach().cpu(), nrow=10, value_range=(0, 1))
            x_init_pre_img = make_grid(
                x_init_pre.detach().cpu(), nrow=10, value_range=(0, 1)
            )
            d_train["input@"] = x_img
            d_train["x_init_before@"] = x_init_pre_img
            d_train["x_init_after@"] = x_init_img
            d_train["x_neg_off@"] = x_neg_off_img

        # MCMC diagnostics
        if x_neg is not None:
            d_train["mcmc/off_before_dist_"] = ((x - x_init) ** 2).mean()
            d_train["mcmc/off_after_dist_"] = ((x - x_neg[: len(x)]) ** 2).mean()
            d_train["mpdr/off_avg_grad_norm_"] = d_sample["avg_grad_norm"].item()

        # for monitoring
        with torch.no_grad():
            z_x_neg = self.ae.encode(x_neg[: len(x)])
        d_train["mpdr/z_inner_"] = (z_x_neg * z_obs).sum(dim=1).mean().item()
        d_loss = self.update_model_off(x.detach(), x_neg.detach(), optimizer, y=y)

        return {**d_train, **d_loss}

    def update_model_off(self, x, x_neg, optimizer, y=None):
        if self.net_x is not None:
            self.net_x.train()
        optimizer.zero_grad()
        # projection recovery
        if x_neg is not None:
            d_off_neg = self.energy_x(x_neg, train=True, y=y)
            d_off_pos = self.energy_x(x, train=True, y=y)
            off_neg_e, vx_neg = (
                d_off_neg["energy"],
                d_off_neg["vx"],
            )
            off_pos_e, vx_pos = (
                d_off_pos["energy"],
                d_off_pos["vx"],
            )
            loss_off = off_pos_e.mean() - off_neg_e.mean()
        else:
            loss_off = torch.tensor(0.0, device=x.device)
            off_pos_e = torch.tensor(0.0, device=x.device)
            off_neg_e = torch.tensor(0.0, device=x.device)
            vx_pos = torch.tensor(0.0, device=x.device)
            vx_neg = torch.tensor(0.0, device=x.device)

        loss = loss_off

        # reguarlization
        d_reg = {}
        netx_norm = self.weight_norm()
        if self.l2_norm_reg_netx is not None:
            loss = loss + self.l2_norm_reg_netx * netx_norm

        reg_gamma_vx = (vx_pos**2).mean() + (vx_neg**2).mean()
        if self.gamma_vx is not None:
            loss += reg_gamma_vx * self.gamma_vx

        if self.gamma_neg_recon is not None:
            loss += self.gamma_neg_recon * (d_off_neg["recon"] ** 2).mean()
                
        if self.gamma_neg_b is not None:
            loss += self.gamma_neg_b * (d_off_neg["b"] ** 2).mean()

        if self.custom_netx_reg:
            loss += self.net_x.reg()

        loss.backward()
        optimizer.step()
        d_train = {
            "loss": loss.item(),
            "mpdr/loss_off_": loss_off.item(),
            "mpdr/off_pos_e_": off_pos_e.mean().item(),
            "mpdr/off_neg_e_": off_neg_e.mean().item(),
            "reg/reg_gamma_vx_": reg_gamma_vx.item(),
            "reg/netx_norm_": netx_norm.item(),
        }
        if self.use_recon_error:
            d_train["recon_error/recon_error_pos_"] = (
                d_off_pos["recon_error"].mean().item()
            )
            d_train["recon_error/recon_error_neg_"] = (
                d_off_neg["recon_error"].mean().item()
            )
            d_train["recon_error/a_"] = self.a.item()
            d_train["recon_error/b_"] = self.b.item()

        if hasattr(self.net_x, 'out_scale'):
            d_train['mpdr/out_scale_'] = self.net_x.out_scale.item()
        d_train = {**d_train, **d_reg}
        return d_train

    def log_prob(self, x, ae, z0, proj_noise, y=None, proj_const=None):
        if proj_const is None:
            proj_const = self.proj_const

        e = self.energy_x(x, y=y)
        if z0 is None:  # test-time sampling mode
            recov = 0.0
        elif self.proj_dist == "geodesic":
            dot = (self.encode(x) * z0).sum(dim=1, keepdim=True)
            eps = 1e-6
            recov = torch.acos(dot - eps) ** 2 / 2 / proj_noise**2
        elif self.proj_dist == "sum":
            recov = (
                ((ae.encode(x) - z0) ** 2).view(len(x), -1).sum(dim=1)
                / 2
                / proj_noise**2
            )
        return (
            -e / self.temperature - recov.view(len(x), -1).sum(dim=1) * self.proj_const
        )

    def init_sample_z(self, z_obs):
        """initialize initial z samples from buffer or from uniform"""
        if self.replay_ratio is None:
            return z_obs.clone().detach()

        n_sample = len(z_obs)
        if len(self.buffer) == 0:
            n_replay = 0
            sample = torch.tensor([])
        else:
            n_replay = (np.random.rand(n_sample) < self.replay_ratio).sum()
            sample = self.buffer.get(n_replay)
        new_sample = self._project(torch.randn(n_sample - n_replay, *z_obs.shape[1:]))
        sample = torch.cat([sample, new_sample]).to(z_obs.device)
        if self.buffer_match_nearest:
            # reorder sample to match their nearest neighbor (dot product) in z_obs
            dot = torch.einsum("iklm,jklm->ij", sample, z_obs)
            _, idx = torch.sort(dot, dim=1, descending=True)
            max_dot = dot[torch.arange(len(dot)), idx[:, 0]]
            _, idx = torch.sort(max_dot, dim=0, descending=True)
            sample = sample[idx]
            assert len(sample) == n_sample

        return sample.to(z_obs.device).detach()

    def on_manifold_init_v2(self, x_init, z_obs, proj_noise, y=None):
        z = self.init_sample_z(z_obs)
        if self.mcmc_n_step_omi is None:
            return {"x_init": x_init}

        proj_noise = proj_noise.reshape(len(proj_noise), *[1] * len(z.shape[1:]))
        n_step = self.mcmc_n_step_omi
        step_size = self.mcmc_stepsize_omi
        noise_scale = self.mcmc_noise_omi

        # randomize omi step
        # if isinstance(n_step, Iterable):
        #     n_step = np.random.choice(n_step)

        f = partial(self.log_prob, ae=self.ae, z0=z_obs, proj_noise=proj_noise, y=y, proj_const=self.proj_const_omi)
        z0 = z.clone().detach()
        d_mcmc = sample_langevin_v2(
            z0,
            lambda zz: -f(self.decode(zz)),
            stepsize=step_size,
            n_step=n_step,
            noise_scale=noise_scale,
            mh=self.mh_omi,
            temperature=self.temperature_omi,
            normalize_grad=self.mcmc_normalize_omi,
            bound="spherical" if self.ae.spherical else None,
        )

        d_sample = {"x_init": self.decode(d_mcmc["sample"]), "d_mcmc": d_mcmc}
        # monitoring
        if n_step > 0:
            omi_energy_grad = (
                torch.stack(d_mcmc["l_drift"]).pow(2).sum(dim=1).sqrt().mean()
            )
            # omi_recov_grad = grad_recov.pow(2).sum(dim=1).sqrt().mean()
            d_sample["mpdr/omi_energy_grad_norm_"] = omi_energy_grad.item()
            # d_sample["mpdr/omi_recov_grad_norm_"] = omi_recov_grad.item()
            d_sample["mcmc/omi_energy_"] = d_mcmc["l_E"][-1].mean().item()
            if self.mh_omi:
                d_sample["mcmc/omi_accept_"] = (
                    torch.cat(d_mcmc["l_accept"]).float().mean().item()
                )
            # d_sample["mcmc/omi_recov_"] = l_recov[-1].mean().item()
            # d_sample["mcmc/omi_energy_tot_"] = l_E_tot[-1].mean().item()
        return d_sample
    
    def clip_vector_norm(self, x, max_norm):
        norm = x.norm(dim=1, keepdim=True)
        x = x * (
            (norm < max_norm).to(torch.float)
            + (norm > max_norm).to(torch.float) * max_norm / norm
            + 1e-6
        )
        return x

    def p_sample_langevin_off_v2(self, *, ae, x_init, z_obs, proj_noise, y=None):
        if z_obs is not None:
            proj_noise = proj_noise.reshape(
                len(proj_noise), *[1] * len(z_obs.shape[1:])
            )
        noise_scale = self.mcmc_noise_x
        if self.mcmc_custom_stepsize:
            # ignore following the original langevin dynamics parameter (stepsize = sigma^2/2)
            step_size = self.mcmc_stepsize_x
        else:
            step_size = self.mcmc_stepsize_x * (noise_scale**2) / 2

        f = partial(self.log_prob, ae=ae, z0=z_obs, proj_noise=proj_noise, y=y)
        x0 = x_init.clone().detach()
        # d_sample = sample_langevin_v2(x0, self.energy_x,
        d_sample = sample_langevin_v2(
            x0,
            lambda xx: -f(xx),
            stepsize=step_size,
            n_step=self.mcmc_n_step_x,
            noise_scale=noise_scale,
            bound=self.mcmc_bound_x,
            mh=self.mh_x,
            temperature=None,
        )  # temperature is already reflected
        d_result = {
            "sample": d_sample["sample"],
            "l_grad_y": d_sample["l_drift"],
            "l_sample": d_sample["l_sample"],
            "mcmc_n_step_x": self.mcmc_n_step_x,
            "d_sample": d_sample,
        }
        l_grad_y = d_sample["l_drift"]
        if len(l_grad_y) > 0:
            d_result["avg_grad_norm"] = (
                torch.stack(l_grad_y).reshape(len(l_grad_y), -1).norm(dim=1).mean()
            )
        else:
            d_result["avg_grad_norm"] = torch.tensor(0)

        return d_result

    def weight_norm(self):
        netx_norm = weight_norm(self.net_x)
        return netx_norm

    def sample_step(self, x, **kwargs):
        # generate sample from the current model
        d_result = {"loss": 0.0}

        d_sample = self.sample(len(x), x.device, image_data=len(x.shape) == 4)

        if len(x.shape) == 4:
            x_sample = d_sample["sample"]
            x_zstart = self.ae.decode(d_sample["l_z_sample"][-1])
            x_zend = self.ae.decode(d_sample["z_sample"])
            x_zstart = make_grid(x_zstart, nrow=8, value_range=(0, 1))
            x_zend = make_grid(x_zend, nrow=8, value_range=(0, 1))
            x_xend = make_grid(x_sample, nrow=8, value_range=(0, 1))
            d_result["sample_z_start@"] = x_zstart
            d_result["sample_z_end@"] = x_zend
            d_result["sample_x_end@"] = x_xend
        d_result = {**d_result, **d_sample}
        return d_result


class MPDR_Ensemble(nn.Module):
    """Using ensemble of projection noise and autoencoders"""

    def __init__(
        self,
        l_ae,
        net_x=None,
        sampling_x="langevin",
        mcmc_n_step_x=None,
        mcmc_stepsize_x=None,
        mcmc_noise_x=None,
        mcmc_bound_x=None,
        mcmc_n_step_omi=None,
        mcmc_stepsize_omi=None,
        mcmc_noise_omi=None,
        mcmc_stepsize_s=None,
        mcmc_noise_s=None,
        mcmc_custom_stepsize=False,
        mcmc_normalize_omi=False,
        temperature=1.0,
        temperature_omi=1.0,
        gamma_vx=None,
        gamma_neg_recon=None,
        gamma_neg_b=None,
        proj_mode="constant",
        proj_noise_start=0.01,
        proj_noise_end=0.01,
        proj_const=1.0,
        proj_const_omi=None,
        proj_dist="sum",
        l2_norm_reg_netx=None,
        energy_method=None,
        eps=1e-6,
        use_net_x=True,
        use_recon_error=False,
        grad_clip_off=None,
        recovery_gaussian_blur=None,
        apply_proj_grad=None,
        replay_ratio=None,
        mh_omi=False,
        mh_x=False,
        return_min=False,
        buffer_match_nearest=False,
        work_on_manifold=False,
        conditional=False,
        custom_netx_reg=False,
        data_aug_pre=False,
        data_aug_post=False,
        data_aug_buffer=False,
        improved_cd=False,
        n_replay_x=None,
    ):

        super().__init__()
        self.l_ae = nn.ModuleList(l_ae)
        self.net_x = net_x
        self.sampling_x = sampling_x
        self.mcmc_n_step_x = mcmc_n_step_x
        self.mcmc_stepsize_x = mcmc_stepsize_x
        self.mcmc_noise_x = mcmc_noise_x
        self.mcmc_bound_x = mcmc_bound_x
        # on-manifold initialization
        self.mcmc_n_step_omi = mcmc_n_step_omi
        self.mcmc_stepsize_omi = mcmc_stepsize_omi
        self.mcmc_noise_omi = mcmc_noise_omi
        # do not apply stepsize=sigma^2/2
        self.mcmc_custom_stepsize = mcmc_custom_stepsize
        self.mcmc_normalize_omi = mcmc_normalize_omi
        self.temperature = temperature
        self.temperature_omi = temperature_omi
        self.gamma_vx = gamma_vx
        self.gamma_neg_recon = gamma_neg_recon
        self.gamma_neg_b = gamma_neg_b
        self.proj_mode = proj_mode
        self.proj_noise_start = proj_noise_start
        self.proj_noise_end = proj_noise_end
        self.proj_const = proj_const
        self.proj_dist = proj_dist
        self.l2_norm_reg_netx = l2_norm_reg_netx
        self.energy_method = energy_method
        self.eps = eps
        self.use_net_x = use_net_x
        self.use_recon_error = use_recon_error
        if use_recon_error:
            # register two scalar parameters, a and b
            self.register_parameter("a", nn.Parameter(torch.tensor(1.0)))
            self.register_parameter("b", nn.Parameter(torch.tensor(0.0)))

        self.grad_clip_off = grad_clip_off
        # parameters for gaussian blur of recovery gradient
        # kernel_size, sigma
        self.recovery_gaussian_blur = recovery_gaussian_blur
        self.apply_proj_grad = (
            mcmc_n_step_x if apply_proj_grad is None else apply_proj_grad
        )
        self.proj_const_omi = proj_const_omi
        self.replay_ratio = replay_ratio
        self.mh_omi = mh_omi
        self.mh_x = mh_x
        self.return_min = return_min
        self.buffer_match_nearest = buffer_match_nearest
        self.work_on_manifold = work_on_manifold
        self.conditional = conditional
        self.custom_netx_reg = custom_netx_reg
        self.data_aug_pre = data_aug_pre
        self.data_aug_post = data_aug_post
        self.data_aug_buffer = data_aug_buffer
        if data_aug_pre or data_aug_post or data_aug_buffer:
            self.data_aug_fn = get_composed_augmentations(aug_dict)
        self.improved_cd = improved_cd
        self.n_replay_x = n_replay_x 
        if n_replay_x is not None:
            self.buffer = SampleBuffer(max_samples=10000)

    def _project(self, x):
        "TODO: update latent space diffusion"
        return x / (x.norm(dim=1, keepdim=True) + self.eps)

    def encode(self, x):
        return self.ae.encode(x)

    def decode(self, z):
        return self.ae.decode(z)

    def vx(self, x, y=None):
        if self.conditional:
            return self.net_x(x, y).flatten()
        else:
            return self.net_x(x).flatten()

    def forward(self, x, y=None):
        return self.energy_x(
            x, y=y, train=False
        )

    def predict(self, x, y=None):
        return self.forward(x, y=y)

    def energy_x(self, x, y=None, train=False):
        vx = self.vx(x, y=y)
        d_out = {"vx": vx}

        if self.energy_method == "exp":
            energy = torch.exp(vx)
        else:
            energy = vx

        if self.use_recon_error:
            recon_error = self.ae.recon_error(x, noise=False)
            # energy = energy + F.relu((self.a**2) * (recon_error - self.b))
            energy = energy + (self.a**2) * (recon_error - self.b)
            d_out["recon_error"] = recon_error

        if hasattr(self.net_x, 'state'):
            # save energy-function specific state for regularization
            d_out = {**d_out, **self.net_x.state}
            # self.net_x.state = None

        if train:
            d_out["energy"] = energy
            return d_out
        else:
            return energy

    def train_step_ae(self, x, optimizer, clip_grad=None):
        """train all autoencoders in the list"""
        raise NotImplementedError
        return self.ae.train_step(x, optimizer)

    def validation_step_ae(self, x, y=None):
        """validation step for all autoencoders in the list"""
        raise NotImplementedError
        return self.ae.validation_step(x, y=y)

    def get_proj_noise(self, x):
        n_sample = len(x)
        device = x.device
        if self.proj_mode == "uniform":
            proj_noise = (
                torch.rand(n_sample, device=device)
                * (self.proj_noise_start - self.proj_noise_end)
                + self.proj_noise_end
            )
        elif self.proj_mode == "constant":
            proj_noise = self.proj_noise_start * torch.ones(n_sample, device=device)
        else:
            raise NotImplementedError

        return proj_noise

    def train_step(self, x, optimizer, clip_grad=None, y=None, mode="off"):
        """Off-manifold projection recovery likelihood
        mode: 'off' for off-manifold training energy function
              'on' for on-manifold training for sampling
        """
        d_train = self.train_step_off(x, optimizer, y=y)
        return d_train

    def train_step_off(self, x, optimizer, y=None):
        """off-manifold training for learning energy function"""
        self.eval()
        l_x_neg = []
        l_x_init = []
        l_x_init_pre = []
        l_x_through = []
        n_split = len(x) // len(self.l_ae)
        for i, ae in enumerate(self.l_ae):
            xx = x[i*n_split:(i+1)*n_split]
            with torch.no_grad():
                proj_noise = self.get_proj_noise(xx)
                # data aug pre
                if self.data_aug_pre:
                    xx = torch.stack([self.data_aug_fn(xxx) for xxx in xx])
                x_init_pre, z_obs = ae.project_diffuse(xx, proj_noise)

            d_omi = self.on_manifold_init_v2(x_init_pre, z_obs, proj_noise, y=y, ae=ae)
            x_init = d_omi["x_init"]
            
            ## off-manifold replay buffer
            if self.n_replay_x is not None:
                x_buffer = self.buffer.get(self.n_replay_x).to(x.device)

                if len(x_buffer) > 0 and self.data_aug_buffer:
                    # data aug post
                    x_buffer = torch.stack([self.data_aug_fn(xxx) for xxx in x_buffer])

                if len(x_buffer) > 0:
                    x_init = torch.cat([x_init, x_buffer], dim=0)
                    z_obs = torch.cat([z_obs, z_obs[:self.n_replay_x]], dim=0)
                    proj_noise = torch.cat([proj_noise, proj_noise[:self.n_replay_x]], dim=0)

            if self.data_aug_post:
                # data aug post
                x_init = torch.stack([self.data_aug_fn(xxx) for xxx in x_init])

            d_sample = self.p_sample_langevin_off_v2(
                ae=ae, x_init=x_init, z_obs=z_obs, proj_noise=proj_noise, y=y
            )
            x_neg = d_sample["sample"]
            l_x_neg.append(x_neg)
            l_x_init.append(x_init)
            l_x_init_pre.append(x_init_pre)
            if self.improved_cd:
                l_x_through.append(d_sample['sample_through'])

        x_neg = torch.cat(l_x_neg, dim=0)
        x_init = torch.cat(l_x_init, dim=0)
        x_init_pre = torch.cat(l_x_init_pre, dim=0)
        if self.improved_cd:
            x_through = torch.cat(l_x_through, dim=0)
        else:
            x_through = None

        if self.n_replay_x is not None:
            self.buffer.push(x_neg)

        d_off = {"x_neg": x_neg, "x_init": x_init}

        d_train = {**d_off, **d_omi}

        """ for visualization"""
        if len(x.shape) == 4 and x_neg is not None and x.shape[1] in {1,3}:
            x_neg_off_img = make_grid(x_neg.detach().cpu(), nrow=10, value_range=(0, 1))
            x_img = make_grid(x.detach().cpu(), nrow=10, value_range=(0, 1))
            x_init_img = make_grid(x_init.detach().cpu(), nrow=10, value_range=(0, 1))
            x_init_pre_img = make_grid(
                x_init_pre.detach().cpu(), nrow=10, value_range=(0, 1)
            )
            d_train["input@"] = x_img
            d_train["x_init_before@"] = x_init_pre_img
            d_train["x_init_after@"] = x_init_img
            d_train["x_neg_off@"] = x_neg_off_img


        d_loss = self.update_model_off(x.detach(), x_neg.detach(), optimizer, y=y, x_through=x_through)

        return {**d_train, **d_loss}

    def update_model_off(self, x, x_neg, optimizer, y=None, x_through=None):
        if self.net_x is not None:
            self.net_x.train()
        optimizer.zero_grad()
        # projection recovery
        if x_neg is not None:
            d_off_neg = self.energy_x(x_neg, train=True, y=y)
            d_off_pos = self.energy_x(x, train=True, y=y)
            off_neg_e, vx_neg = (
                d_off_neg["energy"],
                d_off_neg["vx"],
            )
            off_pos_e, vx_pos = (
                d_off_pos["energy"],
                d_off_pos["vx"],
            )
            loss_off = off_pos_e.mean() - off_neg_e.mean()
        else:
            loss_off = torch.tensor(0.0, device=x.device)
            off_pos_e = torch.tensor(0.0, device=x.device)
            off_neg_e = torch.tensor(0.0, device=x.device)
            vx_pos = torch.tensor(0.0, device=x.device)
            vx_neg = torch.tensor(0.0, device=x.device)

        loss = loss_off

        # improved CD
        if self.improved_cd and x_through is not None:
            self.requires_grad_(False)
            d_off_through = self.energy_x(x_through, train=True, y=y)
            off_through_e = d_off_through["energy"]
            self.requires_grad_(True)
            loss_cd = off_through_e.mean()
            loss += loss_cd

        # reguarlization
        d_reg = {}
        netx_norm = self.weight_norm()
        if self.l2_norm_reg_netx is not None:
            loss = loss + self.l2_norm_reg_netx * netx_norm

        reg_gamma_vx = (vx_pos**2).mean() + (vx_neg**2).mean()
        if self.gamma_vx is not None:
            loss += reg_gamma_vx * self.gamma_vx

        if self.gamma_neg_recon is not None:
            loss += self.gamma_neg_recon * (d_off_neg["recon"] ** 2).mean()
                
        if self.gamma_neg_b is not None:
            loss += self.gamma_neg_b * (d_off_neg["b"] ** 2).mean()

        if self.custom_netx_reg:
            loss += self.net_x.reg()

        loss.backward()
        optimizer.step()
        d_train = {
            "loss": loss.item(),
            "mpdr/loss_off_": loss_off.item(),
            "mpdr/off_pos_e_": off_pos_e.mean().item(),
            "mpdr/off_neg_e_": off_neg_e.mean().item(),
            "reg/reg_gamma_vx_": reg_gamma_vx.item(),
            "reg/netx_norm_": netx_norm.item(),
        }
        if self.use_recon_error:
            d_train["recon_error/recon_error_pos_"] = (
                d_off_pos["recon_error"].mean().item()
            )
            d_train["recon_error/recon_error_neg_"] = (
                d_off_neg["recon_error"].mean().item()
            )
            d_train["recon_error/a_"] = self.a.item()
            d_train["recon_error/b_"] = self.b.item()
        d_train = {**d_train, **d_reg}

        if hasattr(self.net_x, 'out_scale'):
            if not isinstance(self.net_x.out_scale, nn.Module):
                d_train['mpdr/out_scale_'] = self.net_x.out_scale.item()
        return d_train

    def log_prob(self, x, ae, z0, proj_noise, y=None, proj_const=None):
        if proj_const is None:
            proj_const = self.proj_const

        e = self.energy_x(x, y=y)
        if z0 is None:  # test-time sampling mode
            recov = 0.0
        elif self.proj_dist == "geodesic":
            dot = (ae.encode(x) * z0).sum(dim=1, keepdim=True)
            eps = 1e-6
            recov = torch.acos(dot - eps) ** 2 / 2 / proj_noise**2
        elif self.proj_dist == "sum":
            recov = (
                ((ae.encode(x) - z0) ** 2).view(len(x), -1).sum(dim=1)
                / 2
                / proj_noise**2
            )
        return (
            -e / self.temperature - recov.view(len(x), -1).sum(dim=1) * self.proj_const
        )

    def init_sample_z(self, z_obs):
        """initialize initial z samples from buffer or from uniform"""
        if self.replay_ratio is None:
            return z_obs.clone().detach()

        n_sample = len(z_obs)
        if len(self.buffer) == 0:
            n_replay = 0
            sample = torch.tensor([])
        else:
            n_replay = (np.random.rand(n_sample) < self.replay_ratio).sum()
            sample = self.buffer.get(n_replay)
        new_sample = self._project(torch.randn(n_sample - n_replay, *z_obs.shape[1:]))
        sample = torch.cat([sample, new_sample]).to(z_obs.device)
        if self.buffer_match_nearest:
            # reorder sample to match their nearest neighbor (dot product) in z_obs
            dot = torch.einsum("iklm,jklm->ij", sample, z_obs)
            _, idx = torch.sort(dot, dim=1, descending=True)
            max_dot = dot[torch.arange(len(dot)), idx[:, 0]]
            _, idx = torch.sort(max_dot, dim=0, descending=True)
            sample = sample[idx]
            assert len(sample) == n_sample

        return sample.to(z_obs.device).detach()

    def on_manifold_init_v2(self, x_init, z_obs, proj_noise, y=None, ae=None):
        # z = self.init_sample_z(z_obs)
        z = z_obs.detach()
        if self.mcmc_n_step_omi is None:
            return {"x_init": x_init}

        proj_noise = proj_noise.reshape(len(proj_noise), *[1] * len(z.shape[1:]))
        n_step = self.mcmc_n_step_omi
        step_size = self.mcmc_stepsize_omi
        noise_scale = self.mcmc_noise_omi

        f = partial(self.log_prob, ae=ae, z0=z_obs, proj_noise=proj_noise, y=y, proj_const=self.proj_const_omi)
        z0 = z.clone().detach()
        d_mcmc = sample_langevin_v2(
            z0,
            lambda zz: -f(ae.decode(zz)),
            stepsize=step_size,
            n_step=n_step,
            noise_scale=noise_scale,
            mh=self.mh_omi,
            temperature=self.temperature_omi,
            normalize_grad=self.mcmc_normalize_omi,
            bound="spherical" if ae.spherical else None,
        )

        d_sample = {"x_init": ae.decode(d_mcmc["sample"]), "d_mcmc": d_mcmc}
        # monitoring
        if n_step > 0:
            omi_energy_grad = (
                torch.stack(d_mcmc["l_drift"]).pow(2).sum(dim=1).sqrt().mean()
            )
            d_sample["mpdr/omi_energy_grad_norm_"] = omi_energy_grad.item()
            d_sample["mcmc/omi_energy_"] = d_mcmc["l_E"][-1].mean().item()
            if self.mh_omi:
                d_sample["mcmc/omi_accept_"] = (
                    torch.cat(d_mcmc["l_accept"]).float().mean().item()
                )
        return d_sample
    
    def clip_vector_norm(self, x, max_norm):
        norm = x.norm(dim=1, keepdim=True)
        x = x * (
            (norm < max_norm).to(torch.float)
            + (norm > max_norm).to(torch.float) * max_norm / norm
            + 1e-6
        )
        return x

    def p_sample_langevin_off_v2(self, *, ae, x_init, z_obs, proj_noise, y=None):
        if z_obs is not None:
            proj_noise = proj_noise.reshape(
                len(proj_noise), *[1] * len(z_obs.shape[1:])
            )
        noise_scale = self.mcmc_noise_x
        if self.mcmc_custom_stepsize:
            # ignore following the original langevin dynamics parameter (stepsize = sigma^2/2)
            step_size = self.mcmc_stepsize_x
        else:
            step_size = self.mcmc_stepsize_x * (noise_scale**2) / 2

        f = partial(self.log_prob, ae=ae, z0=z_obs, proj_noise=proj_noise, y=y)
        x0 = x_init.clone().detach()
        # d_sample = sample_langevin_v2(x0, self.energy_x,
        d_sample = sample_langevin_v3(
            x0,
            lambda xx: -f(xx),
            stepsize=step_size,
            n_step=self.mcmc_n_step_x,
            noise_scale=noise_scale,
            bound=self.mcmc_bound_x,
            # mh=self.mh_x,
            temperature=None,
            through_sampler=self.improved_cd
        )  # temperature is already reflected
        d_result = {
            "sample": d_sample["sample"],
            "l_grad_y": d_sample["l_drift"],
            "l_sample": d_sample["l_sample"],
            "mcmc_n_step_x": self.mcmc_n_step_x,
            "d_sample": d_sample,
        }
        l_grad_y = d_sample["l_drift"]
        if len(l_grad_y) > 0:
            d_result["avg_grad_norm"] = (
                torch.stack(l_grad_y).reshape(len(l_grad_y), -1).norm(dim=1).mean()
            )
        else:
            d_result["avg_grad_norm"] = torch.tensor(0)

        if self.improved_cd:
            d_result['sample_through'] = d_sample['sample_through']

        return d_result

    def weight_norm(self):
        netx_norm = weight_norm(self.net_x)
        return netx_norm

    def sample_step(self, x, **kwargs):
        # generate sample from the current model
        d_result = {"loss": 0.0}

        d_sample = self.sample(len(x), x.device, image_data=len(x.shape) == 4)

        if len(x.shape) == 4:
            x_sample = d_sample["sample"]
            x_zstart = self.ae.decode(d_sample["l_z_sample"][-1])
            x_zend = self.ae.decode(d_sample["z_sample"])
            x_zstart = make_grid(x_zstart, nrow=8, value_range=(0, 1))
            x_zend = make_grid(x_zend, nrow=8, value_range=(0, 1))
            x_xend = make_grid(x_sample, nrow=8, value_range=(0, 1))
            d_result["sample_z_start@"] = x_zstart
            d_result["sample_z_end@"] = x_zend
            d_result["sample_x_end@"] = x_xend
        d_result = {**d_result, **d_sample}
        return d_result

    # def ae_parameters(self):
    #     return chain(*[ae.parameters() for ae in self.l_ae])
        
