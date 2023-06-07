"""
idnn.py
=======
interpolation deep neural network for dcase anomaly detection
"""
import torch
import torch.nn as nn
from models.modules import FCNet
from models.utils import weight_norm
from torch.distributions import Normal
# from .v5_single import MPDR_Single


class IDNN(nn.Module):
    """Interpolation DNN
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054344
    """

    def __init__(
        self,
        contextnet,
        decoder,
        interp_dim_start,
        interp_dim_end,
        *,
        spherical=True,
        l2_norm_reg_context=None,
        encoding_noise=None,
        eps=1e-6,
        reg_z_norm=None,
        input_noise=None,
        loss="l2",
    ):
        """
        contextnet: context network. encoder name is reserved for future use
        decoder: decoder network
        """
        super().__init__()
        self.contextnet = contextnet 
        self.decoder = decoder
        self.spherical = spherical
        self.l2_norm_reg_context = l2_norm_reg_context
        self.reg_z_norm = reg_z_norm  # for euclidean latent space
        self.encoding_noise = encoding_noise
        self.eps = eps
        self.input_noise = input_noise
        self.loss = loss
        assert loss in {"l2", "l1"}
        self.interp_dim_start = interp_dim_start
        self.interp_dim_end = interp_dim_end
        self.target_dim = interp_dim_end - interp_dim_start


    def clip(self, full_x):
        """divide input into two parts"""
        assert full_x.dim() == 2, (
            "Only 2D data is supported!"
        )
        input_x = torch.cat([full_x[:, :self.interp_dim_start], full_x[:, self.interp_dim_end:]], dim=1)
        target_x = full_x[:, self.interp_dim_start:self.interp_dim_end]
        return input_x, target_x

    def context(self, x, noise=False, train=False):
        """wrapper for contextnet"""
        z = self.contextnet(x)
        if noise and self.encoding_noise is not None:
            z = self._project(z)
            z = z + torch.randn_like(z) * self.encoding_noise

        z_pre = z
        if self.spherical:
            z = self._project(z_pre)

        if train:
            return {"z": z, "z_pre": z_pre}
        else:
            return z

    def decode(self, x):
        """wrapper for decoder"""
        return self.decoder(x)

    def _project(self, z):
        """project to unit sphere"""
        return z / (torch.norm(z, dim=1, keepdim=True) + self.eps)

    def recon_error(self, x, noise=False):
        input_x, target_x = self.clip(x)
        """reconstruction error"""
        z = self.context(input_x, noise=noise)
        out_x = self.decode(z)
        if self.loss == "l2":
            return ((target_x - out_x) ** 2).view(len(x), -1).mean(dim=1)
        elif self.loss == "l1":
            return (torch.abs(target_x - out_x)).view(len(x), -1).mean(dim=1)

    def predict(self, x):
        return self.recon_error(x, noise=False)

    def forward(self, x):
        recon = self.recon_error(x, noise=False)
        self.state = {'recon': recon}
        return recon

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        """train autoencoder by minimizing the reconstruction loss"""
        input_x, target_x = self.clip(x)
        self.train()
        optimizer.zero_grad()
        recon_error = self.recon_error(x, noise=False).mean()
        loss = recon_error
        d_train = {"loss/recon_error_": recon_error.item()}

        contextnet_norm, decoder_norm = self.weight_norm()
        d_train["loss/contextnet_norm_"] = contextnet_norm.item()
        d_train["loss/decoder_norm_"] = decoder_norm.item()
        loss = loss + self.reg()

        # z_norm = (z**2).mean()
        # if self.reg_z_norm is not None:
        #     loss = loss + self.reg_z_norm * z_norm
        # z_pre_norm = z_pre.view(len(x), -1).norm(dim=1).mean()
        # d_train["loss/ae_z_norm_"] = z_norm.item()
        # d_train["loss/ae_z_pre_norm_"] = z_pre_norm.item()

        loss.backward()
        optimizer.step()
        d_train["loss/train_loss_ae_"] = loss.item()

        # required for ood trainer
        d_train["loss"] = loss.item()
        d_train = {**d_train}
        return d_train

    def validation_step(self, x, y=None):
        """validation step"""
        self.eval()
        """clip data"""
        recon_error = self.recon_error(x, noise=False).mean()
        loss = recon_error
        d_result = {"loss/val_recon_error_": recon_error.item()}
        d_result["loss"] = loss.item()
        return d_result

    def weight_norm(self):
        decoder_norm = weight_norm(self.decoder)
        contextnet_norm = weight_norm(self.contextnet)
        return contextnet_norm, decoder_norm

    def reg(self):
        if self.l2_norm_reg_context is not None:
            contextnet_norm, decoder_norm = self.weight_norm()
            return self.l2_norm_reg_context * contextnet_norm
        else:
            return 0


class IDNN_Plus_AE(nn.Module):
    """an energy function that returns the sum of IDNN and AE recon error"""
    def __init__(self, idnn, ae):
        """
        idnn: IDNN class
        ae: AE class
        """
        super().__init__()
        self.idnn = idnn
        self.ae = ae

    def forward(self, x):
        """forward pass"""
        idnn_recon = self.idnn(x)
        ae_recon = self.ae(x)
        recon = idnn_recon + ae_recon
        self.state = {'recon': recon, 'idnn_recon': (idnn_recon**2).mean(), 'ae_recon': (ae_recon**2).mean()}
        return recon

    def predict(self, x):
        return self.forward(x)

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        raise NotImplementedError

    def validation_step(self, x, y=None):
        raise NotImplementedError

    def weight_norm(self):
        idnn_norm = self.idnn.weight_norm()
        ae_norm = self.ae.weight_norm()
        return idnn_norm, ae_norm

# class MPDR_Conditional_IDNN(MPDR_Single):
#     """
#     Same to conditional MPDR, but takes the whole vector as input
#     An input vector is splitted in a similar way to IDNN.
#     """
#     def __init__(self, *, interp_dim_start, interp_dim_end, **kwargs):
#         assert kwargs['conditional'], 'should be conditional'
#         assert not kwargs['use_recon_error'], 'should not use recon_error'
#         self.interp_dim_start = interp_dim_start
#         self.interp_dim_end = interp_dim_end
#         super().__init__(**kwargs)
# 
#     def clip(self, full_x):
#         """divide input into two parts"""
#         assert full_x.dim() == 2, (
#             "Only 2D data is supported!"
#         )
#         input_x = torch.cat([full_x[:, :self.interp_dim_start], full_x[:, self.interp_dim_end:]], dim=1)
#         target_x = full_x[:, self.interp_dim_start:self.interp_dim_end]
#         return input_x.detach(), target_x.detach()
# 
#     def forward(self, x, y=None):
#         condition_x, target_x = self.clip(x)
#         return super().forward(target_x, y=condition_x)
# 
#     def train_step_ae(self, x, optimizer, **kwargs):
#         condition_x, target_x = self.clip(x)
#         return super().train_step_ae(target_x, optimizer)
# 
#     def validation_step_ae(self, x, y=None):
#         condition_x, target_x = self.clip(x)
#         return super().validation_step_ae(target_x, y=target_x)
# 
#     def train_step(self, x, optimizer, clip_grad=None, y=None, mode='off'):
#         condition_x, target_x = self.clip(x)
#         return super().train_step(target_x, optimizer, clip_grad=None, y=condition_x, mode=mode)




    


class FCNet_Conditional_Concat(FCNet):
    """For modeling conditional energy function. Simply concatenate the input to model the conditional energy"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, y):
        xx = torch.cat([x, y], dim=1)
        return super().forward(xx)


class GroupMADE(IDNN):
    """Group Masked Autoencoder for Distribution Estimation"""
    def __init__(self, contextnet, decoder, interp_dim_start, interp_dim_end, n_component, **kwargs):
        super().__init__(contextnet, decoder, interp_dim_start, interp_dim_end, **kwargs)
        self.n_component = n_component

    def recon_error(self, x, noise=False):
        """actually computes negative log likelihood"""
        input_x, target_x = self.clip(x)
        z = self.context(input_x, noise=noise)
        out = self.decode(z)  # (batch, 3 * target_dim * n_component) mean, var, weight
        # reshape
        out = out.view(-1, self.target_dim, self.n_component, 3)
        mean, var, weight = out[:, :, :, 0], out[:, :, :, 1], out[:, :, :, 2]
        var = torch.exp(var)
        # from pudb import set_trace; set_trace()
        # print(var.min())
        weight = torch.softmax(weight, dim=2)
        normal_distributions = Normal(mean, var)
        # repeat target_x to match the shape of normal_distributions
        target_x = target_x.unsqueeze(2).repeat(1, 1, self.n_component)
        log_prob_comp = normal_distributions.log_prob(target_x) + torch.log(weight)
        log_prob = torch.logsumexp(log_prob_comp, dim=2)
        nll = - log_prob.mean(dim=1)
        return nll

    def reg(self):
        reg = 0
        if self.l2_norm_reg_context is not None:
            contextnet_norm, decoder_norm = self.weight_norm()
            reg += self.l2_norm_reg_context * contextnet_norm

        if self.l2_norm_reg_var is not None:
            # (self.decoder.net[-1].weight ** 2).sum() + (self.decoder.net[-1].bias ** 2).sum()
            pass

        if self.l2_norm_reg_pi is not None:
            pass
            
        return reg

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        """train autoencoder by minimizing the reconstruction loss"""
        input_x, target_x = self.clip(x)
        self.train()
        optimizer.zero_grad()
        recon_error = self.recon_error(x, noise=False).mean()
        loss = recon_error
        d_train = {"loss/recon_error_": recon_error.item()}

        contextnet_norm, decoder_norm = self.weight_norm()
        d_train["loss/contextnet_norm_"] = contextnet_norm.item()
        d_train["loss/decoder_norm_"] = decoder_norm.item()
        loss = loss + self.reg()

        loss.backward()
        optimizer.step()
        d_train["loss/train_loss_ae_"] = loss.item()

        # required for ood trainer
        d_train["loss"] = loss.item()
        d_train = {**d_train}
        return d_train

