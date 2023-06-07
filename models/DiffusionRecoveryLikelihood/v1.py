"""
PyTorch re-implementation of the original repo
https://github.com/ruiqigao/recovery_likelihood
"""
import os
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from torchvision.utils import make_grid 


def get_sigma_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    https://github.com/ruiqigao/recovery_likelihood/blob/c77cc0511dedcb8d9ab928438d80acb62aeca96f/model.py#L20
    Get the noise level schedule
    :param beta_start: begin noise level
    :param beta_end: end noise level
    :param num_diffusion_timesteps: number of timesteps
    :return:
    -- sigmas: sigma_{t+1}, scaling parameter of epsilon_{t+1}
    -- a_s: sqrt(1 - sigma_{t+1}^2), scaling parameter of x_t
    
    t=0: data. smaller noise when near to the data.
    a = sqrt(1-sigma^2)
    """
    if beta_schedule == 'drl': # original in diffusion recovery likelihood T6 setting. Not working for num_diffusion_timestemps=1000
        betas = np.linspace(beta_start, beta_end, 1000, dtype=np.float64)
        betas = np.append(betas, 1.)  # to pad 1 at the end. however unnecessary because numpy does not raise index out of range error
        assert isinstance(betas, np.ndarray)
        betas = betas.astype(np.float64)
        assert (betas > 0).all() and (betas <= 1).all()
        sqrt_alphas = np.sqrt(1. - betas)

        idx = np.concatenate([np.arange(num_diffusion_timesteps) * (1000 // ((num_diffusion_timesteps - 1) * 2)), [999]])  # why?
        a_s = np.concatenate(
            [[np.prod(sqrt_alphas[: idx[0] + 1])],
             np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])    
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        sqrt_alphas = np.sqrt(1. - betas)
        a_s = sqrt_alphas
    else:
        raise NotImplementedError
    
    sigmas = np.sqrt(1 - a_s ** 2)

    return torch.tensor(sigmas, dtype=torch.float), torch.tensor(a_s, dtype=torch.float)


class DiffusionRecoveryLikelihood(nn.Module):
    def __init__(self, net, num_timesteps, beta_schedule='drl', sampling='gaussian',
            mcmc_step_size_b_square=2e-4, noise_scale=1.0, paper_langevin_scaling=False, mcmc_num_steps=30, img_sz=32, channel=3, beta_start=0.0001, beta_end=0.02):
        """
        net: a function that takes in y and t and returns the energy
        num_timesteps: number of timesteps
        beta_schedule: 'drl' or 'linear'
        sampling: 'gaussian' or 'langevin'
        paper_langevin_scaling: whether to use the scaling in the paper
        """
        super().__init__()
        self.net = net
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.sampling = sampling
        self.mcmc_step_size_b_square = 2e-4
        self.mcmc_num_steps = mcmc_num_steps
        self.noise_scale = 1.0  # 'MCMC sampling noise scale, 1.0 in training / 0.99 in testing'
        self._init_sigma(beta_start=beta_start, beta_end=beta_end)
        self.paper_langevin_scaling = paper_langevin_scaling
        self.img_sz = img_sz
        self.channel = channel
        
    def _init_sigma(self, beta_start=0.0001, beta_end=0.02):
        """initialize """
        sigmas, a_s = get_sigma_schedule(self.beta_schedule, beta_start=beta_start,  beta_end=beta_end, num_diffusion_timesteps=self.num_timesteps)
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('a_s', a_s)
        self.register_buffer('a_s_cum', torch.cumprod(self.a_s, dim=0))
        self.register_buffer('sigmas_cum', torch.sqrt(1 - self.a_s_cum ** 2))
        a_s_prev = torch.clone(self.a_s)
        a_s_prev[-1] = 1
        self.register_buffer('a_s_prev', a_s_prev)
        
        is_recovery = torch.ones(self.num_timesteps + 1, dtype=torch.float)
        is_recovery[-1] = 0
        self.register_buffer('is_recovery', is_recovery)
        
    def log_prob(self, y, t, tilde_x, b0, sigma, is_recovery, *, dropout):
        """
        b0: step_size_square
        """
        # return self.net(y, t) / b0.flatten() - torch.sum((y - tilde_x) ** 2 / 2 / sigma ** 2 * is_recovery, dim=[1, 2, 3])
        # return - self.net(y, t) / b0.flatten() - ((y - tilde_x) ** 2 / 2 / sigma ** 2 * is_recovery).view(len(y), -1).sum(dim=1)
        if self.paper_langevin_scaling:
            return - self.net(y, t) / b0.flatten() - ((y - tilde_x) ** 2 / 2 / sigma ** 2 * is_recovery).view(len(y), -1).sum(dim=1)
        else:
            return - self.net(y, t) - ((y - tilde_x) ** 2 / 2 / sigma ** 2 * is_recovery).view(len(y), -1).sum(dim=1)
        
    def train_step(self, x, optimizer, **kwargs):
        t = torch.randint(low=0, high=self.num_timesteps, size=(x.shape[0],), device=x.device)
        # sample pairs x_pos, x_perturb
        x_pos, x_perturb = self.q_sample_pairs(x, t)  # these are actually y's
        if self.sampling == 'gaussian':
            x_neg = self.p_sample_gaussian(x_perturb, t, dropout=None)
        elif self.sampling == 'langevin':
            x_neg = self.p_sample_langevin(x_perturb, t, dropout=None)
        else:
            raise NotImplementedError
        d_loss = self.update_model(x_pos, x_neg, t, optimizer)
        d_train = {**d_loss, 't': t.detach(), 'x_pos': x_pos, 'x_perturb': x_perturb, 'x_neg': x_neg}
        return d_train

    def validation_step(self, x, **kwargs):
        # generate sample from the current model
        d_val = {'loss': 0.}

        self.eval()
        noise = torch.randn_like(x)
        x_sample = self.p_sample_progressive(noise).detach().cpu()
        if len(x.shape) == 4:
            x_start = make_grid(x_sample[0], nrow=8, value_range=(-1, 1))
            x_half = make_grid(x_sample[len(x_sample) // 2], nrow=8, value_range=(0, 1))
            x_end = make_grid(x_sample[-1], nrow=8, value_range=(-1, 1))
            d_val['sample_0@'] = x_start 
            d_val['sample_half@'] = x_half 
            d_val['sample_T@'] = x_end 
            d_val['diffused_samples#'] = x_sample
        return d_val
    
    def update_model(self, x_pos, x_neg, t, opt):
        self.train()
        opt.zero_grad()
        a_s = self._extract(self.a_s_prev, t + 1, x_pos.shape)
        y_pos = a_s * x_pos
        y_neg = a_s * x_neg
        pos_e = self.net(y_pos, t)
        neg_e = self.net(y_neg, t)
        loss = pos_e - neg_e
        loss_scale = 1.0 / self.sigmas[t+1] / self.sigmas[1]
        loss = loss_scale * loss
        loss = loss.mean()
        loss.backward()
        opt.step()
        return {'loss': loss.item(), 'pos_e_': pos_e.mean().item(), 'neg_e_': neg_e.mean().item()}
    
    def p_sample_gaussian(self, tilde_x, t, *, dropout):
        sigma = self._extract(self.sigmas, t+1, tilde_x.shape)
        a_s = self._extract(self.a_s_prev, t + 1, tilde_x.shape)
        
        y = tilde_x
        y.requires_grad = True
        E = self.net(y, t)
        grad_y = autograd.grad(E.sum(), y, only_inputs=True)[0]
        noise = torch.randn(*y.shape, device=tilde_x.device)
        y_new = y - sigma * grad_y + torch.sqrt(sigma) * noise
        x = y_new  / a_s.expand_as(y_new)
        return x 
    
    def p_sample_langevin(self, tilde_x, t, *, dropout):
        sigma = self._extract(self.sigmas, t+1, tilde_x.shape)
        sigma_cum = self._extract(self.sigmas_cum, t, tilde_x.shape)
        is_recovery = self._extract(self.is_recovery, t + 1, tilde_x.shape)
        a_s = self._extract(self.a_s_prev, t + 1, tilde_x.shape)
        
        if self.paper_langevin_scaling:
            c_t_square = sigma_cum / self.sigmas_cum[0]
        else:
            c_t_square = 1.
        step_size_square = c_t_square * self.mcmc_step_size_b_square * sigma ** 2
        
        y = tilde_x
        y.requires_grad = True
        for _ in range(self.mcmc_num_steps):
            y.requires_grad = True
            log_prob = self.log_prob(y, t, tilde_x, step_size_square, sigma, is_recovery, dropout=None)
            grad_y = autograd.grad(log_prob.sum(), y, only_inputs=True)[0]
            noise = torch.randn(*y.shape, device=tilde_x.device)
            y_new = y + 0.5 * step_size_square * grad_y + torch.sqrt(step_size_square) * noise * self.noise_scale
            y = y_new.detach()
        x = y  / a_s.expand_as(y_new)
        return x    
        
    def q_sample(self, x_start, t, *, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(*x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        x_t = self._extract(self.a_s_cum, t, x_start.shape) * x_start + self._extract(self.sigmas_cum, t, x_start.shape) * noise
        return x_t
    
    def q_sample_pairs(self, x_start, t, *, noise=None):
        """
        Generate a pair of disturbed images for training
        :param x_start: x_0
        :param t: time step t
        :return: x_t, x_{t+1}
        """
        noise = torch.randn(x_start.shape, device=x_start.device)
        x_t = self.q_sample(x_start, t)
        x_t_plus_one = self._extract(self.a_s, t+1, x_start.shape) * x_t + \
                       self._extract(self.sigmas, t+1, x_start.shape) * noise

        return x_t, x_t_plus_one

    def q_sample_progressive(self, x_0):
        """Generate a full sequence of perturbed samples"""
        x_preds = []
        for t in range(self.num_timesteps + 1):
            t_now = torch.ones(x_0.shape[0], device=x_0.device, dtype=torch.long) * t
            x_t = self.q_sample(x_0, t_now)
            x_preds.append(x_t)
        x_preds = torch.stack(x_preds, dim=0)
        return x_preds

    def p_sample_progressive(self, noise):
        """
        Sample a sequence of images with the sequence of noise levels
        """
        num = noise.shape[0]
        x_neg_t = noise
        if self.img_sz == 1:   # for vector data
          x_neg = torch.zeros([self.num_timesteps, num, self.channel], dtype=torch.float, device=noise.device)
        else:
          x_neg = torch.zeros([self.num_timesteps, num, self.channel, self.img_sz, self.img_sz], dtype=torch.float, device=noise.device)
        x_neg = torch.cat([x_neg, noise.unsqueeze(0)], axis=0)

        for t in range(self.num_timesteps - 1, -1, -1):
          x_neg_t = self.p_sample_langevin(x_neg_t, t, dropout=0.)
          if self.img_sz == 1:
            x_neg_t = torch.reshape(x_neg_t, [num, self.channel])
          else:
            x_neg_t = torch.reshape(x_neg_t, [num, self.channel, self.img_sz, self.img_sz])
          insert_mask = t == torch.arange(self.num_timesteps + 1, dtype=torch.long, device=noise.device)
          insert_mask = torch.reshape(insert_mask.float(), [-1, *([1] * len(noise.shape))])
          x_neg = insert_mask * x_neg_t.unsqueeze(0) + (1. - insert_mask) * x_neg
        return x_neg 
        
    def _extract(self, a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones(x_shape[0], dtype=torch.long) * t
        bs, = t.shape
        assert x_shape[0] == bs
        out = a[t]
        assert out.shape == (bs,)
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))        

    def predict(self, x):
        return self.net(x, 0)
