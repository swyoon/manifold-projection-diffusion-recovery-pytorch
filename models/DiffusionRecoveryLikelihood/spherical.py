"""
Spherical Diffusion Recovery Likelihood
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

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
        betas = np.concatenate([[0], betas])
        assert isinstance(betas, np.ndarray)
        betas = betas.astype(np.float64)
        sqrt_alphas = np.sqrt(1. - betas)

        idx = np.concatenate([[0], np.arange(1, num_diffusion_timesteps) * (1000 // ((num_diffusion_timesteps - 1) * 2)), [999]])
        a_s = np.concatenate(
            [[np.prod(sqrt_alphas[: idx[0] + 1])],
             np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])    
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        betas = np.concatenate([[0], betas])
        sqrt_alphas = np.sqrt(1. - betas)
        a_s = sqrt_alphas
    elif beta_schedule == 'constant':
        betas = np.ones(num_diffusion_timesteps, dtype=np.float64) * beta_start
        betas = np.concatenate([[0], betas])
    else:
        raise NotImplementedError
    
    sigmas = np.sqrt(betas)

    return torch.tensor(sigmas, dtype=torch.float)


class SphericalDiffusionRecoveryLikelihood(nn.Module):
    def __init__(self, net, num_timesteps, beta_schedule='constant', sampling='gaussian',
           mcmc_step_size_b_square=2e-4, noise_scale=1.0, paper_langevin_scaling=False, mcmc_num_steps=30, beta_start=0.0001, beta_end=0.02):
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
        self.mcmc_step_size_b_square = mcmc_step_size_b_square
        self.mcmc_num_steps = mcmc_num_steps
        self.noise_scale = noise_scale  # 'MCMC sampling noise scale, 1.0 in training / 0.99 in testing'
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.paper_langevin_scaling = paper_langevin_scaling
        self._init_sigma()
        
        
    def _init_sigma(self):
        """initialize """
        sigmas = get_sigma_schedule(self.beta_schedule, beta_start=self.beta_start,  beta_end=self.beta_end, num_diffusion_timesteps=self.num_timesteps)
        self.register_buffer('sigmas', sigmas)
        
    def log_prob(self, y, t, tilde_x, b0, sigma, is_recovery, *, dropout):
        """
        b0: step_size_square
        """
        if self.paper_langevin_scaling:
            return - self.net(y, t) / b0.flatten() - ((y - tilde_x) ** 2 / 2 / sigma ** 2 * is_recovery).view(len(y), -1).sum(dim=1)
        else:
            return - self.net(y, t) - ((y - tilde_x) ** 2 / 2 / sigma ** 2 * is_recovery).view(len(y), -1).sum(dim=1)
        
    def train_step(self, x, opt):
        t = torch.randint(low=0, high=self.num_timesteps, size=(x.shape[0],), device=x.device)
        # sample pairs x_pos, x_perturb
        x_pos, x_perturb = self.q_sample_pairs(x, t) 
        if self.sampling == 'gaussian':
            x_neg = self.p_sample_gaussian(x_perturb, t, dropout=None)
        elif self.sampling == 'langevin':
            x_neg = self.p_sample_langevin(x_perturb, t, dropout=None)
        else:
            raise NotImplementedError
        loss = self.update_model(x_pos, x_neg, t, opt)
        d_train = {'loss': loss.item(), 't': t.detach(), 'x_pos': x_pos, 'x_perturb': x_perturb, 'x_neg': x_neg}
        return d_train
    
    def update_model(self, x_pos, x_neg, t, opt):
        opt.zero_grad()
        # a_s = self._extract(self.a_s, t + 1, x_pos.shape)
        # y_pos = a_s * x_pos
        # y_neg = a_s * x_neg
        pos_e = self.net(x_pos, t)
        neg_e = self.net(x_neg, t)
        loss = pos_e - neg_e
        loss_scale = 1.0 / self.sigmas[t+1] / self.sigmas[1]
        loss = loss_scale * loss
        loss = loss.mean()
        loss.backward()
        opt.step()
        return loss
    
    def p_sample_gaussian(self, tilde_x, t, *, dropout):
        sigma = self._extract(self.sigmas, t+1, tilde_x.shape)
        
        y = tilde_x
        y.requires_grad = True
        E = self.net(y, t)
        grad_y = autograd.grad(E.sum(), y, only_inputs=True)[0]
        noise = torch.randn(*y.shape, device=tilde_x.device)
        y_new = y - (sigma ** 2) * grad_y + sigma * noise
        y_new = self._project(y_new)
        return y_new
    
    def p_sample_langevin(self, tilde_x, t, *, dropout, is_recovery=1):
        sigma = self._extract(self.sigmas, t+1, tilde_x.shape)
        
        if self.paper_langevin_scaling:
            raise NotImplementedError
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
        return y    
        
    def q_sample(self, x_start, t, *, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 0 step)
        """
        if noise is None:
            noise = torch.randn(*x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        max_t = t.max()
        x = x_start
        for i in range(max_t.item()):  # simulate diffusion process
            noise = torch.randn(*x_start.shape, device=x_start.device)
            x = x + self._extract_until(self.sigmas, t, i, x_start.shape) * noise
            x = self._project(x)
        return x
    
    def _project(self, x):
        return x / x.view(len(x), -1).norm(dim=1, keepdim=True)
    
    def _extract_until(self, a, t, i, x_shape):
        """
        get i-th value only if i is smaller than t. otherwise return value filled with 1.
        t: target timestep
        i: current timestep
        """
        val = self._extract(a, i, x_shape)
        val[i > t] = 0.
        return val
    
    def q_sample_pairs(self, x_start, t, *, noise=None):
        """
        Generate a pair of disturbed images for training
        :param x_start: x_0
        :param t: time step t
        :return: x_t, x_{t+1}
        """
        noise = torch.randn(x_start.shape, device=x_start.device)
        x_t = self.q_sample(x_start, t)
        x_t_plus_one = self._project(x_t + self._extract(self.sigmas, t+1, x_start.shape) * noise)
        
        return x_t, x_t_plus_one

        
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

