from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch, math

class NoiseSchedule(nn.Module):
    def __init__(self, t=1000, beta_min=1e-4, beta_max=2e-2):
        super().__init__()
        self.t = t
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.register_buffer('betas', torch.linspace(beta_min**0.5, beta_max**0.5, t) ** 2)
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_prods', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('signal_stds', self.alpha_prods.sqrt())
        self.register_buffer('noise_stds', (1 - self.alpha_prods).sqrt())

    def forward(self, x):
        # noises each sample in x to a random timestep 
        B = x.shape[0]
        noise = torch.randn_like(x)
        timesteps = torch.randint(self.t, (B,))
        noised = self.signal_stds[timesteps] * x + self.noise_stds[timesteps] * noise
        return noised.detach(), timesteps, noise # not entirely sure why detach is needed

class DDIMSampler():
    def __init__(self, denoiser,
            noise_schedule=NoiseSchedule(),
            tau_dim=None,
            eta=0.0,
        ):
        self.denoiser = denoiser
        self.noise_schedule = noise_schedule
        self.tau_dim = noise_schedule.t if tau_dim is None else tau_dim
        self.tau = np.linspace(0, self.noise_schedule.t-1, self.tau_dim).astype(np.int32)
        self.eta = eta

        self.alphas = torch.cat([torch.ones((1,)), self.noise_schedule.alpha_prods[self.tau]])
        self.betas = 1 - self.alphas

    def denoise_step(self, x, t, noise_pred):
        beta_ratio = self.betas[t-1] / self.betas[t]
        alpha_ratio = self.alphas[t] / self.alphas[t-1]
        sigma = self.eta * math.sqrt(beta_ratio * (1 - alpha_ratio))
        x0_step = self.alphas[t]**-0.5 * (x - self.betas[t]**0.5 * noise_pred)
        xt_step = math.sqrt(1 - self.alphas[t-1] - sigma**2) * noise_pred
        added_noise = sigma * torch.randn_like(x)
        return self.alphas[t-1]**0.5 * x0_step + xt_step + added_noise

    def get_samples(self, initial_x=None, x_shape=None, context=None, verbose=True):
        if initial_x is None and x_shape is None:
            raise Exception('Either initial_x or x_shape must be defined.')

        x = torch.randn(x_shape) if initial_x is None else initial_x
        for t in tqdm(range(self.tau_dim, 0, -1)):
            t_repeated = t * torch.ones(x.shape[:1], dtype=int).to(x.device)
            x = self.denoise_step(x, t, self.denoiser(x, t_repeated, context))
        return x
