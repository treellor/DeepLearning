"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.3.16
    Description	:
        DDPM 基本模块
    Reference	:
        Denoising Diffusion Probabilistic Models.    2020     Ian J. Goodfellow
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


from torch.nn.modules.normalization import GroupNorm
import copy
from utils.common import EMA, generate_cosine_schedule, generate_linear_schedule

# class EMA():
#     def __init__(self, decay):
#         self.decay = decay
#
#     def update_average(self, old, new):
#         if old is None:
#             return new
#         return old * self.decay + (1 - self.decay) * new
#
#     def update_model_average(self, ema_model, current_model):
#         for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
#             old, new = ema_params.data, current_params.data
#             ema_params.data = self.update_average(old, new)


class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.
    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """

    def __init__(self, model, img_channels=3, img_size=(32, 24), num_timesteps=1000, loss_type="l2"):
        super().__init__()

        self.model = model
        self.img_size = img_size
        self.img_channels = img_channels

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = num_timesteps

        betas = np.linspace(0.0001, 0.02, self.num_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    @torch.no_grad()
    def remove_noise(self, x, t, y):

        return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
        )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, y):
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, y)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        return loss

    def forward(self, x, y=None):
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[1]:
            raise ValueError("image width does not match diffusion parameters")

        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, y)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# def get_diffusion_from_args(config):
#     # activations = {
#     #     "relu": F.relu,
#     #     "mish": F.mish,
#     #     "silu": F.silu,
#     # }
#
#     model = UNet(
#         img_channels=3,
#
#         base_channels=config.base_channels,
#         channel_mults=config.channel_mults,
#         time_emb_dim=config.time_emb_dim,
#
#         dropout=config.dropout,
#         attention_resolutions=config.attention_resolutions,
#
#         num_classes=None,  # if not args.use_labels else 10,
#         initial_pad=0,
#     )
#
#     diffusion = GaussianDiffusion(
#         model, (32, 24), 3, 10,
#         #    ema_decay=config.ema_decay,
#         #    ema_update_rate=config.ema_update_rate,
#         num_timesteps=config.num_timesteps,
#         ema_start=2000,
#         loss_type=config.loss_type,
#     )
#
#     return diffusion
