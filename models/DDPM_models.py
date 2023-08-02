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
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from tqdm import tqdm
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class LossType(object):
    def __init__(self):
        self.L1 = False
        self.L2 = True
        self.SSIM = False
        self.Perceptual =False
        self.KL =False



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
        loss_type (string): loss type, "l1" or "l2" ,"SSIM"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """

    def __init__(self, model, img_channels=3, img_size=(32, 24), timesteps=1000, loss_type=LossType()):
        super().__init__()

        self.model = model
        self.img_size = img_size
        self.img_channels = img_channels

        self.loss_type = loss_type
        self.timesteps = timesteps

        betas = np.linspace(0.0001, 0.02, self.timesteps)

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
    def remove_noise(self, x, t, class_index, cond_image):

        return ((x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, class_index=class_index,
                                                                               cond_image=cond_image)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
                )

    @torch.no_grad()
    def sample(self, batch_size, device, class_index=None, cond_image=None):
        if class_index is not None and batch_size != len(class_index):
            raise ValueError("sample batch size different from length of given class_index")
        if cond_image is not None and batch_size != len(cond_image):
            raise ValueError("sample batch size different from length of given cond_image")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        for t in tqdm(range(self.timesteps - 1, -1, -1), desc=f'sample times', total=self.timesteps):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, class_index, cond_image)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, class_index=None, cond_image=None):
        if class_index is not None and batch_size != len(class_index):
            raise ValueError("sample batch size different from length of given class_index")
        if cond_image is not None and batch_size != len(cond_image):
            raise ValueError("sample batch size different from length of given cond_image")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, class_index, cond_image)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, class_index=None, cond_image=None):

        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, class_index, cond_image)

        loss1 = None
        loss2 = None
        lossSSIM = None

        if self.loss_type.L1:
            loss1 = F.l1_loss(estimated_noise, noise)
        if self.loss_type.L2:
            loss2 = F.mse_loss(estimated_noise, noise)
        if self.loss_type.SSIM:
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(noise.device)
            lossSSIM = ssim(estimated_noise, noise)

        return loss1, loss2, lossSSIM

    def forward(self, x, class_index=None, cond_image=None):
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[1]:
            raise ValueError("image width does not match diffusion parameters")

        t = torch.randint(0, self.timesteps, (b,), device=device)
        return self.get_losses(x, t, class_index, cond_image)
