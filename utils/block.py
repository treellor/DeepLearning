import numpy as np

import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.
    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class Downsample(nn.Module):
    __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb, y):
        return self.upsample(x)


