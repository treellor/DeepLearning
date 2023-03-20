import torch.nn as nn
import torch.nn.functional as F


def get_norm(norm, num_channels):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_channels // 2, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


def get_activation(name):
    if name == "relu":
        return F.relu
    elif name == "mish":
        return F.mish
    elif name == "silu":
        return F.silu
    else:
        raise ValueError("unknown activation type")
