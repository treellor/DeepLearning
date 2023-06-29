import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# def get_norm(norm, num_channels, num_groups):
#     if norm == "in":
#         return nn.InstanceNorm2d(num_channels, affine=True)
#     elif norm == "bn":
#         return nn.BatchNorm2d(num_channels)
#     elif norm == "gn":
#         return nn.GroupNorm(num_groups, num_channels)
#     elif norm is None:
#         return nn.Identity()
#     else:
#         raise ValueError("unknown normalization type")



def get_activation(name):
    if name == "relu":
        return F.relu
    elif name == "mish":
        return F.mish
    elif name == "silu":
        return F.silu
    else:
        raise ValueError("unknown activation type")


class EMA():
    '''
        指数移动平均 ：Exponential Moving Average，
    '''
    def __init__(self,model,device,decay=0.999,ema_start =2000, ema_update_rate =1):
        self.ema_model = copy.deepcopy(model).to( device)
        self.decay = decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step =0

    #    self.shadow = {}
    # def register(self, name, val):
    #     self.shadow[name] = val.clone()
    def update_ema(self, current_model):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(current_model.state_dict())
            else:
                for current_params, ema_params in zip(current_model.parameters(), self.ema_model.parameters()):
                    old, new = ema_params.data, current_params.data
                    ema_params.data = old * self.decay + (1 - self.decay) * new
                    current_params.data = ema_params.data.clone()



    # def __call__(self, name, x):
    #     assert name in self.shadow
    #     new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
    #     self.shadow[name] = new_average.clone()
    #     return new_average
    # def __call__(self, current_model):
    #
    #     for current_params, ema_params in zip(current_model.parameters(), self.ema_model.parameters()):
    #         old, new = ema_params.data, current_params.data
    #         ema_params.data =  old * self.decay + (1 - self.decay) * new
    #         current_params.data = ema_params.data.clone()


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

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)