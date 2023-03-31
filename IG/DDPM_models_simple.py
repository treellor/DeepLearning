"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.3.16
    Description	:
        GAN最基本模型
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

import copy
from functools import partial


import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self,img_shape, n_steps, num_units=128):
        super(UNet, self).__init__()

        # 定义网络结构
        self.Conv1 = nn.Conv2d(img_shape[0], 64, 9, 1, 4)
        self.Conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.Conv3 = nn.Conv2d(32, img_shape[0], 5, 1, 2)
        self.Relu = nn.ReLU()

        # self.linears = nn.ModuleList(
        #     [
        #         nn.Linear(3, num_units),
        #         nn.ReLU(),
        #         nn.Linear(num_units, num_units),
        #         nn.ReLU(),
        #         nn.Linear(num_units, num_units),
        #         nn.ReLU(),
        #         nn.Linear(num_units, 3),
        #     ]
        # )
        # self.step_embeddings = nn.ModuleList(
        #     [
        #         nn.Embedding(n_steps, num_units),
        #         nn.Embedding(n_steps, num_units),
        #         nn.Embedding(n_steps, num_units),
        #     ]
        # )

    def forward(self, x, t):
        #         x = x_0
        # for idx, embedding_layer in enumerate(self.step_embeddings):
        #     t_embedding = embedding_layer(t)
        #     x = self.linears[2 * idx](x)  # 选取的是线性层
        #     x += t_embedding
        #     x = self.linears[2 * idx + 1](x)
        #
        # x = self.linears[-1](x)

        out = self.Relu(self.Conv1(x))
        out = self.Relu(self.Conv2(out))
        out = self.Conv3(out)

        return out


class MLPDiffusion(nn.Module):
    def __init__(self,  img_shape,  n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()
        # 扩散网络
        self.model = UNet(img_shape,n_steps, num_units)
        # 生成次数
        self.n_steps = n_steps
        # 初始化相关参数,不参与梯度运算
        # 制定每一步的beta
        betas = torch.linspace(-6, 6, n_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        # 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_prod", alphas_prod)
        self.register_buffer("alphas_bar_sqrt", alphas_bar_sqrt)
        self.register_buffer("one_minus_alphas_bar_log", one_minus_alphas_bar_log)
        self.register_buffer("one_minus_alphas_bar_sqrt", one_minus_alphas_bar_sqrt)

    def forward(self, x_0):

        """对任意时刻t进行采样计算loss"""
        batch_size = x_0.shape[0]
        # 对一个batchsize样本生成随机的时刻t，t变得随机分散一些，一个batch size里面覆盖更多的t
        t = torch.randint(0, self.n_steps, size=(batch_size,), device=x_0.device)
        t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # t的形状（bz,1,1,1）

        # x0的系数，根号下(alpha_bar_t)
        a = self.alphas_bar_sqrt[t]
        # eps的系数,根号下(1-alpha_bar_t)
        aml = self.one_minus_alphas_bar_sqrt[t]
        # 生成随机噪音eps
        e = torch.randn_like(x_0)
        # 构造模型的输入
        x = x_0 * a + e * aml

        # 送入模型，得到t时刻的随机噪声预测值
        output = self.model(x, t.squeeze(-1).squeeze(-1).squeeze(-1))
        # 与真实噪声一起计算误差，求平均值
        loss = F.mse_loss(output, e)

        return loss

    @torch.no_grad()
    def q_x(self, x_0, t):
        """可以基于x[0]得到任意时刻t的x[t]"""
        noise = torch.randn_like(x_0)
        alphas_t = self.alphas_bar_sqrt[t]
        alphas_1_m_t = self.one_minus_alphas_bar_sqrt[t]
        return alphas_t * x_0 + alphas_1_m_t * noise  # 在x[0]的基础上添加噪声

    @torch.no_grad()
    def p_sample_loop(self, shape, n_steps):
        """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
        cur_x = torch.randn(shape)
        x_seq = [cur_x]
        for i in reversed(range(n_steps)):
            cur_x = self.p_sample(cur_x, i)
            x_seq.append(cur_x)
        return x_seq

    @torch.no_grad()
    def p_sample(self, x, t):
        """从x[T]采样t时刻的重构值"""
        t = torch.tensor([t])

        coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]

        eps_theta = self.model(x, t)

        mean = (1 / (1 - self.betas[t]).sqrt()) * (x - (coeff * eps_theta))

        z = torch.randn_like(x)
        sigma_t = self.betas[t].sqrt()

        sample = mean + sigma_t * z

        return sample






