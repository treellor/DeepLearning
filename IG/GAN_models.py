"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.2.28
    Description	:
        GAN最基本模型
    Reference	:
        Generative Adversarial Nets.    2014     Ian J. Goodfellow
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn


class DDPM(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(DDPM, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x, t):
        #         x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x