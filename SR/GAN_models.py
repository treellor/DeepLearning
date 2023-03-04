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


class Generator(nn.Module):
    def __init__(self, seq_length=128, img_shape=(3, 64, 64)):
        super(Generator, self).__init__()

        def block(in_len, out_len, normalize=True):
            layers = [nn.Linear(in_len, out_len)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_len, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.img_shape = img_shape
        self.model = nn.Sequential(*block(seq_length, 128, normalize=False),
                                   *block(128, 256),
                                   *block(256, 512),
                                   *block(512, 1024),
                                   nn.Linear(1024, int(np.prod(self.img_shape))),
                                   nn.Tanh()
                                   )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
