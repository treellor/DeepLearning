"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.03.05
    Description	:
            Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data   2021
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from math import log


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()

        self.res_scale = res_scale

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels + 0 * out_channels, out_channels, 3, padding=1, bias=True),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels + 1 * out_channels, out_channels, 3, padding=1, bias=True),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels + 2 * out_channels, out_channels, 3, padding=1, bias=True),
                                    nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels + 3 * out_channels, out_channels, 3, padding=1, bias=True),
                                    nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels + 4 * out_channels, in_channels, 3, padding=1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(torch.cat((x, out1), 1))
        out3 = self.layer3(torch.cat((x, out1, out2), 1))
        out4 = self.layer4(torch.cat((x, out1, out2, out3), 1))
        out5 = self.layer5(torch.cat((x, out1, out2, out3, out4), 1))
        return out5.mul(self.res_scale) + x


class RRDB(nn.Module):
    def __init__(self, in_channels, out_channels=32, res_scale=0.2):
        super(RRDB, self).__init__()
        self.res_scale = res_scale

        self.dense_blocks = nn.Sequential(ResidualDenseBlock(in_channels, out_channels, res_scale),
                                          ResidualDenseBlock(in_channels, out_channels, res_scale),
                                          ResidualDenseBlock(in_channels, out_channels, res_scale)
                                          )

    def forward(self, x):
        out = self.dense_blocks(x)
        return out.mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, in_channels, filters=64, scale_factor=4, n_basic_block=23):
        super(GeneratorRRDB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)

        basic_block_layer = []
        for _ in range(n_basic_block):
            basic_block_layer += [RRDB(in_channels=filters, out_channels=filters)]
        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        up_sample_layers = []

        up_sample_block_num = int(log(scale_factor, 2))
        for _ in range(up_sample_block_num):
            up_sample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.up_sampling = nn.Sequential(*up_sample_layers)

        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.basic_block(out1)
        out3 = self.conv2(out2)
        out = self.up_sampling(out1 + out3)
        out = self.conv3(out)
        return out


class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, input_shape, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()

        img_channels, img_height, img_width = input_shape
        self.output_shape = (1, img_height, img_width)

        self.skip_connection = skip_connection
        spectral_norm = torch.nn.utils.spectral_norm

        self.conv0 = nn.Conv2d(img_channels, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # up_sample
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # extra
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))

        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

