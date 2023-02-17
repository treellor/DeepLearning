"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.02.17
    Description	:
    Reference	:

    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, input_channel=64, output_channel=64, kernel_size=3, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(output_channel),  # nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(output_channel)
        )

    def forward(self, x0):
        x1 = self.layer(x0)
        return x0 + x1


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16, n_upsampling=4):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64, 64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)  # nn.BatchNorm2d(64, 0.8)
        )

        # Upsampling layers 4倍上采样
        # nn.Upsample(scale_factor=2),
        if n_upsampling ==4:
            self.upsampling = nn.Sequential(
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            )
        else:
            self.upsampling = nn.Sequential(
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            )

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.down = nn.Sequential(
            DiscriminatorBlock(64, 64, stride=2, padding=1),
            DiscriminatorBlock(64, 128, stride=1, padding=1),
            DiscriminatorBlock(128, 128, stride=2, padding=1),
            DiscriminatorBlock(128, 256, stride=1, padding=1),
            DiscriminatorBlock(256, 256, stride=2, padding=1),
            DiscriminatorBlock(256, 512, stride=1, padding=1),
            DiscriminatorBlock(512, 512, stride=2, padding=1),
        )
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.down(x)
        x = self.dense(x)
        return x


'''
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


'''

if __name__ == '__main__':
    g = GeneratorResNet()
    # a = torch.rand([1, 3, 64, 64])
    # print(g(a).shape)
    # d = Discriminator()
    # b = torch.rand([2, 3, 512, 512])
    # print(d(b).shape)
