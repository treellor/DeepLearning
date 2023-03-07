"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.2.28
    Description	:
    Reference	:
        Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network    2017
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from math import log


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x0):
        x1 = self.layer(x0)
        return x0 + x1


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, n_residual_blocks=16, scale_factor=4):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # Up_Sampling layers
        block = []
        up_sample_block_num = int(log(scale_factor, 2))
        for _ in range(up_sample_block_num):
            block += [
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        self.up_sampling = nn.Sequential(*block)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.up_sampling(out)
        out = self.conv3(out)
        return out


class DiscriminatorNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        # patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        # self.output_shape = (1, patch_h, patch_w)
        self.output_shape = (1, 1, 1)

        def discriminator_block(in_channel, out_channels, first_block=False):
            block_layers = [nn.Conv2d(in_channel, out_channels, kernel_size=3, stride=1, padding=1)]
            if not first_block:
                block_layers.append(nn.BatchNorm2d(out_channels))
            block_layers.append(nn.LeakyReLU(0.2, inplace=True))
            block_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
            block_layers.append(nn.BatchNorm2d(out_channels))
            block_layers.append(nn.LeakyReLU(0.2, inplace=True))
            return block_layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        # layers.append(nn.Conv2d(in_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.dense(self.model(x))
        return out


class TrainerSRGAN:
    def __init__(self, device):
        # Set feature extractor to inference mode
        self.feature_extractor = FeatureExtractor().to(device)
        self.feature_extractor.eval()

        # Losses
        self.criterion_GAN = torch.nn.MSELoss().to(device)
        self.criterion_content = torch.nn.L1Loss().to(device)

    def pre_train_generator(self, optimizer_G, gen_hr, images_h):
        optimizer_G.zero_grad()
        # Content loss
        loss_pixel = self.criterion_content(gen_hr, images_h)
        loss_pixel.backward()
        optimizer_G.step()

    def train_generator(self, optimizer_G, discriminator, gen_hr, images_h, real_labels):
        optimizer_G.zero_grad()
        # Adversarial loss
        loss_GAN = self.criterion_GAN(discriminator(gen_hr), real_labels)

        # Content loss
        gen_features = self.feature_extractor(gen_hr)
        real_features = self.feature_extractor(images_h)
        loss_content = self.criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optimizer_G.step()

        return loss_G

    def get_generate_loss(self, discriminator, gen_hr, images_h, real_labels):
        with torch.no_grad():
            # Adversarial loss
            loss_GAN = self.criterion_GAN(discriminator(gen_hr), real_labels)

            # Content loss
            gen_features = self.feature_extractor(gen_hr)
            real_features = self.feature_extractor(images_h)
            loss_content = self.criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN
            return loss_G

    def train_discriminator(self, optimizer_D, discriminator, gen_hr, images_h, real_labels, fake_labels):
        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = self.criterion_GAN(discriminator(images_h), real_labels)
        loss_fake = self.criterion_GAN(discriminator(gen_hr.detach()), fake_labels)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        return loss_D

    def get_disc_loss(self, discriminator, gen_hr, images_h, real_labels, fake_labels):
        with torch.no_grad():
            # Loss of real and fake images
            loss_real = self.criterion_GAN(discriminator(images_h), real_labels)
            loss_fake = self.criterion_GAN(discriminator(gen_hr.detach()), fake_labels)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            return loss_D


if __name__ == '__main__':
    g = GeneratorResNet()
    # a = torch.rand([1, 3, 64, 64])
    # print(g(a).shape)
    # d = Discriminator()
    # b = torch.rand([2, 3, 512, 512])
    # print(d(b).shape)
