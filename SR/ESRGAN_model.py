"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.02.19
    Description	:
            ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks   2018 CCVP
    Others		:  //其他内容说明
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
        vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])
        for param in vgg19_54.parameters():
            param.requires_grad = False
        self.vgg19_54 = vgg19_54

    def forward(self, img):
        return self.vgg19_54(img)


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


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        img_channels, img_height, img_width = input_shape
        # patch_h, patch_w = int(img_height / 2 ** 4), int(img_width / 2 ** 4)
        # self.output_shape = (1, patch_h, patch_w)
        self.output_shape = (1, 1, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        def discriminator_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
            return layer

        self.down = nn.Sequential(
            discriminator_block(64, 64, kernel_size=3, stride=2, padding=1),
            discriminator_block(64, 128, kernel_size=3, stride=1, padding=1),
            discriminator_block(128, 128, kernel_size=3, stride=2, padding=1),
            discriminator_block(128, 256, kernel_size=3, stride=1, padding=1),
            discriminator_block(256, 256, kernel_size=3, stride=2, padding=1),
            discriminator_block(256, 512, kernel_size=3, stride=1, padding=1),
            discriminator_block(512, 512, kernel_size=3, stride=2, padding=1),
        )
        # self.dense = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
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


class TrainerESRGAN:
    def __init__(self, device):
        # Set feature extractor to inference mode
        self.feature_extractor = FeatureExtractor().to(device)
        self.feature_extractor.eval()

        # Losses
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_content = torch.nn.L1Loss().to(device)
        self.criterion_pixel = torch.nn.L1Loss().to(device)

    def pre_train_generator(self, optimizer_G, img_gen, img_hr):
        optimizer_G.zero_grad()
        # Content loss
        loss_pixel = self.criterion_pixel(img_gen, img_hr)
        loss_pixel.backward()
        optimizer_G.step()

    def train_generator(self, optimizer_G, discriminator, img_gen, img_hr, real_labels):
        optimizer_G.zero_grad()
        # pixel loss
        loss_pixel = self.criterion_pixel(img_gen, img_hr)

        # Adversarial loss (relativistic average GAN)
        # Extract validity predictions from discriminator
        real_score = discriminator(img_hr).detach()
        fake_score = discriminator(img_gen)
        loss_GAN = self.criterion_GAN(fake_score - real_score.mean(0, keepdim=True), real_labels)

        # Content loss
        gen_features = self.feature_extractor(img_gen).detach()
        real_features = self.feature_extractor(img_hr)
        loss_content = self.criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + 5e-3 * loss_GAN + 1e-2 * loss_pixel
        loss_G.backward()

        optimizer_G.step()

        return loss_G

    def get_generate_loss(self, discriminator, img_gen, img_hr, real_labels):
        with torch.no_grad():
            loss_pixel = self.criterion_pixel(img_gen, img_hr)

            # Adversarial loss (relativistic average GAN)
            # Extract validity predictions from discriminator
            real_score = discriminator(img_hr).detach()
            fake_score = discriminator(img_gen)
            loss_GAN = self.criterion_GAN(fake_score - real_score.mean(0, keepdim=True), real_labels)

            # Content loss
            gen_features = self.feature_extractor(img_gen).detach()
            real_features = self.feature_extractor(img_hr)
            loss_content = self.criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = loss_content + 5e-3 * loss_GAN + 1e-2 * loss_pixel
            return loss_G

    def train_discriminator(self, optimizer_D, discriminator, img_gen, img_hr, real_labels, fake_labels):
        optimizer_D.zero_grad()

        real_score = discriminator(img_hr)
        fake_score = discriminator(img_gen.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = self.criterion_GAN(real_score - fake_score.mean(0, keepdim=True), real_labels)
        loss_fake = self.criterion_GAN(fake_score - real_score.mean(0, keepdim=True), fake_labels)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()

        optimizer_D.step()

        return loss_D

    def get_disc_loss(self, discriminator, img_gen, img_hr, real_labels, fake_labels):
        with torch.no_grad():
            real_score = discriminator(img_hr)
            fake_score = discriminator(img_gen.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = self.criterion_GAN(real_score - fake_score.mean(0, keepdim=True), real_labels)
            loss_fake = self.criterion_GAN(fake_score - real_score.mean(0, keepdim=True), fake_labels)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            return loss_D
