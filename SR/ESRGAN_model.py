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

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, res_scale=0.2):
        super(DenseBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels + 0 * out_channels, out_channels, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels + 1 * out_channels, out_channels, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels + 2 * out_channels, out_channels, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels + 3 * out_channels, out_channels, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels + 4 * out_channels, in_channels, 3, padding=1, bias=True), nn.LeakyReLU())

        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = DenseBlock(in_channels, out_channels,res_scale)
        self.layer2 = DenseBlock(in_channels, out_channels,res_scale)
        self.layer3 = DenseBlock(in_channels, out_channels,res_scale)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


def upsample_block(in_channels, scale_factor=2):
    block = []
    for _ in range(scale_factor // 2):
        block += [
            nn.Conv2d(in_channels, in_channels * 4, 1),
            nn.PixelShuffle(2),
            nn.ReLU()
        ]
    return nn.Sequential(*block)


class GeneratorNet(nn.Module):
    def __init__(self, in_channels, out_channels, nf=64, gc=32, scale_factor=4, n_basic_block=23):
        super(GeneratorNet, self).__init__()

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, nf, 3), nn.ReLU())

        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += [ResidualDenseBlock(nf, gc)]

        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, 3), nn.ReLU())
        self.upsample = upsample_block(nf, scale_factor=scale_factor)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, 3), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, out_channels, 3), nn.ReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.basic_block(x1)
        x = self.conv2(x)
        x = self.upsample(x + x1)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, kernel_size=3, padding=1):
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


class PerceptualLoss(nn.Module):
    """
    感知损失
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss