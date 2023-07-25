"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.02.28
    Description	:
            GAN 模型训练
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""

import os
import numpy as np
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from GAN_models import Generator, Discriminator
from utils.data_read import ImageDatasetSingle
from utils.utils import load_model, save_model


def train(opt):
    save_folder_image = os.path.join(opt.save_folder, r"GAN/images")
    save_folder_model = os.path.join(opt.save_folder, r"GAN/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    dataset_train = ImageDatasetSingle(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w)
    dataloader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    img_shape = (opt.img_channels, opt.img_h, opt.img_w)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    generator = Generator(seq_length=opt.seq_length, img_shape=img_shape).to(device)
    discriminator = Discriminator(img_shape=img_shape).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Load pretrained models
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_path_gen, generator, optimizer_G)
        load_model(opt.load_models_path_dis, discriminator, optimizer_D)

    # Losses
    adversarial_loss = torch.nn.BCELoss().to(device)

    n_epochs = opt.epochs

    for epoch in range(trained_epoch+1, trained_epoch + n_epochs+1):
        # Training
        generator.train()
        discriminator.train()

        for batch_idx, img_real in tqdm(enumerate(dataloader_train), desc=f'Training Epoch {epoch}',
                                        total=int(len(dataloader_train))):
            img_real = img_real.to(device)

            # Adversarial ground truths
            valid = torch.ones((img_real.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((img_real.size(0), 1), requires_grad=False).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (img_real.shape[0], opt.seq_length))))
            #z = torch.tensor(    np.random.normal(0, 1, (img_real.shape[0], opt.seq_length))).to(device)
            # Generate a batch of images
            img_gen = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(img_gen), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(img_real), valid)
            fake_loss = adversarial_loss(discriminator(img_gen.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Save image
            if epoch % opt.save_epoch_rate == 0 or (epoch == (trained_epoch + n_epochs)):
                if batch_idx == 0:

                    save_image(img_gen.data[:opt.batch_size],
                               os.path.join(save_folder_image, f"epoch_{epoch}.png"),
                               nrow=8, normalize=False)
                    # Save model checkpoints
                    save_model(os.path.join(save_folder_model, f"epoch_{epoch}_generator.pth"),
                               generator, optimizer_G, epoch)
                    save_model(os.path.join(save_folder_model, f"epoch_{epoch}_discriminator.pth"),
                               discriminator, optimizer_D, epoch)


def run(opt):
    save_folder_image = os.path.join(opt.save_folder, r"GAN/results")
    os.makedirs(save_folder_image, exist_ok=True)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    img_shape = (opt.img_channels, opt.img_h, opt.img_w)

    generator = Generator(seq_length=opt.seq_length, img_shape=img_shape).to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # Load pretrained models
    load_model(opt.load_models_path_gen, generator, optimizer_G)

    generator.eval()
    img_n = para.batch_size
    z = Variable(Tensor(np.random.normal(0, 1, (img_n, opt.seq_length))))
    img_gen = generator(z)
    save_image(img_gen.data[:img_n], os.path.join(save_folder_image, f"results.png"), nrow=10, normalize=False)


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--data_folder', type=str, default='data/coco_sub', help='dataset path')
    parser.add_argument('--img_channels', type=int, default=3, help='the channel of the image')
    parser.add_argument('--img_w', type=int, default=64, help='image width')
    parser.add_argument('--img_h', type=int, default=64, help='image height')

    parser.add_argument('--batch_size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--seq_length", type=int, default=100, help="the length of the noise sequence")
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--save_epoch', type=set, default=set(), help='number of saved epochs')

    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_path_gen', type=str, default=r"./working/SRGAN/models/discriminator.pth",
                        help='load model path')
    parser.add_argument('--load_models_path_dis', type=str, default=r"./working/SRGAN/models/generator.pth",
                        help='load model path')

    args = parser.parse_args(args=[])  # 不添加args=[] kaggle会报错
    return args


if __name__ == '__main__':

    para = parse_args()
    para.data_folder = '../data/sanguo7'
    para.seq_length = 128
    para.img_channels = 3
    para.img_w = 96
    para.img_h = 120
    para.batch_size = 20


    is_train = False
    if is_train:
        para.epochs = 10
        para.save_epoch_rate = 5
        para.load_models = True
        para.load_models_path_gen = r"./working/GAN/models/epoch_10_generator.pth"
        para.load_models_path_dis = r"./working/GAN/models/epoch_10_discriminator.pth"
        train(para)
    else:
        para.load_models = True
        para.load_models_path_gen = r"./working/GAN/models/epoch_20_generator.pth"
        para.load_models_path_dis = r"./working/GAN/models/epoch_20_discriminator.pth"
        run(para)