import numpy as np
import os \
    # , math, sys
# import glob, itertools
import argparse, random

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision.models import vgg19
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
# from PIL import Image
from tqdm import tqdm

from data_read import DatasetHighLow
from SRGAN_models import GeneratorResNet, DiscriminatorNet, FeatureExtractor


def train(dataset_path, n_epochs=2, load_pretrained_model=False, path_gen_model=None, path_dis_model=None):

    os.makedirs(r'.\images\SRGAN',exist_ok=True)
    os.makedirs(r'.\models\SRGAN',exist_ok=True)

    # Get train/test dataset
    # name of the dataset
    # dataset_path = "../input/celeba-dataset/img_align_celeba/img_align_celeba"
    # size of the batches
    batch_size = 4
    # epoch from which to start lr decay
    # decay_epoch = 100
    # number of cpu threads to use during batch generation
    # n_cpu = 8
    # high res. image height
    # hr_height = 256
    # high res. image width
    # hr_width = 256
    # number of image channels
    # channels = 3

    dataset = DatasetHighLow(r"D:\project\Pycharm\DeepLearning\data\coco125\high",
                             r"D:\project\Pycharm\DeepLearning\data\coco125\low")

    train_set, val_set = torch.utils.data.random_split(dataset, [dataset.__len__() - 32, 32])
    test_dataloader = DataLoader(dataset=val_set, num_workers=0, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = GeneratorResNet().to(device)
    discriminator = DiscriminatorNet().to(device)

    # Load pretrained models
    if load_pretrained_model:
        # generator.load_state_dict(torch.load("../input/single-image-super-resolution-gan-srgan-pytorch/saved_models/generator.pth"))
        # discriminator.load_state_dict(torch.load("../input/single-image-super-resolution-gan-srgan-pytorch/saved_models/discriminator.pth"))
        generator.load_state_dict(torch.load(path_gen_model))
        discriminator.load_state_dict(torch.load(path_dis_model))

    feature_extractor = FeatureExtractor().to(device)
    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)

    # Optimizers
    # adam: learning rate
    lr = 0.00008
    # adam: decay of first order momentum of gradient
    b1 = 0.5
    # adam: decay of second order momentum of gradient
    b2 = 0.999
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # n_epochs = 2
    train_gen_losses, train_disc_losses, train_counter = [], [], []
    test_gen_losses, test_disc_losses = [], []
    test_counter = [idx * len(train_dataloader.dataset) for idx in range(1, n_epochs + 1)]
    # number of epochs of training
    count = 1
    for epoch in range(n_epochs):
        # Training
        gen_loss, disc_loss = 0, 0
        for batch_idx, imgs in tqdm(enumerate(train_dataloader), desc=f'Training Epoch {epoch}',
                                    total=int(len(train_dataloader))):
            generator.train()
            discriminator.train()

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            # Adversarial ground truths
            # valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            # fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            valid = Variable(Tensor(np.ones((imgs_lr.size(0), 1, 1, 1))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), 1, 1, 1))), requires_grad=False)

            # Train Generator
            optimizer_G.zero_grad()
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            gen_loss += loss_G.item()
            train_gen_losses.append(loss_G.item())
            disc_loss += loss_D.item()
            train_disc_losses.append(loss_D.item())
            train_counter.append(batch_idx * batch_size + imgs_lr.size(0) + epoch * len(train_dataloader.dataset))
           # tqdm_bar.set_postfix(gen_loss=gen_loss / (batch_idx + 1), disc_loss=disc_loss / (batch_idx + 1))

        # Testing
        gen_loss, disc_loss = 0, 0

        for batch_idx, imgs in tqdm(enumerate(test_dataloader), desc=f'Testing Epoch {epoch}',
                                    total=int(len(test_dataloader))):
            generator.eval()
            discriminator.eval()
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            # Adversarial ground truths

            # valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            # fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            valid = Variable(Tensor(np.ones((imgs_lr.size(0), 1, 1, 1))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), 1, 1, 1))), requires_grad=False)

            # Eval Generator
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # Adversarial loss

            valid_test = Variable(Tensor(np.ones((gen_hr.size(0), 1, 1, 1))), requires_grad=False)
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid_test)
            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            # Eval Discriminator
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            gen_loss += loss_G.item()
            disc_loss += loss_D.item()
          #  tqdm_bar.set_postfix(gen_loss=gen_loss / (batch_idx + 1), disc_loss=disc_loss / (batch_idx + 1))

            # Save image grid with upsampled inputs and SRGAN outputs
            if random.uniform(0, 1) < 0.1:
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
                gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
                save_image(img_grid, f"images\SRGAN\{batch_idx}.png", normalize=False)

        test_gen_losses.append(gen_loss / len(test_dataloader))
        test_disc_losses.append(disc_loss / len(test_dataloader))

        # Save model checkpoints
        if np.argmin(test_gen_losses) == len(test_gen_losses) - 1:
            torch.save(generator.state_dict(), "models\SRGAN\generator.pth")
            torch.save(discriminator.state_dict(), "models\SRGAN\discriminator.pth")

    plt.figure(figsize=(10, 7))
    plt.plot(train_gen_losses, color='orange', label='train gen losses')
    plt.plot(train_disc_losses, color='red', label='train disc losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('../output/loss.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(test_gen_losses, color='orange', label='test gen losses')
    plt.plot(test_disc_losses, color='red', label='test disc losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('../output/loss.png')
    plt.show()


if __name__ == '__main__':
    random.seed(42)
    dataset_path = "../data/T91/"
    train(dataset_path)
