"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2022.12.24
    Description	:
            SRGAN 模型训练
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
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_read import ImageDatasetHighLow
from SRGAN_models import GeneratorResNet, DiscriminatorNet, FeatureExtractor


def train(opt):
    save_folder_image = os.path.join(opt.save_folder, r"SRGAN/images")
    save_folder_model = os.path.join(opt.save_folder, r"SRGAN/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    dataset_high = os.path.join(opt.folder_data, r"high")
    dataset_low = os.path.join(opt.folder_data, r"low")
    dataset = ImageDatasetHighLow(dataset_high, dataset_low)
    data_len = dataset.__len__()
    val_data_len = int(data_len * 0.25)
    train_set, val_set = torch.utils.data.random_split(dataset, [data_len - val_data_len, val_data_len])
    batch_size = opt.batch_size
    test_dataloader = DataLoader(dataset=val_set, num_workers=0, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorResNet().to(device)
    discriminator = DiscriminatorNet().to(device)

    load_models_path_gen = opt.load_models_path_gen
    load_models_path_dis = opt.load_models_path_dis
    # Load pretrained models
    if opt.load_models:
        generator.load_state_dict(torch.load(load_models_path_gen))
        discriminator.load_state_dict(torch.load(load_models_path_dis))

    # Set feature extractor to inference mode
    feature_extractor = FeatureExtractor().to(device)
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
    train_gen_losses, train_disc_losses, train_counter = [], [], []
    val_gen_losses, val_disc_losses = [], []
    # test_counter = [idx * len(train_dataloader.dataset) for idx in range(1, n_epochs + 1)]
    n_epochs = opt.epochs
    scale_factor = 4
    save_epoch = max(int(n_epochs // opt.save_epoch_n), 1)
    for epoch in range(n_epochs):
        # Training
        generator.train()
        discriminator.train()
        gen_loss, disc_loss = 0, 0
        for batch_idx, images_hl in tqdm(enumerate(train_dataloader), desc=f'Training Epoch {epoch}',
                                         total=int(len(train_dataloader))):
            images_l = images_hl["lr"].to(device)
            images_h = images_hl["hr"].to(device)

            # Adversarial ground truths
            # valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            # fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            valid = Variable(Tensor(np.ones((images_l.size(0), 1, 1, 1))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((images_l.size(0), 1, 1, 1))), requires_grad=False)

            # Train Generator
            optimizer_G.zero_grad()
            # Generate a high resolution image from low resolution input
            gen_hr = generator(images_l)
            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(images_h)
            loss_content = criterion_content(gen_features, real_features.detach())
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(images_h), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            gen_loss += loss_G.item()
            train_gen_losses.append(loss_G.item())
            disc_loss += loss_D.item()
            train_disc_losses.append(loss_D.item())
            train_counter.append(batch_idx * batch_size + images_l.size(0) + epoch * len(train_dataloader.dataset))
        # tqdm_bar.set_postfix(gen_loss=gen_loss / (batch_idx + 1), disc_loss=disc_loss / (batch_idx + 1))

        # Testing
        gen_loss, disc_loss = 0, 0
        with torch.no_grad():
            for batch_idx, images_hl in tqdm(enumerate(test_dataloader), desc=f'Validate Epoch {epoch}',
                                             total=int(len(test_dataloader))):
                generator.eval()
                discriminator.eval()

                # Configure model input
                imgs_lr = images_hl["lr"].to(device)
                imgs_hr = images_hl["hr"].to(device)

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
                if batch_idx == 0:
                    if (epoch == n_epochs - 1) or (epoch % save_epoch == 0):
                        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=scale_factor)
                        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
                        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                        img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
                        save_image(img_grid, os.path.join(save_folder_image, f"epoch_{epoch}.png"), normalize=False)

        val_gen_losses.append(gen_loss / len(test_dataloader))
        val_disc_losses.append(disc_loss / len(test_dataloader))

        # Save model checkpoints
        if (epoch == n_epochs - 1) or (epoch % save_epoch == 0):
            torch.save(generator.state_dict(), os.path.join(save_folder_model, f"epoch_{epoch}_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_folder_model, f"epoch_{epoch}_discriminator.pth"))

    plt.figure(figsize=(10, 7))
    plt.plot(train_gen_losses, color='blue', label='train gen losses')
    plt.plot(train_disc_losses, color='red', label='train disc losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, 'train loss.png'))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(val_gen_losses, color='blue', label='Validate gen losses')
    plt.plot(val_disc_losses, color='red', label='Validate disc losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, 'val loss.png'))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--folder_data', type=str, default='data/coco_sub', help='dataset path')
    parser.add_argument('--crop_img_w', type=int, default=128, help='randomly cropped image width')
    parser.add_argument('--crop_img_h', type=int, default=128, help='randomly cropped image height')
    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_path_gen', type=str, default=r"./working/SRGAN/models/discriminator.pth",
                        help='load model path')
    parser.add_argument('--load_models_path_dis', type=str, default=r"./working/SRGAN/models/generator.pth",
                        help='load model path')
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--save_epoch_n', type=int, default=5, help='number of saved epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--show_example', type=bool, default=True, help='show a validation example')
    args = parser.parse_args(args=[])  # 不添加args=[] kaggle会报错
    return args


if __name__ == '__main__':
    para = parse_args()
    para.folder_data = '../data/coco_sub'
    para.load_models = False
    para.epochs = 5
    train(para)
