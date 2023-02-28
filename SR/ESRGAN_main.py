"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.02.25
    Description	:
            ESRGAN 模型训练
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
     #import numpy as np
#from torch.autograd import Variable
    # self.optimizer_generator = Adam(self.generator.parameters(), lr=self.lr, betas=(config.b1, config.b2),
    #                                 weight_decay=config.weight_decay)
    # self.optimizer_discriminator = Adam(self.discriminator.parameters(), lr=self.lr, betas=(config.b1, config.b2),
    #                                     weight_decay=config.weight_decay)
    #
    # self.lr_scheduler_generator = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_generator, self.decay_iter)
    # self.lr_scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_discriminator,
    #                                                                        self.decay_iter)


        # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    # fake_labels = Variable(Tensor(np.zeros((images_l.size(0), 1, 1, 1))), requires_grad=False)
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_read import ImageDatasetHighLow
from ESRGAN_model import GeneratorNet, DiscriminatorNet, PerceptualLoss
from PIL import Image

def save_model(save_path, model, optimizer, epoch_n):
    torch.save({"model_dict": model.state_dict(), "optimizer_dict": optimizer.state_dict(), "epoch_n": epoch_n},
               save_path)
    # print(model.state_dict()['Conv1.weight'])


def load_model(save_path, model, optimizer=None):
    model_data = torch.load(save_path)
    model.load_state_dict(model_data["model_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(model_data["optimizer_dict"])
    epoch_n = model_data["epoch_n"]
    return epoch_n


def train(opt):
    # 创建文件夹
    save_folder_image = os.path.join(opt.save_folder, r"ESRGAN/images")
    save_folder_model = os.path.join(opt.save_folder, r"ESRGAN/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)
    # 读取数据
    dataset_high = os.path.join(opt.folder_data, r"high")
    dataset_low = os.path.join(opt.folder_data, r"low")
    batch_size = opt.batch_size
    dataset = ImageDatasetHighLow(dataset_high, dataset_low)
    data_len = dataset.__len__()
    val_data_len = batch_size * 4
    train_set, val_set = torch.utils.data.random_split(dataset, [data_len - val_data_len, val_data_len])

    test_dataloader = DataLoader(dataset=val_set, num_workers=0, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True)

    # Initialize generator and discriminator
    upsampling_n = opt.upsampling_n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = GeneratorNet(in_channels=3, out_channels=3, scale_factor=upsampling_n).to(device)
    discriminator = DiscriminatorNet().to(device)

    # Optimizers:   adam: learning rate
    lr = 0.00008    # adam: decay of first order momentum of gradient
    b1 = 0.5        # adam: decay of second order momentum of gradient
    b2 = 0.999
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Load pretrained models
    load_models_path_gen = opt.load_models_path_gen
    load_models_path_dis = opt.load_models_path_dis
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(load_models_path_gen, generator, optimizer_G)
        load_model(load_models_path_dis, discriminator, optimizer_D)

    # Losses
    dis_criterion = torch.nn.MSELoss().to(device)
    content_criterion = nn.L1Loss().to(device)
    perception_criterion = PerceptualLoss().to(device).eval()
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)

    # 读取用去显示图像保存
    show_data1 = dataset[0]
    show_data2 = dataset[1]
    show_data3 = dataset[2]
    show_data4 = dataset[3]
    show_image_hr = torch.stack([show_data1["hr"], show_data2["hr"], show_data3["hr"], show_data4["hr"]], 0).to(device)
    show_image_lr = torch.stack([show_data1["lr"], show_data2["lr"], show_data3["lr"], show_data4["lr"]], 0).to(device)

    n_epochs = opt.epochs
    save_epoch = opt.save_epoch
    save_epoch.add(n_epochs)
    train_gen_losses, train_disc_losses = [], []
    val_gen_losses, val_disc_losses = [], []

    for epoch in range(n_epochs):
        # Training
        generator.train()
        discriminator.train()
        train_gen_loss, train_disc_loss = 0, 0
        count_train = 0
        for batch_idx, images_hl in tqdm(enumerate(train_dataloader), desc=f'Training Epoch {epoch}',
                                         total=int(len(train_dataloader))):
            count_train += 1
            images_l = images_hl["lr"].to(device)
            images_h = images_hl["hr"].to(device)

            # Adversarial ground truths
            real_labels = torch.ones((images_l.size(0), 1, 1, 1)).to(device)
            fake_labels = torch.zeros((images_l.size(0), 1, 1, 1)).to(device)

            ##########################
            #   training generator   #
            ##########################
            optimizer_D.zero_grad()
            gen_images = generator(images_l)

            perceptual_loss = perception_criterion(images_h, gen_images)
            content_loss = content_criterion(gen_images, images_h)

            #score_real = discriminator(images_h)
            # score_fake = discriminator(gen_images)
            # loss_real = dis_criterion(score_fake, real_labels)
          #  discriminator_rf = score_real - score_fake.mean()
          #  discriminator_fr = score_fake - score_real.mean()
          #
          #   adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
          #  # adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
          #   adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2
          #  adversarial_loss = adversarial_criterion(discriminator_rf, fake_labels)
            generator_loss = perceptual_loss * 0.1 + content_loss #+ loss_real

            generator_loss.backward()
            optimizer_G.step()

            ##########################
            # training discriminator #
            ##########################
            optimizer_D.zero_grad()

            score_real = discriminator(images_h)
            score_fake = discriminator(gen_images.detach())

            loss_real = dis_criterion(score_real, real_labels)
            loss_fake = dis_criterion(score_fake, fake_labels)
            discriminator_loss = (loss_real + loss_fake) / 2
            #    discriminator_rf = score_real - score_fake.mean()
            #    discriminator_fr = score_fake - score_real.mean()
            #    adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
            #    adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
            # discriminator_loss = (loss_real + loss_fake) / 2
            #
            # self.optimizer_discriminator.zero_grad()

           # discriminator_rf = score_real - score_fake.mean()
           # discriminator_fr = score_fake - score_real.mean()

           # adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
           # adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
           # discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2
           # discriminator_loss  =  adversarial_criterion(discriminator_fr, real_labels)

            discriminator_loss.backward()
            optimizer_D.step()

            train_gen_loss += generator_loss.item()
            train_disc_loss += discriminator_loss.item()

        train_gen_losses.append(train_gen_loss / count_train)
        train_disc_losses.append(train_disc_loss / count_train)

        # Testing
        val_gen_loss, val_disc_loss = 0, 0
        count_val = 0
        with torch.no_grad():
            for batch_idx, images_hl in tqdm(enumerate(test_dataloader), desc=f'Validate Epoch {epoch}',
                                             total=int(len(test_dataloader))):
                count_val += 1
                generator.eval()
                discriminator.eval()

                # Configure model input
                imgs_lr = images_hl["lr"].to(device)
                imgs_hr = images_hl["hr"].to(device)

                # Adversarial ground truths
                real_labels = torch.ones((imgs_lr.size(0), 1, 1, 1)).to(device)
                fake_labels = torch.zeros((imgs_lr.size(0), 1, 1, 1)).to(device)

                # Eval Generator :Generate a high resolution image from low resolution input
                gen_hr = generator(imgs_lr)

                perceptual_loss = perception_criterion(imgs_hr, gen_hr)
                content_loss = content_criterion(gen_hr, imgs_hr)
                generator_loss = perceptual_loss + content_loss

                score_real = discriminator(imgs_hr)
                score_fake = discriminator(gen_hr.detach())
                adversarial_loss_rf = dis_criterion(score_fake, fake_labels)
                adversarial_loss_fr = dis_criterion(score_real, real_labels)
                discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

                val_gen_loss += generator_loss.item()
                val_disc_loss += discriminator_loss.item()

            val_gen_losses.append(val_gen_loss / count_val)
            val_disc_losses.append(val_disc_loss / count_val)

        # Save image grid with upsampled inputs and SRGAN outputs
        if epoch + 1 in save_epoch:
            current_epoch = epoch + 1 + trained_epoch
            generator.eval()
            gen_hr = generator(show_image_lr)
            imgs_lr = nn.functional.interpolate(show_image_lr, scale_factor=upsampling_n)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(show_image_hr, nrow=1, normalize=True)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)

            img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
            image_path_temp =os.path.join(save_folder_image, f"epoch_{current_epoch}.png")
            save_image(img_grid, image_path_temp, normalize=False)
            img = Image.open(image_path_temp)
            plt.imshow(img)
            plt.show()

            # Save model checkpoints
            save_model(os.path.join(save_folder_model, f"epoch_{current_epoch}_generator.pth"),
                       generator, optimizer_G, current_epoch)
            save_model(os.path.join(save_folder_model, f"epoch_{current_epoch}_discriminator.pth"),
                       discriminator, optimizer_D, current_epoch)

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
    parser.add_argument('--upsampling_n', type=int, default=4, help='the size of upsampling')
    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_path_gen', type=str, default=r"./working/SRGAN/models/discriminator.pth",
                        help='load model path')
    parser.add_argument('--load_models_path_dis', type=str, default=r"./working/SRGAN/models/generator.pth",
                        help='load model path')
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--save_epoch', type=set, default=set(), help='number of saved epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--show_example', type=bool, default=True, help='show a validation example')
    args = parser.parse_args(args=[])  # 不添加args=[] kaggle会报错
    return args


if __name__ == '__main__':
    para = parse_args()
    para.folder_data = '../data/coco_sub'
    para.epochs = 3
    # para.save_epoch = set(range(1, 10, 5))
    para.load_models = False
    para.load_models_path_gen = r"./working/ESRGAN/models/epoch_5850_generator.pth"
    para.load_models_path_dis = r"./working/ESRGAN/models/epoch_5850_discriminator.pth"
    train(para)
    # run(para)
