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
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_read import ImageDatasetSingleToPair
from RealESRGAN_models import GeneratorRRDB, UNetDiscriminatorSN
from utils.feature_extraction import FeatureVGG19
from utils.utils import save_model, load_model, AverageMeter, calc_psnr

class TrainerRealESRGAN:
    def __init__(self, device):
        # Set feature extractor to inference mode
        self.feature_extractor = FeatureVGG19().to(device)
        self.feature_extractor.eval()

        # Losses
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_content = torch.nn.L1Loss().to(device)
        self.criterion_pixel = torch.nn.L1Loss().to(device)

    def pre_train_generator(self, optimizer_G, img_gen, img_hr):
        """
            warmup
        :param optimizer_G:
        :param img_gen:
        :param img_hr:
        :return:
        """
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


def train(opt):
    # 创建文件夹
    save_folder_image = os.path.join(opt.save_folder, r"RealESRGAN/images")
    save_folder_model = os.path.join(opt.save_folder, r"RealESRGAN/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    # 读取数据
    dataset = ImageDatasetSingleToPair(opt.data_folder, img_H=opt.hr_height, img_W=opt.hr_width, scale_factor=4)
    img_shape = tuple(dataset[0]['ref'].shape)

    # 数据分成两份
    data_len = dataset.__len__()
    val_data_len = opt.batch_size * 2
    train_set, val_set = torch.utils.data.random_split(dataset, [data_len - val_data_len, val_data_len])
    val_dataloader = DataLoader(dataset=val_set, num_workers=0, batch_size=opt.batch_size, shuffle=False)
    train_dataloader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    generator = GeneratorRRDB(in_channels=img_shape[0], filters=32, scale_factor=opt.scale_factor).to(device)
    discriminator = UNetDiscriminatorSN(img_shape).to(device)

    # Optimizers:   adam: learning rate
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))  # lr = 0.00008
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Load pretrained models
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_path_gen, generator, optimizer_G)
        load_model(opt.load_models_path_dis, discriminator, optimizer_D)

    # 定义训练
    trainer = TrainerRealESRGAN(device)

    n_epochs = opt.epochs

    train_gen_losses, train_disc_losses, train_psnr_all = [], [], []
    val_gen_losses, val_disc_losses, val_psnr_all = [], [], []

    for epoch in range(trained_epoch + 1, n_epochs + trained_epoch + 1):

        generator.train()
        discriminator.train()

        epoch_train_gen = AverageMeter()
        epoch_train_dis = AverageMeter()
        epoch_train_psnr = AverageMeter()
        for batch_idx, images_hl in tqdm(enumerate(train_dataloader), desc=f'Training Epoch {epoch}',
                                         total=int(len(train_dataloader))):
            img_lr = images_hl["test"].to(device)
            img_hr = images_hl["ref"].to(device)

            # Adversarial ground truths
            real_labels = torch.ones((img_hr.size(0), *discriminator.output_shape)).to(device)
            fake_labels = torch.zeros((img_hr.size(0), *discriminator.output_shape)).to(device)

            ##########################
            #   training generator   #
            ##########################
            # Generate a high resolution image from low resolution input
            img_gen = generator(img_lr)

            if epoch <= opt.warm_epochs + trained_epoch:
                trainer.pre_train_generator(optimizer_G, img_gen, img_hr)
                continue

            loss_G = trainer.train_generator(optimizer_G, discriminator, img_gen, img_hr, real_labels)

            train_pnsr = calc_psnr(img_gen.detach(), img_hr)

            epoch_train_gen.update(loss_G.item(), len(img_lr))
            epoch_train_psnr.update(train_pnsr.item(), len(img_lr))

            ##########################
            # training discriminator #
            ##########################
            loss_D = trainer.train_discriminator(optimizer_D, discriminator, img_gen, img_hr, real_labels, fake_labels)

            epoch_train_dis.update(loss_D.item(), len(img_lr))

        train_gen_losses.append(epoch_train_gen.avg)
        train_psnr_all.append(epoch_train_psnr.avg)
        train_disc_losses.append(epoch_train_dis.avg)

        # Testing
        with torch.no_grad():
            generator.eval()
            discriminator.eval()
            epoch_val_gen = AverageMeter()
            epoch_val_dis = AverageMeter()
            epoch_val_psnr = AverageMeter()

            for batch_idx, images_hl in tqdm(enumerate(val_dataloader), desc=f'Validate Epoch {epoch}',
                                             total=int(len(val_dataloader))):
                # Configure model input
                img_lr = images_hl["test"].to(device)
                img_hr = images_hl["ref"].to(device)

                # Adversarial ground truths
                real_labels = torch.ones((img_hr.size(0), *discriminator.output_shape)).to(device)
                fake_labels = torch.zeros((img_hr.size(0), *discriminator.output_shape)).to(device)

                # Eval Generator :Generate a high resolution image from low resolution input
                img_gen = generator(img_lr)

                loss_G = trainer.get_generate_loss(discriminator, img_gen, img_hr, real_labels)

                val_psnr = calc_psnr(img_gen.detach(), img_hr)

                loss_D = trainer.get_disc_loss(discriminator, img_gen, img_hr, real_labels, fake_labels)

                epoch_val_gen.update(loss_G.item(), len(img_lr))
                epoch_val_psnr.update(val_psnr.item(), len(img_lr))
                epoch_val_dis.update(loss_D.item(), len(img_lr))

                if epoch % opt.save_epoch_rate == 0 or (epoch == (trained_epoch + n_epochs)):
                    if batch_idx == 0:
                        save_img_lr = nn.functional.interpolate(img_lr, scale_factor=opt.scale_factor)
                        save_img_lr = make_grid(save_img_lr, nrow=1, normalize=True)
                        save_img_hr = make_grid(img_hr, nrow=1, normalize=True)
                        save_gen_hr = make_grid(img_gen, nrow=1, normalize=True)
                        img_grid = torch.cat((save_img_hr, save_img_lr, save_gen_hr), -1)
                        save_image(img_grid, os.path.join(save_folder_image, f"epoch_{epoch}.png"), normalize=False)
                        # image_show(img_grid)

                        # Save model checkpoints
                        save_model(os.path.join(save_folder_model, f"epoch_{epoch}_generator.pth"),
                                   generator, optimizer_G, epoch)
                        save_model(os.path.join(save_folder_model, f"epoch_{epoch}_discriminator.pth"),
                                   discriminator, optimizer_D, epoch)

            val_gen_losses.append(epoch_val_gen.avg)
            val_disc_losses.append(epoch_val_dis.avg)
            val_psnr_all.append(epoch_val_psnr.avg)

    plt.figure(figsize=(10, 7))
    plt.plot(train_gen_losses, color='blue', label='train gen losses')
    plt.plot(val_gen_losses, color='red', label='Validate gen losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, 'gen_loss{epoch}.png'))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_disc_losses, color='blue', label='train disc losses')
    plt.plot(val_disc_losses, color='red', label='Validate disc losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, 'disc_loss{epoch}.png'))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr_all, color='blue', label='train psnr')
    plt.plot(val_psnr_all, color='red', label='Validate psnr')
    plt.xlabel('Epochs')
    plt.ylabel('DB ')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, 'psnr{epoch}.png'))
    plt.show()


def run(opt):
    save_folder_image = os.path.join(opt.save_folder, r"RealESRGAN/results")
    os.makedirs(save_folder_image, exist_ok=True)

    dataset = ImageDatasetSingleToPair(opt.data_folder, img_H=opt.hr_height, img_W=opt.hr_width, scale_factor=4)
    result_dataloader = DataLoader(dataset=dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    img_shape = tuple(dataset[0]['ref'].shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorRRDB(in_channels=img_shape[0], filters=32, scale_factor=opt.scale_factor).to(device)
    load_model(opt.load_models_path_gen, generator)

    generator.eval()
    for batch_idx, images_hl in tqdm(enumerate(result_dataloader), total=int(len(result_dataloader))):
        # Configure model input
        img_lr = images_hl["test"].to(device)
        img_hr = images_hl["ref"].to(device)
        gen_hr = generator(img_lr)

        img_lr = nn.functional.interpolate(img_lr, scale_factor=opt.scale_factor)
        img_lr = make_grid(img_lr, nrow=1, normalize=True)
        img_hr = make_grid(img_hr, nrow=1, normalize=True)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)

        img_grid = torch.cat((img_hr, img_lr, gen_hr), -1)
        save_image(img_grid, os.path.join(save_folder_image, f"picture_{batch_idx}.png"), normalize=False)


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--data_folder', type=str, default='../data/coco_sub/', help='dataset path')
    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--hr_height', type=int, default=256, help='High resolution image height')
    parser.add_argument('--hr_width', type=int, default=256, help='High resolution image width')
    parser.add_argument('--scale_factor', type=int, default=4, help='Image super-resolution coefficient')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--warm_epochs', type=int, default=0, help='the Pre-training epochs')
    parser.add_argument('--save_epoch_rate', type=int, default=100, help='How many epochs save once')
    parser.add_argument("--save_image_count", type=int, default=5, help="保存图像个数")
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_path_gen', type=str, default=r"./working/RealESRGAN/models/discriminator.pth",
                        help='load model path')
    parser.add_argument('--load_models_path_dis', type=str, default=r"./working/RealESRGAN/models/generator.pth",
                        help='load model path')

    args = parser.parse_args(args=[])  # 不添加args=[] kaggle会报错
    return args


if __name__ == '__main__':

    para = parse_args()
    # para.data_folder = '../data/DIV2K_train_LR_x8'
    para.data_folder = '../data/T91'
    para.save_folder = r"./working/"
    para.hr_height = 128
    para.hr_width = 128
    para.scale_factor = 4
    para.batch_size = 8

    is_train = True
    if is_train:
        para.epochs = 20
        para.warm_epochs = 10
        para.save_epoch_rate = 10
        para.load_models = False
        para.load_models_path_gen = r"./working/RealESRGAN/models/epoch_6_generator.pth"
        para.load_models_path_dis = r"./working/RealESRGAN/models/epoch_6_discriminator.pth"

        train(para)

    else:
        para.load_models = True
        para.load_models_path_gen = r"./working/RealESRGAN/models/epoch_6_generator.pth"
        para.load_models_path_dis = r"./working/RealESRGAN/models/epoch_6_discriminator.pth"

        run(para)
