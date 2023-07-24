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
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_read import ImageDatasetCrop, ImageDatasetPair
from SRGAN_models import GeneratorResNet, DiscriminatorNet, TrainerSRGAN
from utils.utils import save_model, load_model, calc_psnr, AverageMeter


def train(opt):
    save_folder_image = os.path.join(opt.save_folder, r"SRGAN/images")
    save_folder_model = os.path.join(opt.save_folder, r"SRGAN/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    # 读取数据
    # dataset = ImageDatasetCrop(opt.data_folder, opt.hr_height, opt.hr_width, is_Normalize=True, scale_factor=4)

    dataset_high = os.path.join(opt.data_folder, r"high")
    dataset_low = os.path.join(opt.data_folder, r"low")
    dataset = ImageDatasetPair(dataset_high, dataset_low, is_Normalize=True)

    img_shape = tuple(dataset[0]['hr'].shape)

    data_len = dataset.__len__()
    val_data_len = opt.batch_size * 3
    train_set, val_set = torch.utils.data.random_split(dataset, [data_len - val_data_len, val_data_len])
    test_dataloader = DataLoader(dataset=val_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    train_dataloader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = GeneratorResNet(in_channels=img_shape[0], scale_factor=opt.scale_factor).to(device)
    discriminator = DiscriminatorNet(input_shape=img_shape).to(device)
    # define the trainer
    trainer = TrainerSRGAN(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))  # lr = 0.00008
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Load pretrained models
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_path_gen, generator, optimizer_G)
        load_model(opt.load_models_path_dis, discriminator, optimizer_D)

    # 读取用去显示图像保存
    show_data1 = dataset[0]
    show_data2 = dataset[1]
    show_data3 = dataset[2]
    show_data4 = dataset[3]
    show_image_hr = torch.stack([show_data1["hr"], show_data2["hr"], show_data3["hr"], show_data4["hr"]], 0).to(device)
    show_image_lr = torch.stack([show_data1["lr"], show_data2["lr"], show_data3["lr"], show_data4["lr"]], 0).to(device)

    train_gen_losses, train_disc_losses, train_psnr_all = [], [], []
    val_gen_losses, val_disc_losses, val_psnr_all = [], [], []

    n_epochs = opt.epochs
    save_epoch = opt.save_epoch.union({n_epochs + trained_epoch})

    for epoch in range(trained_epoch, n_epochs + trained_epoch):
        # Training
        generator.train()
        discriminator.train()

        epoch_train_gen = AverageMeter()
        epoch_train_dis = AverageMeter()
        epoch_train_psnr = AverageMeter()

        for batch_idx, images_hl in tqdm(enumerate(train_dataloader), desc=f'Training Epoch {epoch}',
                                         total=int(len(train_dataloader))):
            images_l = images_hl["lr"].to(device)
            images_h = images_hl["hr"].to(device)

            # Adversarial ground truths
            real_labels = torch.ones((images_l.size(0), *discriminator.output_shape)).to(device)
            fake_labels = torch.zeros((images_l.size(0), *discriminator.output_shape)).to(device)

            # Generate a high resolution image from low resolution input
            gen_hr = generator(images_l)

            if epoch < opt.warm_epochs + trained_epoch:
                trainer.pre_train_generator(optimizer_G, gen_hr, images_h)
                continue

            # ------------------
            #  Train Generators
            # ------------------
            loss_G = trainer.train_generator(optimizer_G, discriminator, gen_hr, images_h, real_labels)

            train_pnsr = calc_psnr(gen_hr.detach(), images_h)

            epoch_train_gen.update(loss_G.item(), len(images_h))
            epoch_train_psnr.update(train_pnsr.item(), len(images_h))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            loss_D = trainer.train_discriminator(optimizer_D, discriminator, gen_hr, images_h, real_labels, fake_labels)

            epoch_train_dis.update(loss_D.item(), len(images_l))

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

            for batch_idx, images_hl in tqdm(enumerate(test_dataloader), desc=f'Validate Epoch {epoch}',
                                             total=int(len(test_dataloader))):
                # Configure model input
                img_lr = images_hl["lr"].to(device)
                img_hr = images_hl["hr"].to(device)

                # Adversarial ground truths
                valid = torch.ones((img_lr.size(0), *discriminator.output_shape)).to(device)
                fake = torch.zeros((img_lr.size(0), *discriminator.output_shape)).to(device)

                # Eval Generator
                gen_hr = generator(img_lr)
                loss_G = trainer.get_generate_loss(discriminator, gen_hr, img_hr, valid)

                loss_D = trainer.get_disc_loss(discriminator, gen_hr, img_hr, valid, fake)

                val_psnr = calc_psnr(gen_hr.detach(), img_hr)

                epoch_val_gen.update(loss_G.item(), len(img_lr))
                epoch_val_psnr.update(val_psnr.item(), len(img_lr))
                epoch_val_dis.update(loss_D.item(), len(img_lr))

            val_gen_losses.append(epoch_val_gen.avg)
            val_disc_losses.append(epoch_val_dis.avg)
            val_psnr_all.append(epoch_val_psnr.avg)

        # Save image grid with up_sampling inputs and SRGAN outputs
        if epoch + 1 in save_epoch:
            current_epoch = epoch + 1

            generator.eval()
            gen_hr = generator(show_image_lr)
            img_lr = nn.functional.interpolate(show_image_lr, scale_factor=opt.scale_factor)
            img_lr = make_grid(img_lr, nrow=1, normalize=True)
            img_hr = make_grid(show_image_hr, nrow=1, normalize=True)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)

            img_grid = torch.cat((img_hr, img_lr, gen_hr), -1)
            save_image(img_grid, os.path.join(save_folder_image, f"epoch_{current_epoch}.png"), normalize=False)

            # image_show(img_grid.cpu())

            # Save model checkpoints
            save_model(os.path.join(save_folder_model, f"epoch_{current_epoch}_generator.pth"),
                       generator, optimizer_G, current_epoch)
            save_model(os.path.join(save_folder_model, f"epoch_{current_epoch}_discriminator.pth"),
                       discriminator, optimizer_D, current_epoch)

    plt.figure(figsize=(10, 7))
    plt.plot(train_gen_losses, color='blue', label='train gen losses')
    plt.plot(val_gen_losses, color='red', label='Validate gen losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, 'gen_loss.png'))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_disc_losses, color='blue', label='train disc losses')
    plt.plot(val_disc_losses, color='red', label='Validate disc losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, 'disc_loss.png'))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr_all, color='blue', label='train psnr')
    plt.plot(val_psnr_all, color='red', label='Validate psnr')
    plt.xlabel('Epochs')
    plt.ylabel('dB ')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, 'psnr.png'))
    plt.show()


def run(opt):
    save_folder_image = os.path.join(opt.save_folder, r"SRGAN/results")
    os.makedirs(save_folder_image, exist_ok=True)

    # dataset = ImageDatasetCrop(opt.data_folder, opt.hr_height, opt.hr_width, scale_factor=4)

    dataset_high = os.path.join(opt.data_folder, r"high")
    dataset_low = os.path.join(opt.data_folder, r"low")
    dataset = ImageDatasetPair(dataset_high, dataset_low, is_Normalize=True)

    result_dataloader = DataLoader(dataset=dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorResNet(in_channels=dataset[0]['def'].shape[0], scale_factor=opt.scale_factor).to(device)
    load_model(opt.load_models_path_gen, generator)

    generator.eval()
    for batch_idx, images_hl in tqdm(enumerate(result_dataloader), total=int(len(result_dataloader))):
        # Configure model input
        img_lr = images_hl["test"].to(device)
        img_hr = images_hl["def"].to(device)
        gen_hr = generator(img_lr)

        img_lr = nn.functional.interpolate(img_lr, scale_factor=opt.scale_factor)
        img_lr = make_grid(img_lr, nrow=1, normalize=True)
        img_hr = make_grid(img_hr, nrow=1, normalize=True)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)

        img_grid = torch.cat((img_hr, img_lr, gen_hr), -1)
        save_image(img_grid, os.path.join(save_folder_image, f"picture_{batch_idx}.png"), normalize=False)


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--data_folder', type=str, default='data/coco_sub', help='dataset path')
    parser.add_argument('--hr_height', type=int, default=256, help='High resolution image height')
    parser.add_argument('--hr_width', type=int, default=256, help='High resolution image width')

    parser.add_argument('--scale_factor', type=int, default=4, help='Image super-resolution coefficient')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')

    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_path_gen', type=str, default=r"./working/SRGAN/models/discriminator.pth",
                        help='load model path')
    parser.add_argument('--load_models_path_dis', type=str, default=r"./working/SRGAN/models/generator.pth",
                        help='load model path')

    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--save_epoch', type=set, default=set(), help='number of saved epochs')

    args = parser.parse_args(args=[])  # 不添加args=[] kaggle会报错
    return args


if __name__ == '__main__':

    is_train = False

    if is_train:
        para = parse_args()
        # para.data_folder = '../data/T91'
        # para.data_folder = '../data/DIV2K_train_LR_x8'
        para.data_folder = '../data/SAR'
        para.save_folder = r"./working/"
        para.hr_height = 256
        para.hr_width = 256
        para.scale_factor = 2
        para.pre_train_epochs = 20
        para.epochs = 10
        para.batch_size = 10
        # para.save_epoch = set(range(1, 100, 20))
        para.load_models = True
        para.load_models_path_gen = r"./working/SRGAN/models/epoch_6000_generator.pth"
        para.load_models_path_dis = r"./working/SRGAN/models/epoch_6000_discriminator.pth"

        train(para)

    else:
        para = parse_args()
        # para.data_folder = '../data/T91'

        para.data_folder = '../data/SAR_2'
        para.save_folder = r"./working/"
        para.hr_height = 256
        para.hr_width = 256
        para.scale_factor = 2
        para.batch_size = 10
        para.load_models_path_gen = r"./working/SRGAN/models/epoch_6000_generator.pth"
        para.load_models_path_dis = r"./working/SRGAN/models/epoch_6000_discriminator.pth"

        run(para)
