"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.07.21
    Description	:
            基于扩散模型的SAR图像上色算法
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt
from utils.data_read import ImageDatasetPair

from utils.utils import load_model, save_model, AverageMeter, image_show
from utils.common import EMA
from models.DDPM_models import GaussianDiffusion, LossType
from backbone.unet import UNet


def train(opt):
    save_folder_image = os.path.join(opt.save_folder, r"ColorSARDiff/images")
    save_folder_model = os.path.join(opt.save_folder, r"ColorSARDiff/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    dataset_sar = os.path.join(opt.data_folder, r"gray")
    dataset_optical = os.path.join(opt.data_folder, r"color")

    dataset = ImageDatasetPair(dataset_optical, dataset_sar, is_Normalize=True)

    # img_shape = tuple(dataset[0]['def'].shape)

    data_len = dataset.__len__()
    val_data_len = opt.batch_size * 1
    # train_set, val_set = torch.utils.data.random_split(dataset, [data_len - val_data_len, val_data_len])
    val_set = Subset(dataset, list(range(val_data_len)))
    train_set = Subset(dataset, list(range(val_data_len, data_len)))
    val_dataloader = DataLoader(dataset=val_set, num_workers=0, batch_size=opt.batch_size, shuffle=False)
    train_dataloader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(img_channels=opt.img_channels, time_emb_dim=opt.timesteps, is_cond_image=True,
                 channel_mults=(1, 2, 4, 4), attention_layers=(1,))

    loss_type = LossType()
    loss_type.L1 = True
    loss_type.L2 = True
    loss_type.SSIM = True
    diffusion = GaussianDiffusion(model, opt.img_channels, (opt.img_h, opt.img_w), timesteps=opt.timesteps,
                                  loss_type=loss_type).to(device)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=opt.lr)

    # Load pretrained models
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_checkpoint, diffusion, optimizer)

    ema = EMA(model, device)

    n_epochs = opt.epochs

    train_loss1, train_loss2, train_loss_ssim,train_loss_P = [], [], [], []
    # for epoch in range(trained_epoch + 1, trained_epoch + n_epochs + 1):
    for epoch in tqdm(range(trained_epoch + 1, trained_epoch + n_epochs + 1), desc=f'Training Epoch', total=n_epochs):
        # Training
        diffusion.train()
        epoch_train_loss1 = AverageMeter()
        epoch_train_loss2 = AverageMeter()
        epoch_train_loss_ssim = AverageMeter()

        # for batch_idx, imgs in tqdm(enumerate(train_dataloader), desc=f'Training Epoch {epoch}',
        #                             total=int(len(train_dataloader))):

        for batch_idx, imgs in enumerate(train_dataloader):
            images_optical = imgs["def"].to(device)
            images_sar = imgs["test"].to(device)

            optimizer.zero_grad()

            loss1, loss2, loss_ssim = diffusion(x=images_optical, cond_image=images_sar)
            loss = loss1 + loss2 + (1 - loss_ssim)
            epoch_train_loss1.update(loss1.item(), len(images_sar))
            epoch_train_loss2.update(loss2.item(), len(images_sar))
            epoch_train_loss_ssim.update(loss_ssim.item(), len(images_sar))

            loss.backward()
            optimizer.step()

            # acc_train_loss += loss.item()
            # diffusion.update_ema()
            ema.update_ema(diffusion.model)
        train_loss1.append(epoch_train_loss1.avg)
        train_loss2.append(epoch_train_loss2.avg)
        train_loss_ssim.append(epoch_train_loss_ssim.avg)

        # Save models and images
        if epoch % opt.save_epoch_rate == 0 or (epoch == (trained_epoch + n_epochs)):
            diffusion.eval()

            for batch_idx, imgs in enumerate(val_dataloader):
                images_color = imgs["def"].to(device)
                images_gray = imgs["test"].to(device)

                samples = diffusion.sample(batch_size=opt.batch_size, device=device, cond_image=images_gray).to(device)
                if batch_idx == 0:
                    save_image_count = opt.save_image_count if opt.batch_size > opt.save_image_count else opt.batch_size
                    img_gray = make_grid(images_gray[0:save_image_count, :, :, :], nrow=1, normalize=True).to(device)
                    img_color = make_grid(images_color[0:save_image_count, :, :, :], nrow=1, normalize=True).to(device)
                    gen_color = make_grid(samples[0:save_image_count, :, :, :], nrow=1, normalize=True).to(device)

                    img_grid = torch.cat((img_gray, img_color, gen_color), -1)
                    save_image(img_grid, os.path.join(save_folder_image, f"epoch_{epoch}.png"), normalize=False)

                    save_model(os.path.join(save_folder_model, f"epoch_{epoch}_models.pth"), diffusion, optimizer,
                               epoch)
                    image_show(img_grid)

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss1, color='green', label='train loss1')
    plt.plot(train_loss2, color='red', label='train loss2')
    plt.plot(train_loss_ssim, color='blue', label='train loss ssim')
    plt.plot(train_loss_P, color='yellow', label='train loss Perceptual')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, f"loss{trained_epoch + n_epochs}.png"))
    plt.show()


def run(opt):
    save_folder_image = os.path.join(opt.save_folder, r"ColorSARDiff/results")
    os.makedirs(save_folder_image, exist_ok=True)

    dataset_sar = os.path.join(opt.data_folder, r"gray")
    dataset_optical = os.path.join(opt.data_folder, r"color")
    dataset = ImageDatasetPair(dataset_optical, dataset_sar, is_Normalize=True)
    result_dataloader = DataLoader(dataset=dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加
    model = UNet(img_channels=opt.img_channels, is_cond_image=True)

    diffusion = GaussianDiffusion(model, opt.img_channels, (opt.img_h, opt.img_w), timesteps=opt.timesteps,
                                  loss_type="l2").to(device)
    load_model(opt.load_models_checkpoint, diffusion)
    # Initialize generator and discriminator

    for batch_idx, images_hl in tqdm(enumerate(result_dataloader), total=int(len(result_dataloader))):
        # Configure model input
        img_sar = images_hl["test"].to(device)
        img_optical = images_hl["def"].to(device)

        samples = diffusion.sample(batch_size=opt.batch_size, device=device, cond_image=img_sar)

        images_gen = samples + img_sar
        img_sar = make_grid(img_sar, nrow=1, normalize=True).to(device)
        img_optical = make_grid(img_optical, nrow=1, normalize=True).to(device)
        gen_color = make_grid(images_gen, nrow=1, normalize=True).to(device)

        img_grid = torch.cat((img_sar, img_optical, gen_color), -1)

        save_image(img_grid, os.path.join(save_folder_image, f"picture_{batch_idx}.png"), normalize=False)


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--data_folder', type=str, default='data/coco_sub', help='dataset path')
    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--img_channels', type=int, default=3, help='the channel of the image')
    parser.add_argument('--img_w', type=int, default=64, help='image width')
    parser.add_argument('--img_h', type=int, default=64, help='image height')
    parser.add_argument('--batch_size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--save_image_count", type=int, default=5, help="保存图像个数")
    parser.add_argument("--timesteps", type=int, default=1000, help="迭代次数")
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--save_epoch_rate', type=int, default=100, help='How many epochs save once')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_checkpoint', type=str, default=r"./working/SRDiff/models/checkpoint.pth",
                        help='load model path')

    args = parser.parse_args(args=[])  # 不添加args=[] kaggle会报错
    return args


if __name__ == '__main__':

    para = parse_args()
    #para.data_folder = '../data/SAR128'
    para.data_folder = '../data/face'
    para.save_folder = r"./working/"
    para.img_channels = 3
    para.img_w = 24
    para.img_h = 32
    para.batch_size = 1
    para.timesteps = 200

    is_train = True
    if is_train:
        para.epochs = 500
        para.save_epoch_rate = 100
        para.load_models = False
        para.load_models_checkpoint = r"./working/ColorSARDiff/models/epoch_1000_models.pth"
        train(para)
    else:
        para.load_models = True
        para.load_models_checkpoint = r"./working/ColorSARDiff/models/epoch_10_models.pth"
        run(para)
