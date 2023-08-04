"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.03.19
    Description	:
            DDPM  模型训练
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
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from utils.data_read import ImageDatasetSingle
from utils.utils import load_model, save_model, AverageMeter, image_show
from utils.common import EMA
from models.DDPM_models import GaussianDiffusion, LossType
from backbone.unet import UNet, UNetConfig


def train(opt):
    save_folder_image = os.path.join(opt.save_folder, r"DDPM/images")
    save_folder_model = os.path.join(opt.save_folder, r"DDPM/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    dataset_train = ImageDatasetSingle(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w, max_count=160)
    dataloader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # img_shape = (opt.img_channels, opt.img_h, opt.img_w)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(img_channels=opt.img_channels, time_emb_dim=opt.timesteps, channel_mults=(1, 2, 4, 4),  attention_layers=(1,) )

    ema = EMA(model, device)

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

    n_epochs = opt.epochs

    train_loss1, train_loss2, train_loss_ssim = [], [], []
    for epoch in tqdm(range(trained_epoch + 1, trained_epoch + n_epochs + 1), desc=f'Training Epoch',total=n_epochs):
        # Training
        diffusion.train()
        epoch_train_loss1 = AverageMeter()
        epoch_train_loss2 = AverageMeter()
        epoch_train_loss_ssim = AverageMeter()
        # for batch_idx, imgs in tqdm(enumerate(dataloader_train), desc=f'Training Epoch {epoch}',
        #                             total=int(len(dataloader_train))):
        for batch_idx, imgs in enumerate(dataloader_train):
            x = imgs.to(device)

            optimizer.zero_grad()
            loss1, loss2, loss_ssim = diffusion(x)

            loss = loss1 + loss2 + (1 - loss_ssim)
            epoch_train_loss1.update(loss1.item(), len(x))
            epoch_train_loss2.update(loss2.item(), len(x))
            epoch_train_loss_ssim.update(loss_ssim.item(), len(x))

            loss.backward()
            optimizer.step()
            # acc_train_loss += loss.item()
            # diffusion.update_ema()
            ema.update_ema(diffusion.model)

        train_loss1.append(epoch_train_loss1.avg)
        train_loss2.append(epoch_train_loss2.avg)
        train_loss_ssim.append(epoch_train_loss_ssim.avg)
        # Save models and images
        if (epoch % opt.save_epoch_rate) == 0 or (epoch == (trained_epoch + n_epochs)):
            diffusion.eval()
            samples = diffusion.sample(batch_size=opt.batch_size, device=device)
            save_image(samples.data[:opt.batch_size],
                       os.path.join(save_folder_image, f"epoch_{epoch}_result.png"), nrow=10, normalize=False)
            save_model(os.path.join(save_folder_model, f"epoch_{epoch}_models.pth"), diffusion, optimizer, epoch)

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss1, color='green', label='train loss1')
    plt.plot(train_loss2, color='red', label='train loss2')
    plt.plot(train_loss_ssim, color='blue', label='train loss ssim')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, f"loss{trained_epoch + n_epochs}.png"))
    plt.show()


def run(opt):
    save_folder_image = os.path.join(opt.save_folder, r"DDPM/results")
    os.makedirs(save_folder_image, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 添加
    model = UNet(img_channels=opt.img_channels, time_emb_dim=opt.timesteps, channel_mults=(1, 2, 4, 4),attention_layers=(1,))

    diffusion = GaussianDiffusion(model, opt.img_channels, (opt.img_h, opt.img_w), timesteps=opt.timesteps).to(device)
    load_model(opt.load_models_checkpoint, diffusion)

    samples = diffusion.sample(batch_size=opt.batch_size, device=device)
    save_image(samples.data[:opt.batch_size], os.path.join(save_folder_image, f"result.png"), nrow=10, normalize=False)


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
    parser.add_argument('--load_models_checkpoint', type=str, default=r"./working/SRGAN/models/discriminator.pth",
                        help='load model path')

    args = parser.parse_args(args=[])  # 不添加args=[] kaggle会报错
    return args


if __name__ == '__main__':

    para = parse_args()
    para.save_folder = r"./working/"
    para.data_folder = r'../data/flag128'
    # para.data_folder = '../data/SAR128/optical'
    para.timesteps = 100
    para.img_channels = 3
    para.img_w = 128
    para.img_h = 128
    para.batch_size = 1

    is_train = True

    if is_train:
        para.epochs = 10
        para.save_epoch_rate = 2
        para.load_models = False
        para.load_models_checkpoint = r"./working/DDPM/models/epoch_1000_models.pth"
        train(para)
    else:
        para.load_models = True
        para.load_models_checkpoint = r"./working/DDPM/models/epoch_50_models.pth"
        run(para)
