"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.3.3
    Description	:
            FSRCNN 模型训练
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
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image

from utils.data_read import ImageDatasetCrop
from FSRCNN_models import FSRCNN
from utils.utils import save_model, load_model, calc_psnr, AverageMeter


def train(opt):
    # 创建文件夹
    save_folder_image = os.path.join(opt.save_folder, r"FSRCNN/images")
    save_folder_model = os.path.join(opt.save_folder, r"FSRCNN/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    # 读取数据
    dataset = ImageDatasetCrop(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w, is_same_shape=False,
                               scale_factor=opt.scale_factor)

    data_len = dataset.__len__()
    val_data_len = opt.batch_size * 2

    train_set, val_set = torch.utils.data.random_split(dataset, [data_len - val_data_len, val_data_len])
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # 建立模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fsrcnn = FSRCNN(scale_factor=opt.scale_factor).to(device)

    optimizer = optim.Adam(fsrcnn.parameters())
    criterion = nn.MSELoss().to(device)

    # 已经训练的 opoch数量
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_path, fsrcnn, optimizer)

    #  per_image_mse_loss = F.mse_loss(HR, newHR, reduction='none')
    train_loss_all, val_loss_all = [], []
    train_psnr_all, val_psnr_all = [], []

    # 读取显示图像
    show_data1 = dataset[0]
    show_data2 = dataset[1]
    show_data3 = dataset[2]
    show_data4 = dataset[3]
    show_image_hr = torch.stack([show_data1["hr"], show_data2["hr"], show_data3["hr"], show_data4["hr"]], 0).to(device)
    show_image_lr = torch.stack([show_data1["lr"], show_data2["lr"], show_data3["lr"], show_data4["lr"]], 0).to(device)
    # 强制保存最后一个epoch
    n_epochs = opt.epochs
    save_epoch = opt.save_epoch.union({n_epochs + trained_epoch})

    for epoch in tqdm(range(trained_epoch, trained_epoch + n_epochs), desc=f'epoch'):
        fsrcnn.train()
        epoch_train_loss = AverageMeter()
        epoch_train_psnr = AverageMeter()
        for images_hl in train_loader:
            img_lr = images_hl["lr"].to(device)
            img_hr = images_hl["hr"].to(device)

            sr_img = fsrcnn(img_lr)

            optimizer.zero_grad()
            train_loss = criterion(img_hr, sr_img)
            train_loss.backward(retain_graph=True)
            optimizer.step()
            train_psnr = calc_psnr(sr_img.detach(), img_hr)

            epoch_train_loss.update(train_loss.item(), len(img_lr))
            epoch_train_psnr.update(train_psnr.item(), len(img_lr))

        train_loss_all.append(epoch_train_loss.avg)
        train_psnr_all.append(epoch_train_psnr.avg)

        fsrcnn.eval()
        epoch_val_loss = AverageMeter()
        epoch_val_psnr = AverageMeter()
        with torch.no_grad():
            for idx, datas_hl in enumerate(val_loader):
                image_l = datas_hl["lr"].to(device)
                image_h = datas_hl["hr"].to(device)

                sr_img = fsrcnn(image_l)

                val_loss = criterion(sr_img, image_h)
                epoch_val_loss.update(val_loss.item(), len(sr_img))
                val_psnr = calc_psnr(sr_img, image_h)
                epoch_val_psnr.update(val_psnr.item(), len(sr_img))

        val_loss_all.append(epoch_val_loss.avg)
        val_psnr_all.append(epoch_val_psnr.avg)

        # save the last epoch
        if epoch + 1 in save_epoch:
            fsrcnn.eval()
            show_image_sr = fsrcnn(show_image_lr)

            img_lr = nn.functional.interpolate(show_image_lr, scale_factor=opt.scale_factor)
            img_lr = make_grid(img_lr, nrow=1, normalize=True)
            img_hr = make_grid(show_image_hr, nrow=1, normalize=True)
            gen_hr = make_grid(show_image_sr, nrow=1, normalize=True)

            img_grid = torch.cat((img_hr, img_lr, gen_hr), -1)
            save_image(img_grid, os.path.join(save_folder_image, f"epoch_{epoch + 1}.png"), normalize=False)

            # 保存最新的参数和损失最小的参数
            save_model(os.path.join(save_folder_model, f"epoch_{epoch + 1}_model.pth"), fsrcnn,
                       optimizer, epoch + 1)

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_all, color='green', label='train loss')
    plt.plot(val_loss_all, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, r"loss.png"))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr_all, color='green', label='train PSNR dB')
    plt.plot(val_psnr_all, color='red', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, r"psnr.png"))
    plt.show()


def run(opt):

    save_folder_result = os.path.join(opt.save_folder, r"FSRCNN/results")
    os.makedirs(save_folder_result, exist_ok=True)

    dataset = ImageDatasetCrop(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w, is_same_shape=False,
                               scale_factor=opt.scale_factor, max_count=16)
    result_loader = DataLoader(dataset=dataset, num_workers=0, batch_size=opt.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fsrcnn = FSRCNN().to(device)

    load_model(opt.load_models_path, fsrcnn)
    fsrcnn.eval()
    for idx, datas_hl in tqdm(enumerate(result_loader), total=int(len(result_loader))):
        image_l = datas_hl["lr"].to(device)
        image_h = datas_hl["hr"].to(device)
        image_gen = fsrcnn(image_l)

        image_l = nn.functional.interpolate(image_l, scale_factor=opt.scale_factor)

        imgs_lr = make_grid(image_l, nrow=1, normalize=True)
        imgs_hr = make_grid(image_h, nrow=1, normalize=True)
        gen_hr = make_grid(image_gen, nrow=1, normalize=True)

        img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
        save_image(img_grid, os.path.join(save_folder_result, f'picture_{idx}_Image.png'), normalize=False)


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--data_folder', type=str, default='../data/T91', help='dataset path')
    parser.add_argument('--img_w', type=int, default=160, help='randomly cropped image width')
    parser.add_argument('--img_h', type=int, default=160, help='randomly cropped image height')
    parser.add_argument('--scale_factor', type=int, default=4, help='Image super-resolution coefficient')
    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_path', type=str, default=r"./working/SRCNN/models/last_ckpt.pth",
                        help='load model path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--save_epoch', type=set, default=set(), help='number of saved epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')

    args = parser.parse_args(args=[])

    return args


if __name__ == '__main__':

    is_train = True

    if is_train:
        para = parse_args()
        para.folder_data = '../data/DIV2K_train_LR_x8'
        para.save_folder = r"./working/"
        para.img_w = 160
        para.img_h = 160
        para.scale_factor = 4
        para.epochs = 1200
        # para.save_epoch = set(range(1, 100, 20))
        para.load_models = True
        para.load_models_path = r"./working/FSRCNN/models/epoch_800_model.pth"

        train(para)

    else:
        para = parse_args()
        para.folder_data = '../data/T91'
        para.save_folder = r"./working/"
        para.img_w = 160
        para.img_h = 160
        para.scale_factor = 4
        para.batch_size =16
        para.load_models_path = r"./working/FSRCNN/models/epoch_400_model.pth"

        run(para)
