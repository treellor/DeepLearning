"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2022.12.19
    Description	:
            SRCNN 模型训练
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
import torch.optim as optim

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils.data_read import ImageDatasetSingleToPair, OptType
from SRCNN_models import SRCNN
from utils.utils import save_model, load_model, calc_psnr, AverageMeter, image_show


def train(opt):
    # 创建文件夹
    save_folder_image = os.path.join(opt.save_folder, r"SRCNN/images")
    save_folder_model = os.path.join(opt.save_folder, r"SRCNN/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    # 读取数据
    dataset = ImageDatasetSingleToPair(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w, is_same_size=True,
                                       optType=OptType.RESIZE, scale_factor=opt.scale_factor)
    data_len = dataset.__len__()
    val_data_len = opt.batch_size * 2
    train_set, val_set = torch.utils.data.random_split(dataset, [data_len - val_data_len, val_data_len])
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # 建立模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srcnn = SRCNN().to(device)
    optimizer = optim.Adam(srcnn.parameters())
    criterion = nn.MSELoss().to(device)

    # 已经训练的 epoch数量
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_path, srcnn, optimizer)

    # 读取显示图像
    show_image_hr = torch.stack([dataset[i]["ref"] for i in range(0, opt.batch_size)], 0).to(device)
    show_image_lr = torch.stack([dataset[i]["test"] for i in range(0, opt.batch_size)], 0).to(device)

    # 强制保存最后一个epoch
    n_epochs = opt.epochs
    # 评估参数
    train_loss_all, val_loss_all = [], []
    train_psnr_all, val_psnr_all = [], []

    for epoch in tqdm(range(trained_epoch + 1, trained_epoch + n_epochs + 1), desc=f'epoch'):
        epoch_train_loss = AverageMeter()
        epoch_train_psnr = AverageMeter()
        srcnn.train()
        for images_hl in train_loader:
            img_lr = images_hl["test"].to(device)
            img_hr = images_hl["ref"].to(device)

            img_gen = srcnn(img_lr)

            # srcnn.zero_grad()
            optimizer.zero_grad()
            train_loss = criterion(img_gen, img_hr)
            train_loss.backward(retain_graph=True)
            optimizer.step()

            train_psnr = calc_psnr(img_gen.detach(), img_hr)

            epoch_train_loss.update(train_loss.item(), len(img_hr))
            epoch_train_psnr.update(train_psnr.item(), len(img_hr))

        train_loss_all.append(epoch_train_loss.avg)
        train_psnr_all.append(epoch_train_psnr.avg)

        srcnn.eval()
        epoch_val_loss = AverageMeter()
        epoch_val_psnr = AverageMeter()
        with torch.no_grad():
            for idx, datas_hl in enumerate(val_loader):
                image_l = datas_hl["test"].to(device)
                image_h = datas_hl["ref"].to(device)

                image_gen = srcnn(image_l)

                val_loss = criterion(image_gen, image_h)
                epoch_val_loss.update(val_loss.item(), len(image_l))
                val_psnr = calc_psnr(image_gen, image_h)
                epoch_val_psnr.update(val_psnr.item(), len(image_l))

        val_loss_all.append(epoch_val_loss.avg)
        val_psnr_all.append(epoch_val_psnr.avg)

        # save the result
        if epoch % opt.save_epoch_rate == 0 or (epoch == (trained_epoch + n_epochs)):
            srcnn.eval()
            show_image_gen = srcnn(show_image_lr)
            gen_hr = make_grid(show_image_gen, nrow=1, normalize=True)
            img_lr = make_grid(show_image_lr, nrow=1, normalize=True)
            img_hr = make_grid(show_image_hr, nrow=1, normalize=True)
            img_grid = torch.cat((img_hr, img_lr, gen_hr), -1)
            save_image(img_grid, os.path.join(save_folder_image, f"epoch_{epoch}.png"), normalize=False)

            # image_show(img_grid)
            # 保存最新的参数和损失最小的参数
            save_model(os.path.join(save_folder_model, f"epoch_{epoch}_model.pth"), srcnn,
                       optimizer, epoch)

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
    save_folder_result = os.path.join(opt.save_folder, r"SRCNN/results")
    os.makedirs(save_folder_result, exist_ok=True)

    dataset = ImageDatasetSingleToPair(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w, is_same_size=True,
                                       scale_factor=opt.scale_factor, optType=OptType.RESIZE, max_count=16)
    result_loader = DataLoader(dataset=dataset, num_workers=0, batch_size=opt.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srcnn = SRCNN().to(device)
    load_model(opt.load_models_path, srcnn)

    srcnn.eval()
    for idx, datas_hl in tqdm(enumerate(result_loader), total=int(len(result_loader))):
        image_l = datas_hl["test"].to(device)
        image_h = datas_hl["ref"].to(device)
        image_gen = srcnn(image_l)
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
    parser.add_argument('--save_epoch_rate', type=int, default=100, help='How many epochs save once')
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')

    args = parser.parse_args(args=[])

    return args


if __name__ == '__main__':

    para = parse_args()
    para.data_folder = '../data/T91'
    para.save_folder = r"./working/"
    para.img_w = 160
    para.img_h = 160
    para.scale_factor = 8
    para.batch_size = 4

    is_train = True

    if is_train:

        para.epochs = 50
        para.save_epoch_rate = 10
        para.load_models = False
        para.load_models_path = r"./working/SRCNN/models/epoch_50_model.pth"
        train(para)
    else:
        para.load_models = True
        para.load_models_path = r"./working/SRCNN/models/epoch_200_model.pth"
        run(para)
