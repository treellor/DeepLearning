"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2021.12.19
    Description	:
            SRGAN 仿真模型
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""

import os
import numpy as np
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_read import ImageDatasetResize
from SRCNN_model import SRCNN


def psnr(label, outputs, max_val=1.0):
    """
        per_image_psnr = 10 * torch.log10(10 / per_image_mse_loss)
        tensor_average_psnr = torch.mean(per_image_psnr).to(device)
        loss = tensor_average_psnr
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    image_diff = outputs - label
    rmse = math.sqrt(np.mean(image_diff ** 2))
    if rmse == 0:
        return 100
    else:
        PNSR = 20 * math.log10(max_val / rmse)
        return PNSR


def save_example_image(image_hr, image_lr, image_new, save_folder, is_show=False):
    _, _, h, w = image_hr.size()
    torchvision.utils.save_image(image_lr, os.path.join(save_folder, 'LR.png'))
    torchvision.utils.save_image(image_hr, os.path.join(save_folder, 'HR.png'))
    torchvision.utils.save_image(image_new, os.path.join(save_folder, 'newHR.png'))

    im1 = Image.open(os.path.join(save_folder, 'LR.png'))
    im2 = Image.open(os.path.join(save_folder, 'HR.png'))
    im3 = Image.open(os.path.join(save_folder, 'newHR.png'))

    dst = Image.new('RGB', (w * 3, h))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (w, 0))
    dst.paste(im3, (w * 2, 0))
    dst.save(os.path.join(save_folder, 'image.png'))
    if is_show:
        img = Image.open(os.path.join(save_folder, 'image.png'))
        plt.imshow(img)
        plt.title('new Image')
        plt.show()


def train(opt):
    folder_save_image = opt.folder_save_image
    folder_save_model = opt.folder_save_model
    os.makedirs(folder_save_image, exist_ok=True)
    os.makedirs(folder_save_model, exist_ok=True)

    data_folder = opt.folder_data
    batch_size = opt.batch_size
    img_h = opt.crop_img_h
    img_w = opt.crop_img_w
    dataset = ImageDatasetResize(data_folder, [img_h, img_w], is_same_shape=True)
    # dataset = ImageDatasetCrop(data_folder, [256,128], is_same_shape=True)

    data_len = dataset.__len__()
    val_data_len = int(data_len * 0.25)
    train_set, val_set = torch.utils.data.random_split(dataset, [data_len - val_data_len, val_data_len])
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=int(batch_size // 2), shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srcnn = SRCNN()
    if opt.load_models:
        srcnn.load_state_dict(torch.load(opt.load_models_path))

    if torch.cuda.device_count() > 1:
        srcnn = nn.DataParallel(srcnn)

    srcnn.to(device)

    optimizer = optim.Adam(srcnn.parameters())
    criterion = nn.MSELoss().to(device)

    #             per_image_mse_loss = F.mse_loss(HR, newHR, reduction='none')

    train_loss_all, val_loss_all = [], []
    train_psnr_all, val_psnr_all = [], []

    best_loss = 1e100

    # 读取显示图像
    show_example = opt.show_example
    show_image_hr = None
    show_image_lr = None
    n_epochs = opt.epochs
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        train_psnr = 0.0
        srcnn.train()
        for images_hl in train_loader:
            LR = images_hl["lr"].to(device)
            HR = images_hl["hr"].to(device)

            if show_example and epoch == 0 and show_image_hr is None:
                show_image_hr = HR
                show_image_lr = LR

            newHR = srcnn(LR)
            srcnn.zero_grad()
            optimizer.zero_grad()
            train_loss_content = criterion(HR, newHR)
            train_loss_content.backward(retain_graph=True)
            optimizer.step()
            train_loss += train_loss_content.item()
            train_psnr += psnr(HR, newHR)

        final_loss = train_loss / len(train_loader.dataset)
        final_psnr = train_psnr / int(len(train_set) / train_loader.batch_size)
        train_loss_all.append(final_loss)
        train_psnr_all.append(final_psnr)

        # 验证
        srcnn.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            #  for _, data in tqdm(enumerate(dataloader), total=int(len(val_set) / dataloader.batch_size)):
            for idx, datas_hl in enumerate(val_loader):
                image_l = datas_hl["lr"].to(device)
                image_h = datas_hl["hr"].to(device)

                # image_l = data[0].to(device)
                # image_h = data[1].to(device)
                image_gen = srcnn(image_l)
                val_loss_content = criterion(image_gen, image_h)
                val_loss += val_loss_content.item()
                val_psnr += psnr(image_h, image_gen)
                # 显示训练结果
                if idx == 0:
                    if epoch == n_epochs - 1:
                        show_image_gen = srcnn(show_image_lr)
                        save_example_image(show_image_hr, show_image_lr, show_image_gen, folder_save_image, True)
                    else:
                        if show_example:
                            if epoch % 10 == 0:
                                show_image_gen = srcnn(show_image_lr)
                                save_example_image(show_image_hr, show_image_lr, show_image_gen, folder_save_image,
                                                   True)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_psnr = val_psnr / int(len(val_set) / val_loader.batch_size)
        val_loss_all.append(val_epoch_loss)
        val_psnr_all.append(val_epoch_psnr)

        # 保存最新的参数和损失最小的参数
        torch.save(srcnn.state_dict(), os.path.join(folder_save_model, 'last_ckpt.pth'))
        if best_loss > final_loss:
            torch.save(srcnn.state_dict(), os.path.join(folder_save_model, 'best_ckpt.pth'))
            best_loss = final_loss

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_all, color='orange', label='train loss')
    plt.plot(val_loss_all, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('../output/loss.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr_all, color='green', label='train PSNR dB')
    plt.plot(val_psnr_all, color='blue', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    # plt.savefig('../output/psnr.png')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--folder_data', type=str, default='data/coco128.yaml', help='dataset path')
    parser.add_argument('--crop_img_w', type=int, default=64, help='randomly cropped image width')
    parser.add_argument('--crop_img_h', type=int, default=64, help='randomly cropped image height')
    parser.add_argument('--folder_save_image', type=str, default=r".\images\SRCNN", help='image save path')
    parser.add_argument('--folder_save_model', type=str, default=r".\models\SRCNN", help='model save path')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_path', type=str, default=r".\models\SRCNN\last_ckpt.pth", help='load model path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--show_example', type=bool, default=True, help='show a validation example')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    para = parse_args()
    para.folder_data = '../data/T91/'
    para.load_models = False
    train(para)
