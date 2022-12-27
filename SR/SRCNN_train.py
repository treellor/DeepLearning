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

from data_read import ImageDatasetResize, ImageDatasetCrop
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


def save_image(image_hr, image_lr, image_new, save_folder, epoch_num="Last", is_show=False):

    image_lr_path = os.path.join(save_folder, f'Epoch_{epoch_num}_LR.png')
    image_hr_path = os.path.join(save_folder, f'Epoch_{epoch_num}_HR.png')
    image_gen_path = os.path.join(save_folder, f'Epoch_{epoch_num}_Gen.png')
    image_all_path = os.path.join(save_folder, f'Epoch_{epoch_num}_Image.png')

    torchvision.utils.save_image(image_lr, image_lr_path)
    torchvision.utils.save_image(image_hr, image_hr_path)
    torchvision.utils.save_image(image_new, image_gen_path)

    im1 = Image.open(image_hr_path)
    im2 = Image.open(image_lr_path)
    im3 = Image.open(image_gen_path)

    _, _, h, w = image_hr.size()
    dst = Image.new('RGB', (w * 3, h))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (w, 0))
    dst.paste(im3, (w * 2, 0))
    dst.save(image_all_path)
    if is_show:
        img = Image.open(image_all_path)
        plt.imshow(img)
        plt.title('H    L    Gen Image')
        plt.show()


def train(opt):
    save_folder_image = os.path.join(opt.save_folder, r"SRCNN/images")
    save_folder_model = os.path.join(opt.save_folder, r"SRCNN/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    data_folder = opt.folder_data
    batch_size = opt.batch_size
    img_h = opt.crop_img_h
    img_w = opt.crop_img_w
    # dataset = ImageDatasetResize(data_folder, [img_h, img_w], is_same_shape=True)
    dataset = ImageDatasetCrop(data_folder, [img_h, img_w], is_same_shape=True)

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

    #  per_image_mse_loss = F.mse_loss(HR, newHR, reduction='none')
    train_loss_all, val_loss_all = [], []
    train_psnr_all, val_psnr_all = [], []

    # 读取显示图像
    show_image_hr = None
    show_image_lr = None
    n_epochs = opt.epochs
    save_epoch = max(int(n_epochs//opt.save_epoch_n), 1)
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        train_psnr = 0.0
        srcnn.train()
        for images_hl in train_loader:
            LR = images_hl["lr"].to(device)
            HR = images_hl["hr"].to(device)

            if epoch == 0 and show_image_hr is None:
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

        srcnn.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for idx, datas_hl in enumerate(val_loader):
                image_l = datas_hl["lr"].to(device)
                image_h = datas_hl["hr"].to(device)

                image_gen = srcnn(image_l)
                val_loss_content = criterion(image_gen, image_h)
                val_loss += val_loss_content.item()
                val_psnr += psnr(image_h, image_gen)

                # save the testing
                show_example = opt.show_example
                if epoch == n_epochs - 1:
                    show_example = True
                if idx == 0:
                    if (epoch == n_epochs - 1) or (epoch % save_epoch == 0):
                        show_image_gen = srcnn(show_image_lr)
                        save_image(show_image_hr, show_image_lr, show_image_gen, save_folder_image, epoch, show_example)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_psnr = val_psnr / int(len(val_set) / val_loader.batch_size)
        val_loss_all.append(val_epoch_loss)
        val_psnr_all.append(val_epoch_psnr)

        # 保存最新的参数和损失最小的参数
        if (epoch == n_epochs - 1) or (epoch % save_epoch == 0):
            torch.save(srcnn.state_dict(), os.path.join(save_folder_model, f"epoch_{epoch}_model.pth"))

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_all, color='orange', label='train loss')
    plt.plot(val_loss_all, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, r"loss.png"))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr_all, color='green', label='train PSNR dB')
    plt.plot(val_psnr_all, color='blue', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(os.path.join(save_folder_image, r"psnr.png"))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--folder_data', type=str, default='data/T91', help='dataset path')
    parser.add_argument('--crop_img_w', type=int, default=64, help='randomly cropped image width')
    parser.add_argument('--crop_img_h', type=int, default=64, help='randomly cropped image height')
    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_path', type=str, default=r"./working/SRCNN/models/last_ckpt.pth", help='load model path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--save_epoch_n', type=int, default=5, help='number of saved epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--show_example', type=bool, default=False, help='show a validation example')
    args = parser.parse_args(args=[])
    return args


if __name__ == '__main__':

    para = parse_args()
    para.folder_data = '../data/T91'
    para.load_models = True
    train(para)
