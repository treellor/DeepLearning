"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2021.12.19
    Description	:
            网络训练
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

from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, ToPILImage

from data_read import DatasetFromFolder
from SRCNN_model import SRCNN

def psnr(label, outputs, max_val=1.0):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    image_diff = outputs - label
    rmse = math.sqrt(np.mean(image_diff ** 2))
    if rmse == 0:
        return 100
    else:
        PNSR = 20 * math.log10(max_val / rmse)
        return PNSR


def get_image_hl(path):
    data_to_tensor = Compose([ToTensor()])
    img_hr = Image.open(path)
    w, h = img_hr.size

    img_lr = img_hr.resize((int(w / 3), int(h / 3)))
    img_lr = img_lr.resize((w, h), Image.Resampling.BICUBIC)
    img_hr = data_to_tensor(img_hr)
    img_lr = data_to_tensor(img_lr)
    img_hr = img_hr.unsqueeze(0)
    img_lr = img_lr.unsqueeze(0)
    return img_hr, img_lr


def show_example_image(image_hr, image_lr, image_new, save_folder):
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
    img = Image.open(os.path.join(save_folder, 'image.png'))
    plt.imshow(img)
    plt.title('new Image')
    plt.show()


def train(imageForder, n_epochs=31, load_pretrained_model=False):
    folder_save_image = r".\images\SRCNN"
    folder_model = r".\models\SRCNN"

    os.makedirs(folder_save_image, exist_ok=True)
    os.makedirs(folder_model, exist_ok=True)

    #imageForder = '../data/T91/'
    dataset = DatasetFromFolder(imageForder, Compose([RandomCrop([60, 60]), ToTensor()]))
    # dataset  = DatasetHighLow(r"D:\project\Pycharm\DeepLearning\data\coco125\high",  r"D:\project\Pycharm\DeepLearning\data\coco125\low")

    train_set, val_set = torch.utils.data.random_split(dataset, [dataset.__len__() - 16, 16])
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=8, shuffle=True)
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srcnn = SRCNN()

    load_pretrained_models = False
    # Load pretrained models
    if load_pretrained_models:
        srcnn.load_state_dict(torch.load(os.path.join(folder_model, "last_ckpt.pth")))

    if torch.cuda.device_count() > 1:
        srcnn = nn.DataParallel(srcnn)

    srcnn.to(device)

    optimizer = optim.Adam(srcnn.parameters())
    criterion = nn.MSELoss().to(device)

    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []

    best_loss = 1e100
    best_psnr = 0  # 记录最高

    show_example = True

    # 读取显示图像
    show_image_hr, show_image_lr = get_image_hl(imageForder + 't1.png')
    show_image_hr = show_image_hr.to(device)
    show_image_lr = show_image_lr.to(device)

    for epoch in tqdm(range(n_epochs)):
        # print("Epoch:{}".format(epoch))
        running_loss = 0.0
        running_psnr = 0.0
        srcnn.train()
        is_shown = False
        for HR, LR in train_loader:
            HR = HR.to(device)
            LR = LR.to(device)

            newHR = srcnn(LR)
            srcnn.zero_grad()
            optimizer.zero_grad()  # 可能不需要
            loss = criterion(HR, newHR)
            #             per_image_mse_loss = F.mse_loss(HR, newHR, reduction='none')
            #             per_image_psnr = 10 * torch.log10(10 / per_image_mse_loss)
            #             tensor_average_psnr = torch.mean(per_image_psnr).to(device)
            #             loss = tensor_average_psnr
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            batch_psnr = psnr(HR, newHR)
            running_psnr += batch_psnr

        final_loss = running_loss / len(train_loader.dataset)
        final_psnr = running_psnr / int(len(train_set) / train_loader.batch_size)
        train_loss.append(final_loss)
        train_psnr.append(final_psnr)

        # 验证
        srcnn.eval()
        val_loss_temp = 0.0
        val_psnr_temp = 0.0
        with torch.no_grad():
            #  for _, data in tqdm(enumerate(dataloader), total=int(len(val_set) / dataloader.batch_size)):
            for _, data in enumerate(val_loader):
                image_data = data[0].to(device)
                label = data[1].to(device)

                outputs = srcnn(image_data)
                loss_temp = criterion(outputs, label)
                val_loss_temp += loss_temp.item()
                batch_psnr_temp = psnr(label, outputs)
                val_psnr_temp += batch_psnr_temp

        val_epoch_loss = val_loss_temp / len(val_loader.dataset)
        val_epoch_psnr = val_psnr_temp / int(len(val_set) / val_loader.batch_size)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)

        # 保存最新的参数和损失最小的参数
        torch.save(srcnn.state_dict(), os.path.join(folder_model, 'last_ckpt.pth'))
        if best_loss > final_loss:
            torch.save(srcnn.state_dict(), os.path.join(folder_model, 'best_ckpt.pth'))
            best_loss = final_loss

        # 显示训练结果
        if ((epoch % 15 == 0) or epoch == n_epochs - 1) and show_example == True:
            srcnn.eval()
            show_image_new = srcnn(show_image_lr)
            show_example_image(show_image_hr, show_image_lr, show_image_new, folder_save_image)

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('../output/loss.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    # plt.savefig('../output/psnr.png')
    plt.show()


if __name__ == '__main__':
    imageForder = '../data/T91/'
    train(imageForder)
