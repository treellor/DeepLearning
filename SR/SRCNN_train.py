"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2021.12.19
    Description	:
            创建数据集程序
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""

import os
import tarfile
import numpy as np
import math

from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch import no_grad
from tqdm import tqdm


from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, ToPILImage

from SRCNN_data import DatasetFromFolder,DatasetHighLow
from SRCNN_model import SRCNN


def psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    image_diff = outputs - label
    rmse = math.sqrt(np.mean(image_diff ** 2))
    if rmse == 0:
        return 100
    else:
        PNSR = 20 * math.log10(max_val / rmse)
        return PNSR

def validation(mdl, dataloader, running_epoch):
    mdl.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with no_grad():
        for _, data in tqdm(enumerate(dataloader), total=int(len(val_set)/dataloader.batch_size)):
            image_data = data[0].to(device)
            label = data[1].to(device)

            outputs = mdl(image_data)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr

        outputs = outputs.cpu()
#         save_image(outputs, f"../output/val_sr{running_epoch}.png")

    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(val_set)/dataloader.batch_size)
    return final_loss, final_psnr


def show_example(path, SRCNN, device):
    SRCNN.eval()
    data_to_tensor = Compose([ToTensor()])
    data_to_PIL = Compose([ToPILImage()])

    img_show = Image.open(path)
    img_show = data_to_tensor(img_show)  #图像裁剪

    result_image = img_show
    _, h, w = result_image.size()

    resize_image = data_to_PIL(result_image)
    resize_image = resize_image.resize((int(w / 3), int(h / 3)))
    resize_image = resize_image.resize((w, h), Image.Resampling.BICUBIC)
    resize_image = data_to_tensor(resize_image).to(device)

    newHR = SRCNN(resize_image.unsqueeze(0))

    torchvision.utils.save_image(resize_image, './image/LR.png')
    torchvision.utils.save_image(result_image, './image/HR.png')
    torchvision.utils.save_image(newHR, './image/newHR.png')

    im1 = Image.open('./image/LR.png')
    im2 = Image.open('./image/HR.png')
    im3 = Image.open('./image/newHR.png')
    dst = Image.new('RGB', (w * 3, h))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (w, 0))
    dst.paste(im3, (w * 2, 0))
    dst.save('./image/image.png')
    img = Image.open('./image/image.png')
    plt.imshow(img)
    plt.title('new Image')
    plt.show()


if __name__ == '__main__':

    imageForder = '../data/T91/'
    dataset = DatasetFromFolder(imageForder, Compose([RandomCrop([60, 60]), ToTensor()]))
    #dataset  = DatasetHighLow(r"D:\project\Pycharm\DeepLearning\data\coco125\high",  r"D:\project\Pycharm\DeepLearning\data\coco125\low")

    len = dataset.__len__()

    train_set, val_set = torch.utils.data.random_split(dataset, [len-16, 16])
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=8, shuffle=True)
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=16, shuffle=True)
    print(torch.cuda.is_available()  )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRCNN = SRCNN()
    if torch.cuda.device_count() > 1:
        SRCNN = nn.DataParallel(SRCNN)
    SRCNN.to(device)

    optimizer = optim.Adam(SRCNN.parameters())
    criterion = nn.MSELoss().to(device)


    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    NUM_EPOCHS = 31

    new_point = 0
    os.system('mkdir checkpoint')
    os.system('mkdir image')

    for epoch in range(NUM_EPOCHS):
        batch_idx = 0
        running_loss = 0.0
        running_psnr = 0.0
        SRCNN.train()
        for HR, LR in train_loader:
            HR = HR.to(device)
            LR = LR.to(device)

            newHR = SRCNN(LR)
            SRCNN.zero_grad()
            optimizer.zero_grad()# 可能不需要
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
            batch_idx += 1

        if epoch % 15 == 0:
            show_example(imageForder + 't1.png', SRCNN, device)
            print("Epoch:{} batch[{}/{}] loss:{}".format(epoch, batch_idx, len(train_loader), loss))


        final_loss = running_loss / len(train_loader.dataset)
        final_psnr = running_psnr / int(len(train_set) / train_loader.batch_size)
        train_loss.append(final_loss)
        train_psnr.append(final_psnr)

        val_epoch_loss, val_epoch_psnr = validation(SRCNN, val_loader, epoch)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)

        torch.save(SRCNN.state_dict(), './checkpoint/ckpt_%d.pth' % (new_point))
        new_point += 1