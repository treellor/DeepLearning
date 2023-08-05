"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.3.3
    Description	:
            定义公共函数
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""

import os
import torchvision
from PIL import Image
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


class AverageMeter(object):
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(save_path, model, optimizer, epoch_n):
    torch.save({"model_dict": model.state_dict(), "optimizer_dict": optimizer.state_dict(), "epoch_n": epoch_n},
               save_path)


def load_model(save_path, model, optimizer=None):
    model_data = torch.load(save_path)
    model.load_state_dict(model_data["model_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(model_data["optimizer_dict"])
    epoch_n = model_data["epoch_n"]
    return epoch_n


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def calc_psnr2(img1, img2):
    return 20. * torch.log10(1. / torch.sqrt(torch.mean((img1 - img2) ** 2)))

