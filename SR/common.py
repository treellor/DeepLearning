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


def save_image_train(image_hr, image_lr, image_new, save_folder, epoch_num="Last", is_show=False):
    image_lr_path = os.path.join(save_folder, f'Epoch_{epoch_num}_LR.png')
    image_hr_path = os.path.join(save_folder, f'Epoch_{epoch_num}_HR.png')
    image_gen_path = os.path.join(save_folder, f'Epoch_{epoch_num}_Gen.png')
    image_all_path = os.path.join(save_folder, f'Epoch_{epoch_num}_Image.png')

    torchvision.utils.save_image(make_grid(image_lr, nrow=1, normalize=True), image_lr_path)
    torchvision.utils.save_image(make_grid(image_hr, nrow=1, normalize=True), image_hr_path)
    torchvision.utils.save_image(make_grid(image_new, nrow=1, normalize=True), image_gen_path)

    im1 = Image.open(image_hr_path)
    im2 = Image.open(image_lr_path)
    im3 = Image.open(image_gen_path)

    n, _, h, w = image_hr.size()
    dst = Image.new('RGB', (w * 3, h * n))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (w, 0))
    dst.paste(im3, (w * 2, 0))
    dst.save(image_all_path)

    if is_show:
        img = Image.open(image_all_path)
        plt.imshow(img)
        plt.title('   H                           L                        Gen Image')
        plt.show()


def image_show(imgs):
    # 将tensor转换为numpy数组
    image_np = imgs.cpu().numpy()
    # 调整颜色通道的顺序
    image_np = image_np.transpose((1, 2, 0))
    # 显示图像
    plt.imshow(image_np)
    plt.show()
