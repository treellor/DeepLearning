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
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToPILImage, ToTensor
import torchvision.transforms as tfs


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class DatasetFromFolder(Dataset):
    def __init__(self, path, transforms=None, down_sampling=4):
        super(DatasetFromFolder, self).__init__()
        self.down_sampling = down_sampling
        if transforms is None:
            self.transforms = Compose([ToTensor()])
        else:
            t = transforms.transforms
            is_exist = False
            for v in t:
                if isinstance(v, ToTensor):
                    is_exist = True
                    break
            if is_exist:
                self.transforms = transforms
            else:
                self.transforms = Compose([*t, ToTensor()])

        self.data_transform_PIL = Compose([ToPILImage()])
        self.data_transform_Tensor = Compose([ToTensor()])
        self.filePaths = []
        folders = os.listdir(path)
        for f in folders:
            fp = path + f
            if is_image_file(fp):
                self.filePaths.append(path + f)

    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, item):
        img = Image.open(self.filePaths[item])
        if img.mode != 'RGB':
            raise ValueError("Image:{} isn't RGB mode.".format(self.filePaths[item]))

        # img, _cb, _cr = img.convert('YCbCr').split()
        if self.transforms is not None:
            img = self.transforms(img)

        result_image = img
        _, h, w = result_image.size()

        resize_image = self.data_transform_PIL(result_image)
        resize_image = resize_image.resize((int(w / self.down_sampling), int(h / self.down_sampling)))
        resize_image = resize_image.resize((w, h), Image.BICUBIC)
        resize_image = self.data_transform_Tensor(resize_image)

        return result_image, resize_image


class DatasetHighLow(Dataset):
    def __init__(self, path_folder_high, path_folder_low):
        super(DatasetHighLow, self).__init__()

        self.data_transform_Tensor = Compose([ToTensor()])
        self.filePathsHigh = []
        self.filePathsLow = []
        images_high = os.listdir(path_folder_high)
        images_low = os.listdir(path_folder_low)
        for f in images_high:
            if f in images_low:
                if is_image_file(path_folder_high + f):
                    self.filePathsHigh.append(os.path.join(path_folder_high, f))
                    self.filePathsLow.append(os.path.join(path_folder_low, f))

    def __len__(self):
        return len(self.filePathsHigh)

    def __getitem__(self, item):
        img1 = Image.open(self.filePathsHigh[item])
        img2 = Image.open(self.filePathsLow[item])
        image_high = self.data_transform_Tensor(img1)
        image_low = self.data_transform_Tensor(img2)
        return {"lr": image_low, "hr": image_high}
        #return image_high, image_low


class SRGANDataset(Dataset):
    def __init__(self, data_path, ty="train"):
        self.dataset = []
        self.path = data_path
        self.ty = ty
        f = open(os.path.join(data_path, "{}.txt".format(ty)))
        self.dataset.extend(f.readlines())
        f.close()
        self.tfs = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_name = self.dataset[index].strip()
        img = Image.open(os.path.join(self.path, self.ty, "img", img_name))
        label = Image.open(os.path.join(self.path, self.ty, "label", img_name))
        img = self.tfs(img)
        label = self.tfs(label)
        return img, label


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, files, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = tfs.Compose(
            [
                tfs.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                tfs.ToTensor(),
                tfs.Normalize(mean, std),
            ]
        )
        self.hr_transform = tfs.Compose(
            [
                tfs.Resize((hr_height, hr_height), Image.BICUBIC),
                tfs.ToTensor(),
                tfs.Normalize(mean, std),
            ]
        )
        self.files = files

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

import torch
from torch.utils.data import DataLoader
from SRGAN_models import GeneratorResNet, DiscriminatorNet, FeatureExtractor
from torch.autograd import Variable
if __name__ == '__main__':
    datas = DatasetHighLow(r"D:\project\Pycharm\DeepLearning\data\coco125\high",
                           r"D:\project\Pycharm\DeepLearning\data\coco125\low")
    '''
        e = SRGANDataset(r"T:\\srgan")
    a, b = e[0]
    print(a)
    '''

    dataset = DatasetHighLow(r"D:\project\Pycharm\DeepLearning\data\coco125\high",
                             r"D:\project\Pycharm\DeepLearning\data\coco125\low")
    train_set, val_set = torch.utils.data.random_split(dataset, [dataset.__len__() - 32, 32])
    test_dataloader = DataLoader(dataset=val_set, num_workers=0, batch_size=16, shuffle=True)
    train_dataloader = DataLoader(dataset=train_set, num_workers=0, batch_size=16, shuffle=True)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    discriminator = DiscriminatorNet().to(device)
    for  imgs in train_dataloader:

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            # Adversarial ground truths
            #A  = discriminator(imgs_hr)
            #print(A)

           # valid = Variable(Tensor(np.ones((imgs_lr.size(0), 1))), requires_grad=False)
           # fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

