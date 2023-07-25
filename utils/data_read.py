"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2022.12.27
    Description	:
            read the dataset
    Others		:
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tfs

from enum import Enum


# 定义枚举类型
class OptType(Enum):
    """
    定义对读取图像的操作类型
    """
    No = 0  # 不处理
    CROP = 1  # 裁剪
    RESIZE = 2  # 缩放


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class ImageDatasetSingle(Dataset):
    def __init__(self, image_folder, img_H=64, img_W=64, optType=OptType.No, is_Normalize=False, mean=None, std=None,
                 max_count=None):
        super(ImageDatasetSingle, self).__init__()

        transforms_temp = [tfs.ToTensor()]

        if optType == OptType.CROP:
            transforms_temp.append(tfs.RandomCrop((img_H, img_W), Image.BICUBIC))
        elif optType == OptType.RESIZE:
            transforms_temp.append(tfs.Resize((img_H, img_W), tfs.InterpolationMode.BICUBIC))

        if is_Normalize:
            if std is None:
                std = [0.229, 0.224, 0.225]
            if mean is None:
                mean = [0.485, 0.456, 0.406]
            transforms_temp.append(tfs.Normalize(mean, std))

        self.image_transforms = tfs.Compose(transforms_temp)

        self.filePaths = []
        folders = os.listdir(image_folder)
        count = 0
        for f in folders:
            fp = os.path.join(image_folder, f)
            if is_image_file(fp):
                img = Image.open(fp)
                if img.mode != 'RGB':
                    continue
                w, h = img.size
                if w < img_W or h < img_H:
                    continue
                if max_count is not None:
                    if count >= max_count:
                        break
                self.filePaths.append(fp)
                count = count + 1

    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, item):
        img = Image.open(self.filePaths[item])
        if img.mode != 'RGB':
            raise ValueError("Image:{} isn't RGB mode.".format(self.filePaths[item]))
        # img, _cb, _cr = img.convert('YCbCr').split() 图像转换

        read_image = self.image_transforms(img)
        return read_image


class ImageDatasetSingleToPair(Dataset):
    def __init__(self, image_folder, img_H=128, img_W=128, optType=OptType.CROP, scale_factor=4, is_same_size=False,
                 is_Normalize=True,
                 mean=None, std=None, max_count=None):
        """
        :param image_folder: 图像文件夹
        :param img_H: 图像高
        :param img_W: 图像宽
        :param optType: 默认操作
        :param scale_factor: 图像缩放尺度
        :param is_same_size: 缩放后图像是否拉伸到原大小
        :param is_Normalize: 是否进行归一化
        :param mean: 归一化均值
        :param std: 归一化方差
        :param max_count: 读取文件最大数
        """
        super(ImageDatasetSingleToPair, self).__init__()

        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]

        self.is_same_size = is_same_size

        ref_transforms_temp = [tfs.ToTensor()]

        if optType == OptType.CROP:
            ref_transforms_temp.append(tfs.RandomCrop((img_H, img_W), Image.BICUBIC))
        elif optType == OptType.RESIZE:
            ref_transforms_temp.append(tfs.Resize((img_H, img_W), tfs.InterpolationMode.BICUBIC))

        if is_Normalize:
            ref_transforms_temp.append(tfs.Normalize(mean, std))

        self.ref_transforms = tfs.Compose(ref_transforms_temp)

        test_transforms_temp = [
            tfs.Resize((img_H // scale_factor, img_W // scale_factor), tfs.InterpolationMode.BICUBIC)]
        if is_same_size:
            test_transforms_temp.append(tfs.Resize((img_H, img_W), tfs.InterpolationMode.BICUBIC))
        if is_Normalize:
            test_transforms_temp.append(tfs.Normalize(mean, std))

        self.test_transforms = tfs.Compose(test_transforms_temp)

        self.filePaths = []
        folders = os.listdir(image_folder)
        count = 0
        for f in folders:
            fp = os.path.join(image_folder, f)
            if is_image_file(fp):
                img = Image.open(fp)
                if img.mode != 'RGB':
                    continue
                w, h = img.size
                if w < img_W or h < img_H:
                    continue
                if max_count is not None:
                    if count >= max_count:
                        break
                self.filePaths.append(fp)
                count = count + 1

    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, item):
        img = Image.open(self.filePaths[item])
        if img.mode != 'RGB':
            raise ValueError("Image:{} isn't RGB mode.".format(self.filePaths[item]))
        # img, _cb, _cr = img.convert('YCbCr').split() 图像转换

        img_ref = self.ref_transforms(img)
        img_test = self.test_transforms(img_ref)

        return {"ref": img_ref, "test": img_test}


class ImageDatasetPair(Dataset):
    """
        读取2幅成对图片，不做裁剪处理
    """

    def __init__(self, path_folder_def, path_folder_test, is_Normalize=False, mean=None, std=None, max_count=None):
        super(ImageDatasetPair, self).__init__()

        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]

        self.transforms_Tensor = tfs.Compose([tfs.ToTensor()])
        self.is_Normalize = is_Normalize
        if self.is_Normalize:
            self.transforms_Normalize = tfs.Compose([tfs.Normalize(mean, std)])

        self.filePathsDef = []
        self.filePathsTest = []
        images_Def = os.listdir(path_folder_def)
        images_Test = os.listdir(path_folder_test)
        count = 0
        for f in images_Def:
            if f in images_Test:
                if is_image_file(path_folder_def + f):
                    if max_count is not None:
                        if count >= max_count:
                            break

                    count = count + 1
                    self.filePathsDef.append(os.path.join(path_folder_def, f))
                    self.filePathsTest.append(os.path.join(path_folder_test, f))

    def __len__(self):
        return len(self.filePathsDef)

    def __getitem__(self, item):
        img1 = Image.open(self.filePathsDef[item])
        img2 = Image.open(self.filePathsTest[item])
        image_def = self.transforms_Tensor(img1)
        image_test = self.transforms_Tensor(img2)
        if self.is_Normalize:
            image_def = self.transforms_Normalize(image_def)
            image_test = self.transforms_Normalize(image_test)
        return {"def": image_def, "test": image_test}


if __name__ == '__main__':
    dataset = ImageDatasetPair(r"D:\project\Pycharm\DeepLearning\data\coco125\high",
                               r"D:\project\Pycharm\DeepLearning\data\coco125\low", is_Normalize=True)
