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


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class ImageDatasetResize(Dataset):
    """
     scale the image to fix shape
    """

    def __init__(self, path, resize_shape_hw=None, is_same_shape=False, down_sampling=4, is_Normalize=False,
                 mean=None, std=None, max_count=None):

        super(ImageDatasetResize, self).__init__()

        if resize_shape_hw is None:
            resize_shape_hw = [128, 128]
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        hr_height, hr_width = resize_shape_hw

        hr_trans = [tfs.Resize((hr_height, hr_height), tfs.InterpolationMode.BICUBIC), tfs.ToTensor()]
        if is_Normalize:
            hr_trans.append(tfs.Normalize(mean, std))
        self.hr_transforms = tfs.Compose(hr_trans)

        lr_trans = [tfs.Resize((hr_height // down_sampling, hr_width // down_sampling), tfs.InterpolationMode.BICUBIC),
                    tfs.ToTensor()]
        if is_Normalize:
            lr_trans.append(tfs.Normalize(mean, std))
        self.lr_transforms = tfs.Compose(lr_trans)

        self.is_same_shape = is_same_shape
        if is_same_shape:
            self.data_transform_PIL = tfs.Compose([tfs.ToPILImage()])

        self.filePaths = []
        folders = os.listdir(path)
        count = 0
        for f in folders:
            fp = os.path.join(path, f)
            if is_image_file(fp):
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
        img_lr = self.lr_transforms(img)
        img_hr = self.hr_transforms(img)
        if self.is_same_shape:
            img_lr = self.data_transform_PIL(img_lr)
            img_lr = self.hr_transforms(img_lr)
        return {"lr": img_lr, "hr": img_hr}


class ImageDatasetCrop(Dataset):
    """
    crop the images to fix shape
    """

    def __init__(self, path, crop_shape_hw=None, is_same_shape=False, down_sampling=4, is_Normalize=False,
                 mean=None, std=None, max_count=None):

        super(ImageDatasetCrop, self).__init__()

        if crop_shape_hw is None:
            crop_shape_hw = [128, 128]
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        self.is_save_shape = is_same_shape
        hr_height, hr_width = crop_shape_hw
        self.crop_transforms = tfs.Compose([tfs.RandomCrop((hr_height, hr_height), Image.BICUBIC)])
        self.resize_transforms_down = tfs.Compose([tfs.Resize((hr_height // down_sampling, hr_width // down_sampling),
                                                              tfs.InterpolationMode.BICUBIC)])

        hr_trans = [tfs.ToTensor()]
        if is_Normalize:
            hr_trans.append(tfs.Normalize(mean, std))
        self.hr_transforms = tfs.Compose(hr_trans)

        lr_trans = [tfs.ToTensor()]
        if is_same_shape:
            lr_trans.append(tfs.Resize((hr_height, hr_width), tfs.InterpolationMode.BICUBIC))
        if is_Normalize:
            lr_trans.append(tfs.Normalize(mean, std))
        self.lr_transforms = tfs.Compose(lr_trans)

        self.filePaths = []
        folders = os.listdir(path)
        count = 0
        for f in folders:
            fp = os.path.join(path, f)

            img = Image.open(fp)
            if img.mode != 'RGB':
                continue
            w, h = img.size
            if w < hr_width or h < hr_height:
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

        crop_image = self.crop_transforms(img)
        image_down = self.resize_transforms_down(crop_image)
        img_hr = self.hr_transforms(crop_image)
        img_lr = self.lr_transforms(image_down)
        return {"lr": img_lr, "hr": img_hr}


class ImageDatasetHighLow(Dataset):
    def __init__(self, path_folder_high, path_folder_low,
                 is_Normalize=False, mean=None, std=None, max_count=None):
        super(ImageDatasetHighLow, self).__init__()

        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        self.transforms_Tensor = tfs.Compose([tfs.ToTensor()])
        self.is_Normalize = is_Normalize
        if self.is_Normalize:
            self.transforms_Normalize = tfs.Compose([tfs.Normalize(mean, std)])
        self.filePathsHigh = []
        self.filePathsLow = []
        images_high = os.listdir(path_folder_high)
        images_low = os.listdir(path_folder_low)
        count = 0
        for f in images_high:
            if f in images_low:
                if is_image_file(path_folder_high + f):
                    if max_count is not None:
                        if count >= max_count:
                            break
                    else:
                        count = count + 1
                        self.filePathsHigh.append(os.path.join(path_folder_high, f))
                        self.filePathsLow.append(os.path.join(path_folder_low, f))

    def __len__(self):
        return len(self.filePathsHigh)

    def __getitem__(self, item):
        img1 = Image.open(self.filePathsHigh[item])
        img2 = Image.open(self.filePathsLow[item])
        image_high = self.transforms_Tensor(img1)
        image_low = self.transforms_Tensor(img2)
        if self.is_Normalize:
            image_high = self.transforms_Normalize(image_high)
            image_low = self.transforms_Normalize(image_low)
        return {"lr": image_low, "hr": image_high}


if __name__ == '__main__':
    dataset = ImageDatasetHighLow(r"D:\project\Pycharm\DeepLearning\data\coco125\high",
                                  r"D:\project\Pycharm\DeepLearning\data\coco125\low", is_Normalize=True)
