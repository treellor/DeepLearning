import os
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToPILImage, ToTensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class DatasetFromFolder(Dataset):
    def __init__(self, path, transforms=None, downsample=2):
        super(DatasetFromFolder, self).__init__()
        self.downsample = downsample
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

        #img, _cb, _cr = img.convert('YCbCr').split()
        if self.transforms is not None:
            img = self.transforms(img)

        result_image = img   #self.data_transform_Tensor(img)

        _, h, w = result_image.size()

        resize_image = self.data_transform_PIL(result_image)
        resize_image = resize_image.resize((int(w / self.downsample), int(h / self.downsample)))
        resize_image = resize_image.resize((w, h), Image.Resampling.BICUBIC)
        resize_image = self.data_transform_Tensor(resize_image)

        return result_image, resize_image

'''
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
'''