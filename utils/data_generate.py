"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2022.12.19
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
from PIL import Image
from torchvision import transforms as tfs
from tqdm import tqdm


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def generate_crop_image_set(input_folder, save_folder, crop_size=None, down_sampling=4):
    if crop_size is None:
        crop_size = [128, 128]
    crop_transforms = tfs.Compose([tfs.RandomCrop(crop_size)])

    save_folder_high = save_folder + "\\high"
    save_folder_low = save_folder + "\\low"
    print( save_folder_low )
    if not os.path.exists(save_folder_high):
        os.makedirs(save_folder_high)
    if not os.path.exists(save_folder_low):
        os.makedirs(save_folder_low)

    count = 1
    folders = os.listdir(input_folder)
    f_list = open(os.path.join(save_folder, "images_list.txt"), "w")
    print("开始转换文件")
    for _, f in tqdm(enumerate(folders), total=len(folders)):
        fp = input_folder + f
        if is_image_file(fp):
            img = Image.open(fp)
            w, h = img.size
            if w < crop_size[0] or h < crop_size[1]:
                continue
            if img.mode != 'RGB':
                continue
            print(len(img.split()))
            t = os.path.splitext(fp)[-1]
            file_name = '{:0>8}'.format(count) + t
            img_h = crop_transforms(img)
            img_l = img_h.resize((int(crop_size[0] // down_sampling), int(crop_size[1] // down_sampling)))
            img_h.save(os.path.join(save_folder_high, file_name))
            img_l.save(os.path.join(save_folder_low, file_name))
            f_list.write( file_name + "\n")
            f_list.flush()
            count = count + 1

    f_list.close()


if __name__ == '__main__':
    generate_crop_image_set(r"D:\project\Pycharm\datas\coco128\coco128\images\train2017\\",
                            r"D:\project\Pycharm\datas\coco128\coco128\images\AAA", [256, 256])


