"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.07.21
    Description	:
            基于扩散模型的SAR图像上色算法
    Others		:  //其他内容说明
    History		:
     1.Date:
       Author:
       Modification:
     2.…………
"""

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils.data_read import ImageDatasetPair

from utils.utils import load_model, save_model
from utils.common import EMA
from models.DDPM_models import GaussianDiffusion
from backbone.unet import UNet
from torchsummary import summary


class UNetConfig:
    def __init__(self):
        # self.use_labels = False
        self.base_channels = 128
        self.channel_mults = (1, 2, 2, 2)
        self.num_res_blocks = 2
        self.dropout = 0.1
        self.attention_resolutions = (1,)
        self.time_emb_dim = 128 * 4
        self.num_classes = None if not False else 10
        self.initial_pad = 0


def train(opt):
    save_folder_image = os.path.join(opt.save_folder, r"ColorSARDiff/images")
    save_folder_model = os.path.join(opt.save_folder, r"ColorSARDiff/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    dataset_sar = os.path.join(opt.data_folder, r"sar")
    dataset_optical = os.path.join(opt.data_folder, r"optical")

    dataset = ImageDatasetPair(dataset_optical, dataset_sar, is_Normalize=True)

    img_shape = tuple(dataset[0]['def'].shape)

    data_len = dataset.__len__()
    val_data_len = opt.batch_size * 3
    train_set, val_set = torch.utils.data.random_split(dataset, [data_len - val_data_len, val_data_len])

    test_dataloader = DataLoader(dataset=val_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    train_dataloader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    config = UNetConfig()

    model = UNet(img_channels=opt.img_channels,
                 base_channels=config.base_channels,
                 channel_mults=config.channel_mults,
                 time_emb_dim=config.time_emb_dim,
                 dropout=config.dropout,
                 attention_resolutions=config.attention_resolutions,
                 num_classes=None,  # if not args.use_labels else 10,
                 initial_pad=0,
                 )

    diffusion = GaussianDiffusion(model, opt.img_channels, (opt.img_h, opt.img_w), timesteps=opt.timesteps,
                                  loss_type="l2").to(device)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=opt.lr)

    # Load pretrained models
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_checkpoint, diffusion, optimizer)

    ema = EMA(model, device)
    # Losses
    # adversarial_loss = torch.nn.BCELoss().to(device)

    n_epochs = opt.epochs

    # 读取显示图像
    show_image_optical = torch.stack([dataset[i]["def"] for i in range(0, opt.batch_size)], 0).to(device)
    show_image_sar = torch.stack([dataset[i]["test"] for i in range(0, opt.batch_size)], 0).to(device)

    for epoch in range(trained_epoch + 1, trained_epoch + n_epochs + 1):
        # Training
        diffusion.train()
        for batch_idx, imgs in tqdm(enumerate(train_dataloader), desc=f'Training Epoch {epoch}',
                                    total=int(len(train_dataloader))):
            images_optical = imgs["def"].to(device)
            images_sar = imgs["test"].to(device)


            optimizer.zero_grad()

            loss = diffusion(x=images_optical, y=images_sar)
            loss.backward()
            optimizer.step()

            # acc_train_loss += loss.item()
            # diffusion.update_ema()
            ema.update_ema(diffusion.model)

        # Save models and images
        if epoch % opt.save_epoch_rate == 0 or (epoch == (trained_epoch + n_epochs)):
            diffusion.eval()
            samples = diffusion.sample(batch_size=opt.batch_size, device=device, y=show_image_sar).to(device)

            img_sar = make_grid(show_image_sar, nrow=1, normalize=True).to(device)
            img_optical = make_grid(show_image_optical, nrow=1, normalize=True).to(device)
            gen_color = make_grid(samples, nrow=1, normalize=True).to(device)

            img_grid = torch.cat((img_sar, img_optical, gen_color), -1)
            save_image(img_grid, os.path.join(save_folder_image, f"epoch_{epoch}.png"), normalize=False)

            save_model(os.path.join(save_folder_model, f"epoch_{epoch}_models.pth"), diffusion, optimizer, epoch)


def run(opt):
    save_folder_image = os.path.join(opt.save_folder, r"ColorSARDiff/results")
    os.makedirs(save_folder_image, exist_ok=True)

    dataset_sar = os.path.join(opt.data_folder, r"sar")
    dataset_optical = os.path.join(opt.data_folder, r"optical")
    dataset = ImageDatasetPair(dataset_optical, dataset_sar, is_Normalize=True)
    result_dataloader = DataLoader(dataset=dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = UNetConfig()
    # 添加
    model = UNet(img_channels=opt.img_channels,
                 base_channels=config.base_channels,
                 channel_mults=config.channel_mults,
                 time_emb_dim=config.time_emb_dim,
                 dropout=config.dropout,
                 attention_resolutions=config.attention_resolutions,
                 num_classes=None,  # if not args.use_labels else 10,
                 initial_pad=0,
                 )

    diffusion = GaussianDiffusion(model, opt.img_channels, (opt.img_h, opt.img_w), timesteps=opt.timesteps,
                                  loss_type="l2").to(device)
    load_model(opt.load_models_checkpoint, diffusion)
    # Initialize generator and discriminator

    for batch_idx, images_hl in tqdm(enumerate(result_dataloader), total=int(len(result_dataloader))):
        # Configure model input
        img_sar = images_hl["test"].to(device)
        img_optical = images_hl["def"].to(device)

        samples = diffusion.sample(batch_size=opt.batch_size, device=device, y=img_sar)

        images_gen = samples + img_sar
        img_sar = make_grid(img_sar, nrow=1, normalize=True).to(device)
        img_optical = make_grid(img_optical, nrow=1, normalize=True).to(device)
        gen_color = make_grid(images_gen, nrow=1, normalize=True).to(device)

        img_grid = torch.cat((img_sar, img_optical, gen_color), -1)
        save_image(img_grid, os.path.join(save_folder_image, f"picture_{batch_idx}.png"), normalize=False)


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--data_folder', type=str, default='data/coco_sub', help='dataset path')
    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--img_channels', type=int, default=3, help='the channel of the image')
    parser.add_argument('--img_w', type=int, default=64, help='image width')
    parser.add_argument('--img_h', type=int, default=64, help='image height')
    parser.add_argument('--batch_size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--timesteps", type=int, default=1000, help="迭代次数")
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--save_epoch_rate', type=int, default=100, help='How many epochs save once')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_checkpoint', type=str, default=r"./working/SRDiff/models/checkpoint.pth",
                        help='load model path')

    args = parser.parse_args(args=[])  # 不添加args=[] kaggle会报错
    return args


if __name__ == '__main__':

    para = parse_args()
    para.data_folder = '../data/SAR128'
    para.save_folder = r"./working/"
    para.img_channels = 3
    para.img_w = 128
    para.img_h = 128
    para.batch_size = 4
    para.timesteps = 100
    para.seq_length = 256

    is_train = True
    if is_train:
        para.epochs = 4
        para.save_epoch_rate = 2
        para.load_models = False
        para.load_models_checkpoint = r"./working/ColorSARDiff/models/epoch_1_models.pth"
        train(para)
    else:
        para.load_models = True
        para.load_models_checkpoint = r"./working/ColorSARDiff/models/epoch_10_models.pth"
        run(para)
