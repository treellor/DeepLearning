"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.03.19
    Description	:
            DDPM  模型训练
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
from torchvision.utils import save_image

from utils.data_read import ImageDatasetResizeSingle
from utils.utils import load_model, save_model
from utils.common import EMA
from DDPM_models import GaussianDiffusion
from backbone.unet import UNet

class DDPMConfig:
    def __init__(self):
        self.num_timesteps = 1000
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
    save_folder_image = os.path.join(opt.save_folder, r"DDPM/images")
    save_folder_model = os.path.join(opt.save_folder, r"DDPM/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    dataset_train = ImageDatasetResizeSingle(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w, max_count=160)
    dataloader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # img_shape = (opt.img_channels, opt.img_h, opt.img_w)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    config = DDPMConfig()

    model = UNet(img_channels=opt.img_channels,
                 base_channels=config.base_channels,
                 channel_mults=config.channel_mults,
                 time_emb_dim=config.time_emb_dim,
                 dropout=config.dropout,
                 attention_resolutions=config.attention_resolutions,
                 num_classes=None,  # if not args.use_labels else 10,
                 initial_pad=0,
                 )
    ema = EMA(model, device)

    diffusion = GaussianDiffusion(model, opt.img_channels, (opt.img_h, opt.img_w), num_timesteps=config.num_timesteps,
                                  loss_type="l2").to(device)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=opt.lr)

    # Load pretrained models
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_checkpoint, diffusion, optimizer)

    # Losses
    # adversarial_loss = torch.nn.BCELoss().to(device)

    n_epochs = opt.epochs
    save_epoch = opt.save_epoch.union({n_epochs + trained_epoch})

    # acc_train_loss = 0

    for epoch in range(trained_epoch, trained_epoch + n_epochs):
        # Training
        diffusion.train()
        for batch_idx, imgs in tqdm(enumerate(dataloader_train), desc=f'Training Epoch {epoch}',
                                    total=int(len(dataloader_train))):
            x = imgs.to(device)

            optimizer.zero_grad()
            loss = diffusion(x)
            loss.backward()
            optimizer.step()

            # acc_train_loss += loss.item()
            # diffusion.update_ema()
            ema.update_ema(diffusion.model)

        # Save models and images
        if (epoch + 1) % opt.save_img_rate == 0:
            diffusion.eval()
            samples = diffusion.sample(batch_size=opt.batch_size, device=device)
            save_image(samples.data[:opt.batch_size],
                       os.path.join(save_folder_image, f"epoch_{epoch + 1}_result.png"), nrow=10, normalize=False)
        if epoch + 1 in save_epoch:
            save_model(os.path.join(save_folder_model, f"epoch_{epoch + 1}_models.pth"), diffusion, optimizer,
                       epoch + 1)


def run(opt):
    save_folder_image = os.path.join(opt.save_folder, r"DDPM/results")
    os.makedirs(save_folder_image, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DDPMConfig()
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

    diffusion = GaussianDiffusion(model, opt.img_channels, (opt.img_h, opt.img_w), num_timesteps=config.num_timesteps,
                                  loss_type="l2").to(device)
    load_model(opt.load_models_checkpoint, diffusion)

    # if args.use_labels:
    #     for label in range(10):
    #         y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
    #         samples = diffusion.sample(args.num_images // 10, device, y=y)
    #
    #         for image_id in range(len(samples)):
    #             image = ((samples[image_id] + 1) / 2).clip(0, 1)
    #             torchvision.utils.save_image(image, f"{args.save_dir}/{label}-{image_id}.png")
    # else:

    samples = diffusion.sample(batch_size=opt.batch_size, device=device)
    save_image(samples.data[:opt.batch_size], os.path.join(save_folder_image, f"result.png"), nrow=10, normalize=False)


def parse_args():
    parser = argparse.ArgumentParser(description="You should add those parameter!")
    parser.add_argument('--data_folder', type=str, default='data/coco_sub', help='dataset path')
    parser.add_argument('--img_channels', type=int, default=3, help='the channel of the image')
    parser.add_argument('--img_w', type=int, default=64, help='image width')
    parser.add_argument('--img_h', type=int, default=64, help='image height')

    parser.add_argument('--batch_size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--save_epoch', type=set, default=set(), help='number of saved epochs')
    parser.add_argument('--save_img_rate', type=int, default=5, help='')

    parser.add_argument('--save_folder', type=str, default=r"./working/", help='image save path')
    parser.add_argument('--load_models', type=bool, default=False, help='load pretrained model weight')
    parser.add_argument('--load_models_checkpoint', type=str, default=r"./working/SRGAN/models/discriminator.pth",
                        help='load model path')

    args = parser.parse_args(args=[])  # 不添加args=[] kaggle会报错
    return args


if __name__ == '__main__':

    is_train = True
    if is_train:
        para = parse_args()
        para.data_folder = '../data/face'
        #para.data_folder = r'D:\4-数据\archive\v_2\urban\s1'

        para.seq_length = 256
        para.img_channels = 3
        para.img_w = 24
        para.img_h = 32
        para.epochs = 60
        para.batch_size = 50
        # para.save_epoch = set(range(1, 10, 5))
        para.save_img_rate = 60
        para.load_models = True
        para.load_models_checkpoint = r"./working/DDPM/models/epoch_60_models.pth"
        train(para)
    else:
        para = parse_args()
        para.data_folder = '../data/face'
        para.seq_length = 128
        para.img_channels = 3
        para.img_w = 24
        para.img_h = 32
        para.batch_size = 24

        # para.save_epoch = set(range(1, 100, 10))
        para.load_models = True
        para.load_models_checkpoint = r"./working/DDPM/models/epoch_100_models.pth"
        run(para)
