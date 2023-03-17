"""
    Author		:  Treellor
    Version		:  v1.0
    Date		:  2023.02.28
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
import numpy as np
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from GAN_models import Generator, Discriminator
from utils.data_read import ImageDatasetResizeSingle
from utils.utils import load_model, save_model

import argparse
import datetime
import torch

from torch.utils.data import DataLoader
from torchvision import datasets

from DDPM_models import get_diffusion_from_args

class DDPMConfig:
    def __init__(self):
        self.num_timesteps = 1000
        self.schedule = "linear"
        self.loss_type = "l2"
        self.use_labels = False

        self.base_channels = 128
        self.channel_mults = (1, 2, 2, 2)
        self.num_res_blocks = 2
        self.norm = "gn"
        self.dropout = 0.1
        self.activation = "silu"
        self.attention_resolutions = (1,)

        self.ema_decay = 0.9999
        self.ema_update_rate = 1

        self.img_channels = 3

        self.time_emb_dim = 128 * 4

        self.num_classes = None  if not  False else 10
        self.initial_pad = 0

        self.schedule_low=1e-4
        self.schedule_high=0.02

#
# def create_argparser():
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
#     defaults = dict(
#         learning_rate=2e-4,
#         batch_size=128,
#         iterations=800000,
#
#         log_to_wandb=False,
#         log_rate=1000,
#         checkpoint_rate=1000,
#         log_dir="~/ddpm_logs",
#         project_name=None,
#         run_name=run_name,
#
#         model_checkpoint=None,
#         optim_checkpoint=None,
#
#         schedule_low=1e-4,
#         schedule_high=0.02,
#
#         device=device,
#     )
#     defaults.update(diffusion_defaults())
#
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser


# def cycle(dl):
#     """
#     https://github.com/lucidrains/denoising-diffusion-pytorch/
#     """
#     while True:
#         for data in dl:
#             yield data


# def str2bool(v):
#     """
#     https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
#     """
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ("yes", "true", "t", "y", "1"):
#         return True
#     elif v.lower() in ("no", "false", "f", "n", "0"):
#         return False
#     else:
#         raise argparse.ArgumentTypeError("boolean value expected")


# def add_dict_to_argparser(parser, default_dict):
#     """
#     https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
#     """
#     for k, v in default_dict.items():
#         v_type = type(v)
#         if v is None:
#             v_type = str
#         elif isinstance(v, bool):
#             v_type = str2bool
#         parser.add_argument(f"--{k}", default=v, type=v_type)


def train(opt):
    save_folder_image = os.path.join(opt.save_folder, r"DDPM/images")
    save_folder_model = os.path.join(opt.save_folder, r"DDPM/models")
    os.makedirs(save_folder_image, exist_ok=True)
    os.makedirs(save_folder_model, exist_ok=True)

    dataset_train = ImageDatasetResizeSingle(opt.data_folder, img_H=opt.img_h, img_W=opt.img_w)
    dataloader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batch_size, shuffle=True)

    # img_shape = (opt.img_channels, opt.img_h, opt.img_w)

    # args = create_argparser().parse_args()
    #    device = args.device

    # u_net = UNet(         )
    #
    #
    # model = UNet(
    #     img_channels=3,
    #
    #     base_channels=args.base_channels,
    #     channel_mults=args.channel_mults,
    #     time_emb_dim=args.time_emb_dim,
    #     norm=args.norm,
    #     dropout=args.dropout,
    #     activation=activations[args.activation],
    #     attention_resolutions=args.attention_resolutions,
    #
    #     num_classes=None if not args.use_labels else 10,
    #     initial_pad=0,
    # )

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    diff = DDPMConfig()

    # 添加
    diffusion = get_diffusion_from_args(diff).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=opt.lr)

    # Load pretrained models
    trained_epoch = 0
    if opt.load_models:
        trained_epoch = load_model(opt.load_models_checkpoint, diffusion, optimizer)

    # Losses
    # adversarial_loss = torch.nn.BCELoss().to(device)

    n_epochs = opt.epochs
    save_epoch = opt.save_epoch.union({n_epochs + trained_epoch})

    acc_train_loss = 0

    # 读取用去显示图像保存
    # show_data1 = dataset_train[0]
    # show_data2 = dataset_train[1]
    # show_data3 = dataset_train[2]
    # show_data4 = dataset_train[3]
    # test_image = torch.stack([show_data1, show_data2, show_data3, show_data4], 0).to(device)

    for epoch in range(trained_epoch, trained_epoch + n_epochs):
        # Training
        diffusion.train()

        for batch_idx, img_real in tqdm(enumerate(dataloader_train), desc=f'Training Epoch {epoch}',
                                        total=int(len(dataloader_train))):

            x = img_real.to(device)
            loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            diffusion.update_ema()

            # Save image
            if epoch + 1 in save_epoch:
                if batch_idx == 0:
                    current_epoch = epoch + 1
                    save_model(os.path.join(save_folder_model, f"epoch_{current_epoch}_models.pth"), diffusion,
                               optimizer, current_epoch)


def create_argparser2():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=10000, device=device)
    defaults.update(diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    add_dict_to_argparser(parser, defaults)
    return parser


def run(opt):
    save_folder_image = os.path.join(opt.save_folder, r"DDPM/results")
    os.makedirs(save_folder_image, exist_ok=True)

    # Initialize generator and discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


###############################################################
# args = create_argparser()2.parse_args()
# device = args.device
#
# try:
#     diffusion = get_diffusion_from_args(args).to(device)
#     diffusion.load_state_dict(torch.load(args.model_path))
#
#     if args.use_labels:
#         for label in range(10):
#             y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
#             samples = diffusion.sample(args.num_images // 10, device, y=y)
#
#             for image_id in range(len(samples)):
#                 image = ((samples[image_id] + 1) / 2).clip(0, 1)
#                 torchvision.utils.save_image(image, f"{args.save_dir}/{label}-{image_id}.png")
#     else:
#         samples = diffusion.sample(args.num_images, device)
#
#         for image_id in range(len(samples)):
#             image = ((samples[image_id] + 1) / 2).clip(0, 1)
#             torchvision.utils.save_image(image, f"{args.save_dir}/{image_id}.png")
# except KeyboardInterrupt:
#     print("Keyboard interrupt, generation finished early")
#
#
#
#
# img_shape = (opt.img_channels, opt.img_h, opt.img_w)
#
# generator = Generator(seq_length=opt.seq_length, img_shape=img_shape).to(device)
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# # Load pretrained models
# load_model(opt.load_models_path_gen, generator, optimizer_G)
#
# generator.eval()
# img_n = 200
# z = Variable(Tensor(np.random.normal(0, 1, (img_n, opt.seq_length))))
# img_gen = generator(z)
# save_image(img_gen.data[:img_n], os.path.join(save_folder_image, f"results.png"), nrow=10, normalize=False)


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
        para.seq_length = 128
        para.img_channels = 3
        para.img_w = 24
        para.img_h = 32
        para.epochs = 100
        para.batch_size = 64

        # para.save_epoch = set(range(1, 100, 10))
        para.load_models = False
        para.load_models_path_gen = r"./working/GAN/models/epoch_100_generator.pth"
        para.load_models_path_dis = r"./working/GAN/models/epoch_100_discriminator.pth"
        train(para)
    else:
        para = parse_args()
        para.data_folder = '../data/face'
        para.seq_length = 128
        para.img_channels = 3
        para.img_w = 24
        para.img_h = 32
        para.batch_size = 64

        # para.save_epoch = set(range(1, 100, 10))
        para.load_models = True
        para.load_models_path_gen = r"./working/GAN/models/epoch_100_generator.pth"
        para.load_models_path_dis = r"./working/GAN/models/epoch_100_discriminator.pth"
        run(para)
