import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.
    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0  # 确保输入的维度是偶数
        self.dim = dim
        self.scale = scale  # 缩放因子，默认为1.0

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2  # 计算输入维度的一半
        emb = math.log(10000) / half_dim  # 计算编码中的常数项
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 计算编码的正弦因子
        emb = torch.outer(x * self.scale, emb)  # 计算输入张量与正弦因子的外积
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored  #为了后面同一调用，这里多了2个没用的参数
        y: ignored
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored   #为了后面同一调用，这里多了2个没用的参数
        y: ignored
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb, y):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    __doc__ = r"""Applies QKV self-attention with a residual connection.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance(in), group(gn), batch(bn), or None(none)). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()

        self.in_channels = in_channels
        self.norm = norm
        if self.norm == "gn":
            self.norm_layer = nn.GroupNorm(num_groups, in_channels)
        elif self.norm == "in":
            self.norm_layer = nn.InstanceNorm2d(in_channels)
        elif self.norm == "bn":
            self.norm_layer = nn.BatchNorm2d(in_channels)
        elif self.norm == "none":
            self.norm_layer = None
        else:
            raise ValueError(f"Invalid normalization method: {self.norm}")

        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量 x 的形状信息：batch_size (b), 通道数 (c), 高度 (h), 宽度 (w)

        if self.norm_layer is not None:
            x = self.norm_layer(x)  # 如果设置了归一化层，则对输入张量进行归一化处理

        # 将输入张量 x 经过线性转换得到 q (query), k (key), v (value) 三个张量，并按通道数进行切分
        q, k, v = torch.split(self.to_qkv(x), self.in_channels, dim=1)

        # 将 q 张量的维度进行重新排列，将通道数维度放到最后，然后展平为形状 (batch_size, 高度*宽度, 通道数)
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        # 将 k 张量的维度进行变换，变为形状 (batch_size, 通道数, 高度*宽度)
        k = k.view(b, c, h * w)
        # 将 v 张量的维度进行重新排列，将通道数维度放到最后，然后展平为形状 (batch_size, 高度*宽度, 通道数)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        # 计算 q 和 k 的点积，并对其除以一个缩放因子，用于缓解点积的大小影响
        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        # 对点积结果进行 softmax 操作，得到注意力权重
        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)  # 利用注意力权重对 v 进行加权求和，得到注意力向量
        assert out.shape == (b, h * w, c)
        # 将注意力向量的形状变换为 (batch_size, 通道数, 高度, 宽度)，并对维度进行重新排列
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        # 将注意力向量加上输入张量 x，得到最终输出
        return self.to_out(out) + x


class ResidualBlock(nn.Module):
    __doc__ = r"""Applies two conv blocks with residual connection. Adds time and class conditioning by adding bias after first convolution.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        class_index: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        dropout (float): dropout rate. Default: 0.1
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): which normalization to use (instance(in), group(gn), batch(bn), or None(none)). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        use_attention (bool): if True applies AttentionBlock to the output. Default: False
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            dropout=0.1,
            time_emb_dim=None,
            num_classes=None,
            activation=F.relu,
            norm="gn",
            num_groups=32,
            use_attention=False,
    ):
        super().__init__()

        self.activation = activation  # 设置激活函数，用于卷积层之后的激活操作

        self.norm_1 = nn.GroupNorm(num_groups, in_channels)  # 第一个批归一化层，对输入数据进行归一化处理
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm_2 = nn.GroupNorm(num_groups, out_channels)
        self.conv_2 = nn.Sequential(  # 第二个卷积层，包含了一个丢弃层和一个二维卷积层
            nn.Dropout(p=dropout),  # 丢弃层，用于在训练时随机将部分神经元置为0，以防止过拟合
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        # 时间条件偏置，用于根据时间嵌入调整特征图
        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        # 类别条件偏置，用于根据类别调整特征图
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None
        # 残差连接层，如果输入通道数和输出通道数不相等，则进行卷积操作；如果相等，则为恒等映射
        self.residual_connection = nn.Conv2d(in_channels, out_channels,
                                             1) if in_channels != out_channels else nn.Identity()
        # 注意力层，如果不使用注意力，则为恒等映射；如果使用注意力，则实例化 AttentionBlock 类，将注意力块作为一个组件添加到残差块中
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)

    def forward(self, x, time_emb=None, class_index=None):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]

        if self.class_bias is not None:
            if class_index is None:
                raise ValueError("class conditioning was specified but class_index is not passed")

            out += self.class_bias(class_index)[:, :, None, None]

        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        out = self.attention(out)

        return out


class UNet(nn.Module):
    __doc__ = """UNet model used to estimate noise.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        class_index: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        img_channels (int): number of image channels
        base_channels (int): number of base channels (after the first convolution)
        is_cond_image (bool): Flag indicating whether the input is a conditional image. .Default: False.
        channel_mults (tuple): tuple of channel multipliers. Default: (1, 2, 4, 8)
        num_res_blocks (int): number of residual blocks in each downsampling and upsampling layer. Default: 2
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        time_emb_scale (float): linear scale to be applied to timesteps. Default: 1.0
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        dropout (float): dropout rate at the end of each residual block. Default: 0.1
        activation (function): activation function. Default: torch.nn.functional.relu
        attention_layers (tuple): list of relative resolutions at which layers to apply attention. The topmost layer is 
                            denoted as 0, and subsequent layers are denoted as 1, 2, 3, and so on. Default: ()
        norm (string or None): which normalization to use (instance(in), group(gn), batch(bn), or None(none)). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        initial_pad (int): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
    """

    def __init__(
            self,
            img_channels,
            base_channels=128,
            is_cond_image=False,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=None,
            time_emb_scale=1.0,
            num_classes=None,
            dropout=0.1,
            activation=F.relu,
            attention_layers=(),
            norm="gn",
            num_groups=32,
            initial_pad=0,
    ):
        super().__init__()

        self.initial_pad = initial_pad
        self.activation = activation
        self.is_cond_image = is_cond_image  # 是否添加控制图像
        # 类编码
        self.num_classes = num_classes
        # 时间编码器
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None

        # 初始化卷积层
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        # 下采样和上采样的残差块
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels
        # 下采样残差块
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=self.activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_layers,
                ))
                now_channels = out_channels
                channels.append(now_channels)
            # 添加完残差快后，进行降采样
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        # 中间残差块
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=self.activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True if len(attention_layers) > 0 else False  # 如果使用attention，设置为True，否则设置为False
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=self.activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,
            ),
        ])

        # 上采样的残差块
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=self.activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_layers,
                ))
                now_channels = out_channels

            # 参考快后升采样
            if i != 0:
                self.ups.append(Upsample(now_channels))

        assert len(channels) == 0

        # 输出卷积层
        self.out_norm = nn.GroupNorm(num_groups, base_channels)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x, time=None, class_index=None, cond_image=None):
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        # 时间编码处理
        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")

            time_emb = self.time_mlp(time)
        else:
            time_emb = None

        # 类别编码处理
        if self.num_classes is not None and class_index is None:
            raise ValueError("class conditioning was specified but class_index is not passed")

        if self.is_cond_image is True and cond_image is None:
            raise ValueError("image conditioning was specified but cond_image is not passed")

        x = self.init_conv(x)
        if self.is_cond_image is True:
            cond_image = self.init_conv(cond_image)
            x = x + cond_image

        # 存储各层的特征图
        skips = [x]

        # 下采样
        for layer in self.downs:
            x = layer(x, time_emb, class_index)
            skips.append(x)

        # 中间残差块
        for layer in self.mid:
            x = layer(x, time_emb, class_index)

        # 上采样
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, class_index)

        # 输出处理
        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        # 去除额外填充
        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x


class UNetConfig:
    def __init__(self):
        self.base_channels = 128  # Unet网络中第一次卷积操作的输入通道数
        self.is_cond_image=False  # 是否输入控制图像，图像超分等需要
        self.channel_mults = (1, 2, 4, 8)  # Unet网络各层通道数相对于base_channels的倍数
        self.num_res_blocks = 2  # Unet网络中每个下采样/上采样块中的残差块个数
        self.time_emb_dim = None,  # 时间编码维度（可选，如果使用时间编码）
        self.time_emb_scale = 1.0,  # 时间编码的线性缩放因子（可选，如果使用时间编码）
        self.num_classes = None  # 分类类别数量（可选，如果进行分类任务）
        self.dropout = 0.1  # 残差块中的dropout率
        self.activation = F.silu,  # Unet网络中的激活函数（默认为SiLU函数）
        self.attention_resolutions = ()  # 需要应用注意力机制的相对分辨率列表（可选） 四层都添加(0,1,2,3)
        norm = "gn",  # 选择的归一化方法，可以是"gn"（GroupNorm）、"in"（InstanceNorm）、"bn"（BatchNorm）、None（无归一化）
        num_groups = 32,  # 归一化中的分组数（如果使用GroupNorm）
        self.initial_pad = 0  # 输入图像的初始填充大小（用于处理非2的幂次大小的图像）
