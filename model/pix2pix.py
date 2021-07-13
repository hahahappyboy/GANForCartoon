import torch
import torch.nn as nn


"""搭建生成器"""

class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64): # U-net结构 3 256 256
        super(UnetGenerator, self).__init__()
        # 下采样层，把图片编码成一系列特征
        self.down1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1) # 64 128 128
        # 下采样 LeakyReLU Conv2d BatchNorm2d
        self.down2 = Downsample(ngf, ngf * 2) # 128 64 64
        self.down3 = Downsample(ngf * 2, ngf * 4) # 256 32 32
        self.down4 = Downsample(ngf * 4, ngf * 8) # 512 16 16
        self.down5 = Downsample(ngf * 8, ngf * 8) # 512 8 8
        self.down6 = Downsample(ngf * 8, ngf * 8) # 512 4 4
        self.down7 = Downsample(ngf * 8, ngf * 8) # 512 2 2

        self.center = Downsample(ngf * 8, ngf * 8) # 512 1 1
        # 把特征重建出一张一张图片
        # 上采样 ReLU ConvTranspose2d BatchNorm2d
        self.up7 = Upsample(ngf * 8, ngf * 8, use_dropout=True) # 1024 2 2
        self.up6 = Upsample(ngf * 8 * 2, ngf * 8, use_dropout=True) # 1024 4 4
        self.up5 = Upsample(ngf * 8 * 2, ngf * 8, use_dropout=True) # 1024 8 8
        self.up4 = Upsample(ngf * 8 * 2, ngf * 8) # 1024 16 16
        self.up3 = Upsample(ngf * 8 * 2, ngf * 4) # 512 32 32
        self.up2 = Upsample(ngf * 4 * 2, ngf * 2) # 256 64 64
        self.up1 = Upsample(ngf * 2 * 2, ngf) # 128 128 128

        self.output_block = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1), # 3 256 256
            nn.Tanh()
        )

    def forward(self, x):# 3 256 256
        d1 = self.down1(x)# 64 128 128
        d2 = self.down2(d1)# 128 64 64
        d3 = self.down3(d2)# 256 32 32
        d4 = self.down4(d3)# 512 16 16
        d5 = self.down5(d4) # 512 8 8
        d6 = self.down6(d5) # 512 4 4
        d7 = self.down7(d6)# 512 2 2

        c = self.center(d7)# 512 1 1

        x = self.up7(c, d7) # 1024 2 2
        x = self.up6(x, d6) # 1024 4 4
        x = self.up5(x, d5) # 1024 8 8
        x = self.up4(x, d4) # 1024 16 16
        x = self.up3(x, d3) # 512 32 32
        x = self.up2(x, d2) # 256 64 64
        x = self.up1(x, d1) # 128 128 128

        x = self.output_block(x) # 3 256 256
        return x


class Downsample(nn.Module): # 下采样层，作用：把特征的大小缩小一半
    # LeakyReLU => conv => batch norm
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(Downsample, self).__init__()

        self.layers = nn.Sequential(  # 下采样，
            nn.LeakyReLU(0.2),  # LeakyReLU, leaky=0.2与论文中保持一致
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),  # Conv2D
            nn.BatchNorm2d(out_dim)  # BatchNorm2D

        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Upsample(nn.Module):# 上采样层 作用：把特征扩大一倍
    # ReLU => deconv => batch norm => dropout
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, use_dropout=False):
        super(Upsample, self).__init__()

        sequence = [  # 上采样，
            nn.ReLU(),  # ReLU
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),  # Conv2DTranspose，反卷积
            nn.BatchNorm2d(out_dim)  # nn.BatchNorm2D，归一化
        ]

        if use_dropout:
            sequence.append(nn.Dropout(p=0.5))

        self.layers = nn.Sequential(*sequence)

    def forward(self, x, skip): # 上采样有一个跳跃链接，所以有个cat操作
        x = self.layers(x) # 把x特征扩大一倍
        x = torch.cat([x, skip], dim=1)  # 与skip跳跃连接，沿着第二维度拼接起来 如2*3 2*3 -> 2*6 ,因为为NCHW，按照C拼接起来
        return x


"""鉴别器的搭建"""

class NLayerDiscriminator(nn.Module): # 由多个卷积模块构成
    def __init__(self, input_nc=6, ndf=64):
        super(NLayerDiscriminator, self).__init__()

        self.layers = nn.Sequential( # 6 256 256
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),# 64 128 128
            nn.LeakyReLU(0.2),

            ConvBlock(ndf, ndf * 2), # 128 64 64
            ConvBlock(ndf * 2, ndf * 4),# 256 32 32
            ConvBlock(ndf * 4, ndf * 8, stride=1), # 512 31 31

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),# 1 30 30
            nn.Sigmoid() # 输出的值映射到0-1之间
        )

    def forward(self, input):
        return self.layers(input)

class ConvBlock(nn.Module):
    # conv => batch norm => LeakyReLU
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential( # 把特征图的大小缩小一般
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),  # Conv2D
            nn.BatchNorm2d(out_dim),  # BatchNorm2D
            nn.LeakyReLU(0.2)  # LeakyReLU, leaky=0.2
        )

    def forward(self, x):
        x = self.layers(x)
        return x
