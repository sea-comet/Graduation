"""A small Unet-like zoo"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from src.models.layers import ConvBnRelu, UBlock, conv1x1, UBlockCbam, CBAM


class Unet(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    # 这里inplanes = 4, 就是4个模态的channel, num_classes = 3, width传入的是48, deep_supervision 好像是True
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]  # [48,96,192,384]
        print("features: ", features)

        self.deep_supervision = deep_supervision  # 这里设置成了True
        # UBlock 在layers.py里面 它跟图上有些不同就是因为它中间的都除了2 ！！！！！！试试把除以2去掉！！
        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)  # 4，24，48
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)  # 48，48，96
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)  # 96，96，192
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)  # 192，192，384
        #  注意bottom这里有dilation: (2,2)
        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)  # 384，384，384

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)  # 384*2,192

        self.downsample = nn.MaxPool3d(2, 2)  # 下采样 应该是除了channel其它都变了

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)  # 384,192,96
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)  # 192,96,48
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)  # 96,48,24

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)  # 上采样 变成了2倍

        self.outconv = conv1x1(features[0] // 2, num_classes)  # 最后一个输出class的卷积！！# 24，3

        if self.deep_supervision:  # 深度监督
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),  # 384，3 # upsample 应该是脑子z,y,x分别扩大了scale_factor倍，channel没变
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),  # 192，3
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),  # 96，3
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),  # 48，3
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # MD,何凯明还有这玩意儿！！
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)  # weight初始化为1
                nn.init.constant_(m.bias, 0)  # bias初始化为0

    def forward(self, x):

        print("x shape: ",x.shape)
        down1 = self.encoder1(x)
        print("encoder 1 shape: ",down1.shape)
        down2 = self.downsample(down1)
        print("encoder 1 down shape: ", down2.shape)
        down2 = self.encoder2(down2)
        print("encoder 2 shape: ", down2.shape)
        down3 = self.downsample(down2)
        print("encoder 2 down shape: ", down3.shape)
        down3 = self.encoder3(down3)
        print("encoder 3 shape: ", down3.shape)
        down4 = self.downsample(down3)
        print("encoder 3 down shape: ", down4.shape)
        down4 = self.encoder4(down4)
        print("encoder 4 down shape: ", down4.shape)

        bottom = self.bottom(down4)
        print("bottom shape: ", bottom.shape)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))  # 384*2 --> 192
        print("bottom_2 shape: ", bottom_2.shape)

        # Decoder

        up3 = self.upsample(bottom_2)
        print("before decoder 3 upsample shape: ", up3.shape)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        print("decoder 3 shape: ", up3.shape)
        up2 = self.upsample(up3)
        print("before decoder 2 upsample shape: ", up2.shape)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        print("decoder 2 shape: ", up2.shape)
        up1 = self.upsample(up2)
        print("before decoder 1 upsample shape:: ", up1.shape)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))
        print("decoder 1 shape: ", up1.shape)

        out = self.outconv(up1)
        print("output shape: ", out.shape)

        if self.deep_supervision:  # 如果有深度监督，就返回两个东西
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out


class EquiUnet(Unet):  # 继承的Unet，所以forward函数和Unet一样
    """Almost the most basic U-net: all Block have the same size if they are at the same level.
    #  每一层都有相同大小, 这个好像是原Unet,论文中用的可能是 Unet
    """
    name = "EquiUnet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]  # [48,96,192,384]
        print("model features: ", features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0], features[0], norm_layer, dropout=dropout)  # 4,48,48
        self.encoder2 = UBlock(features[0], features[1], features[1], norm_layer, dropout=dropout)  # 48,96,96
        self.encoder3 = UBlock(features[1], features[2], features[2], norm_layer, dropout=dropout)  # 96,192,192
        self.encoder4 = UBlock(features[2], features[3], features[3], norm_layer, dropout=dropout)  # 192,384,384

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)  # 384,384,384

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)  # 384*2-->192

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)  # 192*2,192,96
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)  # 96*2,96,48
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)  # 48*2,48,48

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)  # 48,3

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),  # 384,3
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),  # 192,3
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),  # 96,3
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),  # 48,3
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()


class Att_EquiUnet(Unet):  # 继承的Unet，所以forward函数和Unet一样
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print("model features: ", features)

        self.deep_supervision = deep_supervision

        # 原本的Equi版本
        self.encoder1 = UBlock(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)
        # 后面用的都是UBlock了
        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()


# 我改的这玩意儿！！！
class Atten_Unet(Unet):  # 继承的Unet，所以forward函数和Unet一样
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print("model features: ", features)

        self.deep_supervision = deep_supervision

        # UBlockCbam 在layers.py 里面有！！这儿改了！！改成不是Equi的版本了！！
        self.encoder1 = UBlockCbam(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)  # 4，24，48
        self.encoder2 = UBlockCbam(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)  # 48，48，96
        self.encoder3 = UBlockCbam(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)  # 96，96，192
        self.encoder4 = UBlockCbam(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)  # 192，192，384

        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)
        # 后面用的都是UBlock了
        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()
