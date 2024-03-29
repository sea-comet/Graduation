from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


def get_norm_layer(norm_type="group"):  # norm_type:  group normalization Kaiming He
    if "group" in norm_type:
        try:
            grp_nb = int(norm_type.replace("group", ""))
            return lambda planes: default_norm_layer(planes, groups=grp_nb)  # default_norm_layer --> up
        except ValueError as e:
            print(e)
            print('using default group number')
            return default_norm_layer
    elif norm_type == "none":
        return None


def default_norm_layer(planes, groups=16):
    groups_ = min(groups, planes)
    if planes % groups_ > 0:
        divisor = 16
        while planes % divisor > 0:
            divisor /= 2
        groups_ = int(planes // divisor)
    return nn.GroupNorm(groups_, planes)



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):  # kernel_size 是3的卷积
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)  # group


def conv1x1(in_planes, out_planes, stride=1, bias=True): # 1*1 卷积
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ConvBnRelu(nn.Sequential):

    def __init__(self, inplanes, planes, norm_layer=None, dilation=1, dropout=0):
        if norm_layer is not None:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [           # conv3x3 在上面
                        ('conv', conv3x3(inplanes, planes, dilation=dilation)),
                        ('bn', norm_layer(planes)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )
        else:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation, bias=True)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )


class UBlock(nn.Sequential):
    """
       Unet mainstream downblock. # Unet encoder 基础block
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock, self).__init__(  # init 传进去的是一个字典
            OrderedDict(
                [                   # ConvBnRelu 在上面
                    ('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)),
                    ('ConvBnRelu2',ConvBnRelu(midplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout)),
                ])
        )


class UBlockCbam(nn.Sequential):  # 在AttUnet中用的Block, 跟UBlock（2个卷积层）相比多了一个CBAM
    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlockCbam, self).__init__(
            OrderedDict(
                [
                    ('UBlock', UBlock(inplanes, midplanes, outplanes, norm_layer, dilation=dilation, dropout=dropout)),
                    ('CBAM', CBAM(outplanes, norm_layer=norm_layer)),
                ])
        )



class BasicConv(nn.Module):  # 传进去的in_planes=2，out_planes=1
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=None):
        super(BasicConv, self).__init__()
        bias = False
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


'''
    CBAM module
    Including Spatial_Attention & Channel_Attention module.
    
'''

class CBAM(nn.Module):  # 一个ChannelGate加一个SpatialGate, gate_channels的接收维度就是UBlock的输出维度
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, norm_layer=None):
        super(CBAM, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)  # pool_types  'avg' or 'max'
        self.SpatialGate = SpatialGate(norm_layer)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(    # MLP: Multilayer Perceptron
            Flatten(),    # 先展平
            nn.Linear(gate_channels, gate_channels // reduction_ratio),  # 每个UBlock的输出维度 --> 除以16
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)  # 再还原，除以16了的UBlock的输出维度 --> UBlock的输出维度
        )
        self.pool_types = pool_types

    def forward(self, x):   # avg_pool,mlp, max_pool 共用 mlp
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, norm_layer=None):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()  # channel = 2 --> channel = 1 !
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, norm_layer=norm_layer)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)     # broadcasting
        return x * scale  


if __name__ == '__main__':
    print(UBlock(4, 4))
