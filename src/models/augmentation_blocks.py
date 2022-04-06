"""The strange way used to perform data augmentation during the Brats 2020 challenge...

Be aware, any batch size above 1 could fail miserably (in an unpredicted way).
"""

from random import randint, random, sample, uniform

import torch
from torch import nn


class DataAugmenter(nn.Module):  # 按batch进行数据增广
    """Performs random flip and rotation batch wise, and reverse it if needed.
    Works"""

    # 实际训练的时候p=0.8, drop_channel是True, 其它是False !!!
    def __init__(self, p=0.5, noise_only=False, channel_shuffling=False, drop_channnel=False):
        super(DataAugmenter, self).__init__()
        self.p = p
        self.transpose = []
        self.flip = []
        self.toggle = False
        self.noise_only = noise_only
        self.channel_shuffling = channel_shuffling
        self.drop_channel = drop_channnel

    def forward(self, x):
        with torch.no_grad():
            if random() < self.p:  # 这是干什么？？？p是0.8   # 说明有0.8的概率会加噪声
                x = x * uniform(0.8, 1.2)  # 在0.9~1.1 之间随便选一个数
                std_per_channel = torch.stack(  # 对４个模态分别进行标准化？？
                    list(torch.std(x[:, i][x[:, i] > 0]) for i in range(x.size(1)))
                )
                # 加噪声
                noise = torch.stack([torch.normal(0, std * 0.1, size=x[0, 0].shape) for std in std_per_channel]
                                    ).to(x.device)
                x = x + noise
                if random() < 0.1 and self.channel_shuffling:
                    new_channel_order = sample(range(x.size(1)), x.size(1))
                    x = x[:, new_channel_order]
                    print("channel shuffling")
                if random() < 0.1 and self.drop_channel:  # 说明有0.2的概率会扔掉channel
                    x[:, sample(range(x.size(1)), 1)] = 0
                    print("channel Dropping")
                if self.noise_only: # 咱能不能直接用noise, 不用channel dropping?????!!!!记得改改！！！这个在train.py 里面传参True就可以
                    return x
                # transpose 和 flip ————> 感觉这里比例太大了，有0.8，可以改小一点！！记得改！！
                self.transpose = sample(range(2, x.dim()), 2) # 在z,y,x 这3个维度任选2个
                self.flip = randint(2, x.dim() - 1) # 在z,y,x 这3个维度任选1个
                self.toggle = not self.toggle    # 这下toggle设置成True了
                new_x = x.transpose(*self.transpose).flip(self.flip) # 任选2个维度transpose,又任选一个维度flip
                return new_x
            else: # p>0.8 的时候-->也就是有0.2的概率啥augmentation都不做
                return x

    def reverse(self, x): # 好像又给transpose和flip回去了--> 因为算loss对比的时候需要把它再转置旋转回去！！
        if self.toggle:
            self.toggle = not self.toggle # 如果toggle是True,就设置成False,现在是False
            if isinstance(x, list):  # case of deep supervision # 这是啥意思？？？用于深度监督？？？？
                seg, deeps = x # 这啥玩意儿？？拆开的是啥？？？see！！
                reversed_seg = seg.flip(self.flip).transpose(*self.transpose)
                reversed_deep = [deep.flip(self.flip).transpose(*self.transpose) for deep in deeps]
                return reversed_seg, reversed_deep
            else:
                return x.flip(self.flip).transpose(*self.transpose)
        else:
            return x
