from random import randint, random, sample, uniform
import torch
from torch import nn


class DataAugmenter(nn.Module):  # 按batch进行数据增广
    """Performs random flip and rotation batch wise, and reverse it if needed.
    Works"""

    # p=0.8, drop_channel是True!！
    def __init__(self, p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True):
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
            if random() < self.p:    # 0.8的概率
                x = x * uniform(0.9, 1.1)  # rescale
                std_per_channel = torch.stack(   # 4个modality, respectively
                    list(torch.std(x[:, i][x[:, i] > 0]) for i in range(x.size(1)))
                )
                # 加噪声
                noise = torch.stack([torch.normal(0, std * 0.1, size=x[0, 0].shape) for std in std_per_channel]
                                    ).to(x.device)
                x = x + noise
                # if random() < 0.2 and self.channel_shuffling:
                #     new_channel_order = sample(range(x.size(1)), x.size(1))
                #     x = x[:, new_channel_order]
                #     print("channel shuffling")
                if random() < 0.2 and self.drop_channel:  # 0.2 probability --> drop channel
                    x[:, sample(range(x.size(1)), 1)] = 0
                    print("channel Dropping")
                if self.noise_only:
                    return x
                # transpose 和 flip ————> 概率0.8，可再试调参 TODO
                self.transpose = sample(range(2, x.dim()), 2) # 在z,y,x 3个维度任选2个
                self.flip = randint(2, x.dim() - 1) # 在z,y,x 这3个维度任选1个
                self.toggle = not self.toggle
                new_x = x.transpose(*self.transpose).flip(self.flip) # 任选2个维度transpose,又任选一个维度flip
                return new_x
            else: # 有0.2的概率啥augmentation都不做
                return x

    def reverse(self, x): # 又给transpose和flip回去了--> because算loss对比的时候需要把它再转置旋转回去
        if self.toggle:
            self.toggle = not self.toggle # toggle now False
            return x.flip(self.flip).transpose(*self.transpose)
        else:
            return x
