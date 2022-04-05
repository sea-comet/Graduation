import torch
import torch.nn as nn


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]  # 0是ET, 1是TC, 2是WT吗？
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False): # 算出来是一个（0，1）之间的数
        smooth = 1.      # 论文里的那个smooth系数,分子分母同时加上的那个
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs) # model 训练完以后得出的就是概率，用sigmoid可以转化为（0，1）之前的数

        if metric_mode:   # 从这儿开始都是给 metric mode，前面不是 metric mode
            inputs = inputs > 0.5  # 相当于大于0.5d的都转化成了1
            if targets.sum() == 0:   # targets里面没有某一部分肿瘤：ET, TC 或者WT
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:   # inputs 里面也没有某一部分肿瘤：ET, TC 或者WT，直接返回1，否则返回0
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)  # 计算inputs 和targets同时为1的数量
        # metric 模式，衡量标准！！！
        if metric_mode:   # 2*(inputs 和targets都=1的数量) / (inputs=1的数量 + targets=1的数量)
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:  # Loss 模式！！后面还要用1减 公式在论文第5页，有平方, 为什么有平方？就很奇怪
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        flag = 0       # 我新加的，打印出inputs和targets的shape
        if flag == 0:
            print("inputs shape: ",inputs.shape)
            print("target shape: ",target.shape)
        flag += 1
        for i in range(target.size(1)):   # i 应该为3个肿瘤区域的map, 分别为ET, TC, WT
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)  # binary_dice 函数在上面
            # 此时metric_mode 是False!!

        final_dice = dice / target.size(1) # 总dice就为ET,TC,WT dice 的平均
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):  # 把每个样本的3个类别ET,TC,WT的dice 分别计算，放在列表里 如: [[0.2, 0.8, 0.6],[0.9,0.3,0.7],...]
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))  # metric_mode 是True
            dices.append(dice)
        return dices
