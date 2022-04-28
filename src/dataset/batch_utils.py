import random

import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


def determinist_collate(batch):
    batch = pad_batch_to_max_shape(batch)
    return default_collate(batch)

# 在train.py中给validation使用 在validation中将含肿瘤的最大部分切出
def pad_batch_to_max_shape(batch):
    shapes = (sample['label'].shape for sample in batch)
    _, z_sizes, y_sizes, x_sizes = list(zip(*shapes))
    maxs = [int(max(z_sizes)), int(max(y_sizes)), int(max(x_sizes))]
    for i, max_ in enumerate(maxs):
        max_stride = 16             # 注意16
        if max_ % max_stride != 0:
            # Make it divisible by 16
            maxs[i] = ((max_ // max_stride) + 1) * max_stride
    zmax, ymax, xmax = maxs
    for elem in batch:
        exple = elem['label']
        zpad, ypad, xpad = zmax - exple.shape[1], ymax - exple.shape[2], xmax - exple.shape[3]
        assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
        # free data augmentation
        left_zpad, left_ypad, left_xpad = [random.randint(0, pad) for pad in (zpad, ypad, xpad)]
        right_zpad, right_ypad, right_xpad = [pad - left_pad for pad, left_pad in
                                              zip((zpad, ypad, xpad), (left_zpad, left_ypad, left_xpad))]
        pads = (left_xpad, right_xpad, left_ypad, right_ypad, left_zpad, right_zpad)
        elem['image'], elem['label'] = F.pad(elem['image'], pads), F.pad(elem['label'], pads)
    return batch

 # 用于test.py 中的使用
def pad_batch1_to_compatible_size(batch):
    print("pad过的batch shape: ", batch.shape)
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx): # 循环为z,y,x维度的大小
        max_stride = 16  # 让z,y,x 分别都为16的倍数
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
    zmax, ymax, xmax = zyx # 新的z,y,x
    zpad, ypad, xpad = zmax - batch.size(2), ymax - batch.size(3), xmax - batch.size(4) # z,y,x分别需要填充的大小！！
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads) # 填充过的脑子图！每个batch只有一个
    return batch, (zpad, ypad, xpad)
