"""functions to correctly pad or crop non uniform sized MRI (before batching in the dataloader).
"""
import random

import numpy as np

# padding or cropping crop, pad
def pad_or_crop_image(image, seg=None, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    # get_crop_slice down ,crop（128，128，128）
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]: # 需要pad,to_pad[1]为left,to_pad[2]为right
            padlist.append((to_pad[1], to_pad[2]))
        else: # 不需要pad
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg       # seg crop to the same size
    return image


def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right) # dim-right = left+target_size，slice could return cropping index
    elif dim <= target_size:
        return slice(0, dim)


def normalize(image): # tick
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image



def irm_min_max_preprocess(image, low_perc=1, high_perc=99): # perc是percentage. 用最大值和最小值来normalize
    """
    Remove outliers voxels first
    Then use min-max scale.
    """

    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc]) # 找到某某百分位的数，如50就是中位数
    image = np.clip(image, low, high) # 用来切除（%0到1%，以及99%到100%）的outlier
    image = normalize(image) # normalize函数在上面
    return image


def zscore_normalise(img: np.ndarray) -> np.ndarray:  # 用均值和标准差来normalize
    slices = (img != 0)
    img[slices] = (img[slices] - np.mean(img[slices])) / np.std(img[slices])
    return img



