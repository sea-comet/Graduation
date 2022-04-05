from itertools import combinations, product
# combinations 就是排列组合中的组合的意思
import torch

trs = list(combinations(range(2, 5), 2)) + [None]  # [(2,3),(2,4),(3,4),None]   在2，3，4 这3个数中选2个的所有组合,再加个None
flips = list(range(2, 5)) + [None]  # [2,3,4,None]
rots = list(range(1, 4)) + [None]  # [1,2,3,None]
transform_list = list(product(flips, rots))  # 在flips中任选一个，再在rots中任选一个 的所有组合


def simple_tta(x):  # x是一个batch的inputs
    """Perform all transpose/mirror transform possible only once.

    Sample one of the potential transform and return the transformed image and a lambda function to revert the transform
    Random seed should be set before calling this function

    # 任选一种transform，返回transform了的image和一个lambda函数，来还原这个transform

    """
    out = [[x, lambda z: z]]
    for flip, rot in transform_list[:-1]:
        if flip and rot:  # 就是两个都不是None
            trf_img = torch.rot90(x.flip(flip), rot, dims=(3, 4))  # 这些为什么还原reverse的时候都跟之前不一样呢？？奇怪！！
            back_trf = revert_tta_factory(flip, -rot)              # 哦！！我知道了！！因为放入model训练的时候也transform过一次
        elif flip:
            trf_img = x.flip(flip)
            back_trf = revert_tta_factory(flip, None)
        elif rot:
            trf_img = torch.rot90(x, rot, dims=(3, 4))
            back_trf = revert_tta_factory(None, -rot)
        else:
            raise
        out.append([trf_img, back_trf])  # back_trf是个lambda函数！
    return out


def apply_simple_tta(model, x, average=True):  # 这个是inference.py里面用到的函数！！x 是inputs,就是一个batch
    todos = simple_tta(x)
    out = []
    for im, revert in todos:  # revert是个lambda函数
        if model.deep_supervision: # 有深度监督,return两个值
            out.append(revert(model(im)[0]).sigmoid_().cpu()) # 放入model训练了再还原一下！！
            # 记住放入model训练的时候也transform过，所以revert的时候都不是正好跟之前相反的
        else:  # 无深度监督
            out.append(revert(model(im)).sigmoid_().cpu())
    if not average:
        return out  # 这个没平均
    return torch.stack(out).mean(dim=0)  # 这个平均了


def revert_tta_factory(flip, rot):
    if flip and rot:
        return lambda x: torch.rot90(x.flip(flip), rot, dims=(3, 4))
    elif flip:
        return lambda x: x.flip(flip)
    elif rot:
        return lambda x: torch.rot90(x, rot, dims=(3, 4))
    else:
        raise
