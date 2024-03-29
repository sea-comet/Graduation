import os
import pathlib
import pprint

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff
from torch.cuda.amp import autocast

from src.dataset.batch_utils import pad_batch1_to_compatible_size

#  The names of all validation metrics
SENS = "sens"
SPEC = "spec"
HAUSSDORF = "haussdorf"
DICE = "dice"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]


def save_args(args):  # save srgs as yaml format
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    del config['save_folder']
    del config['seg_folder']
    pprint.pprint(config)
    config_file = args.save_folder / (args.exp_name + ".yaml")
    with config_file.open("w") as file:
        yaml.dump(config, file)


def save_checkpoint(state: dict, save_folder: pathlib.Path):  # save model weights
    """Save Training state."""
    best_filename = f'{str(save_folder)}/model_best.pth.tar'
    torch.save(state, best_filename)


class AverageMeter(object):  # update, reset, calculate
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):  # 展示输出
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# TODO remove dependency to args
def reload_ckpt(args, model, optimizer, scheduler):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))


def reload_ckpt_bis(ckpt, model, optimizer=None):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        try:
            checkpoint = torch.load(ckpt)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{ckpt}' (epoch {start_epoch})")
            return start_epoch
        except RuntimeError:
            # TO account for checkpoint from Alex nets
            print("Loading model Alex style")
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")


def count_parameters(model):  # 计算model参数数量的函数
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_metrics(preds, targets, patient):   # generate_segmentations里被调用
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
        )

        if np.sum(targets[i]) == 0:  # targets[i]取ET,TC,WT 中的某一个map, seg.nii里恰好没有这一类肿瘤
            print(f"{label} not present for {patient}")
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            # 如果真实脑子某一个channel啥也没有，pred必须每个voxel为空，dice才等于1，但凡有一个1,dice都等于0
            # TN: true negative, FP: false positive
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i]))) # logical_and 和 logical_not
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp) # SPECIFICITY
            haussdorf_dist = np.nan

        else: # 这一类肿瘤不是空
            preds_coords = np.argwhere(preds[i]) # 返回是1的点的坐标，每个点的坐标放一起，element-wise
            targets_coords = np.argwhere(targets[i])
            # directed_hausdorff 函数从 scipy.spatial.distance引入
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]  # 算 Hausdorff

            tp = np.sum(l_and(preds[i], targets[i]))  # true positive
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))  # true negative
            fp = np.sum(l_and(preds[i], l_not(targets[i])))  # false positive
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))  # false negative

            sens = tp / (tp + fn)  # sensitivity灵敏度
            spec = tn / (tn + fp)  # specificity特异性

            dice = 2 * tp / (2 * tp + fp + fn)   # dice score 计算公式

        # metrics 是个字典，对于每个patient,每个ET,TC,WT 有一个
        metrics[HAUSSDORF] = haussdorf_dist
        metrics[DICE] = dice
        metrics[SENS] = sens
        metrics[SPEC] = spec
        # 打印每个病人metrics
        # print("metrics: ")
        # pp.pprint(metrics)
        metrics_list.append(metrics) # metrics_list 是一个含有3个metrics字典的list, 对ET,TC,WT各一个

    return metrics_list



def save_metrics(epoch, metrics, writer, current_epoch, save_folder=None):
    metrics = list(zip(*metrics)) # 这是分别算好的ET,TC,WT的dice, every patient 3个metric
    # print("save_metrics里面的metrics: ", metrics)

    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    # print(metrics)
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1) # 设置y轴数值的范围
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    print(f"\nEpoch {current_epoch} :{'val' + 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    with open(f"{save_folder}/val.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + 'Val :'}",   # 把结果存在TXT文件里
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()], file=f)
    for key, value in metrics.items(): # 取出字典中的key和value
        tag = f"val/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


def generate_segmentations(data_loader, model, writer, args):
    metrics_list = []
    for i, batch in enumerate(data_loader):
        # 计算数据loading 时间
        inputs = batch["image"]
        patient_id = batch["patient_id"][0]
        ref_path = batch["seg_path"][0] # 用来对比的ground truth
        crops_idx = batch["crop_indexes"]
        inputs, pads = pad_batch1_to_compatible_size(inputs)
        # pad_batch1_to_compatible_size --> batch_utils.py
        inputs = inputs.cuda() # pad好了的inputs
        with autocast():
            with torch.no_grad():
                pre_segs = model(inputs)
                pre_segs = torch.sigmoid(pre_segs) # 0/1
        # delete padding
        maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu() # 去掉padding后的最小边框肿瘤
        # segs是整个脑子的大小
        segs = torch.zeros((1, 3, 155, 240, 240))
        # segs与预测的pre_segs（最小肿瘤框）合体，组合在一起
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
        segs = segs[0].numpy() > 0.5
        # 现在seg的维度是（3，155，240，240）是对于ET,TC,WT 分3个channel的map图！！

        et = segs[0]    # seg的维度为3的那3个图好像是：ET, NET+ET, NET+ET+edema，有重叠的内，中，外3层
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape) # 形状: (155,240,240),就是一个脑子, 要把ET,NET,EDEMA 放在一个脑子中呈现
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        # labelmap 合体, 这是自己预测的脑子各部分图！！一个脑子有1，2，4 这几种label
        labelmap = sitk.GetImageFromArray(labelmap)

        # ground truth 图
        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img) # ground truth 脑子的array，带有1，2，4 标签的
        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 4  # ET
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)  # TC
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)  # WT
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt]) # 多了一维，现在是（3，155，240，240）是对于ET,TC,WT的map, 3为channel

        # 计算model预测出的脑子和真实脑子的各种 metrics，get a list,是ET,TC,WT分别的metrics的字典,[HAUSSDORF, DICE, SENS, SPEC]
        patient_metric_list = calculate_metrics(segs, refmap, patient_id) # calculate_metrics up
        metrics_list.append(patient_metric_list)
        # labelmap --> my prediction， nii.gz 文件存在了seg文件夹下
        labelmap.CopyInformation(ref_seg_img)
        print(f"Writing new segmentation as {args.seg_folder}/{patient_id}.nii.gz\n")
        sitk.WriteImage(labelmap, f"{args.seg_folder}/{patient_id}.nii.gz")

    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes") # METRICS = [HAUSSDORF, DICE, SENS, SPEC]
    overlap_figure = overlap[0].get_figure()
    writer.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    writer.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict() # mean

    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer.add_scalar(f"benchmark_{metric}/{label}", score) # e.g. DICE/ET: 0.89
    df.to_csv((args.save_folder / 'results.csv'), index=False)




