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


def save_args(args):  # 保存args设置，保存成yaml格式
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    del config['save_folder']
    del config['seg_folder']
    pprint.pprint(config)
    config_file = args.save_folder / (args.exp_name + ".yaml")
    with config_file.open("w") as file:
        yaml.dump(config, file)


def save_checkpoint(state: dict, save_folder: pathlib.Path):  # 已看
    """Save Training state."""
    best_filename = f'{str(save_folder)}/model_best.pth.tar'
    torch.save(state, best_filename)


class AverageMeter(object):  # 更新数据，重置数据, 计算平均值
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


def calculate_metrics(preds, targets, patient):   # 在generate_segmentations函数里被调用了
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X 确定？？？它好像应该是没有前面那个1 ？？
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

        if np.sum(targets[i]) == 0:  # targets[i]取ET,TC,WT 中的某一个map, 真实脑子里恰好没有这一类肿瘤，啥也没有！！
            print(f"{label} not present for {patient}")
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            # 如果真实脑子某一个channel啥也没有，pred也得啥也没有dice才等于1，但凡有一个1,dice都等于0
            # tn: true negative, fp: false positive
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i]))) # logical_and 和 logical_not
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp)
            haussdorf_dist = np.nan

        else: # 这一类肿瘤不是没一丁点
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

            dice = 2 * tp / (2 * tp + fp + fn)   # 记住！！！这里是dice的计算公式！！！

        # metrics 是个字典，每个patient,每个ET,TC,WT的class有一个
        metrics[HAUSSDORF] = haussdorf_dist
        metrics[DICE] = dice
        metrics[SENS] = sens
        metrics[SPEC] = spec
        # 注意这里有打印每个病人metrics的地方！！！！！！！！！
        # print("metrics: ")
        # pp.pprint(metrics)
        metrics_list.append(metrics) # metrics_list 是一个含有3个metrics字典的list, 对ET,TC,WT各一个

    return metrics_list



def save_metrics(epoch, metrics, writer, current_epoch, teacher=False, save_folder=None): # teacher 是False!!
    metrics = list(zip(*metrics)) # 这是分别算好的ET,TC,WT的dice！！是每个人的3个metric
    # print("save_metrics里面的metrics: ", metrics)
    # TODO check if doing it directly to numpy work
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
    print(f"\nEpoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",   # 把结果存在了TXT文件里！！
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()], file=f)
    for key, value in metrics.items(): # 取出字典中的key和value
        tag = f"val{'_teacher' if teacher else ''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


def generate_segmentations(data_loader, model, writer, args):
    metrics_list = []
    for i, batch in enumerate(data_loader):
        # measure data loading time
        inputs = batch["image"]
        patient_id = batch["patient_id"][0]
        ref_path = batch["seg_path"][0] # 用来对比的ground truth
        crops_idx = batch["crop_indexes"]
        inputs, pads = pad_batch1_to_compatible_size(inputs) # 把脑子每个维度填充到16倍数大小，pads是三维元组，
        # pad_batch1_to_compatible_size 函数在dataset下batch_utils.py里面
        inputs = inputs.cuda() # pad好了的inputs
        with autocast():
            with torch.no_grad():
                if model.deep_supervision: # 有深度监督
                    pre_segs, _ = model(inputs)
                else:                      # 无深度监督
                    pre_segs = model(inputs)
                pre_segs = torch.sigmoid(pre_segs) # 把概率映射到0~1之间
        # remove pads # 去掉padding !!!
        maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu() # 去掉padding后的最小边框脑子
        # segs其实是整个脑子的大小
        segs = torch.zeros((1, 3, 155, 240, 240))
        # segs与预测的pre_segs（最小肿瘤框）合体
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
        segs = segs[0].numpy() > 0.5   # 这是预测的含完整3个label 1,2,4 的label图,只取出了一个脑子，
        # 现在seg的维度是（3，155，240，240）是对于ET,TC,WT 分3个channel的map图！！

        et = segs[0] # 注意这里！！！seg的维度为3的那3个图好像分别是：ET, NET+ET, NET+ET+edema，就是有重叠的内，中，外3层
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape) # 形状: (155,240,240),就是一个脑子！！要把ET,NET,EDEMA 放在一个脑子中呈现！
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        # labelmap 合体成功！！这是自己预测的脑子各部分图！！一个脑子有1，2，4 这几种label
        labelmap = sitk.GetImageFromArray(labelmap)

        # ground truth 图
        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img) # ground truth 脑子的array，带有1，2，4 标签的
        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 4  # ET
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)  # TC
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)  # WT
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt]) # 多了一维，现在是（3，155，240，240）是对于ET,TC,WT的map

        # 计算model预测出的脑子和真实脑子的各种 metrics，得到一个list,是ET,TC,WT分别的metrics的字典, 包含[HAUSSDORF, DICE, SENS, SPEC]
        patient_metric_list = calculate_metrics(segs, refmap, patient_id) # calculate_metrics函数在上面！！
        metrics_list.append(patient_metric_list)
        # labelmap 里是自己预测的图像，再加进真实图像，大概是要做对比？？？？nii.gz 文件存在了seg文件夹下面！！
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
    summary = grouped_df.mean().to_dict() # 求的均值！！
    # 这里有点儿没看懂，再看看！！
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer.add_scalar(f"benchmark_{metric}/{label}", score) # 例如：DICE/ET: 0.89
    df.to_csv((args.save_folder / 'results.csv'), index=False)


def update_teacher_parameters(model, teacher_model, global_step, alpha=0.99 / 0.999): # ？？没有看到在哪里用到了
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    # print("teacher updated!")



