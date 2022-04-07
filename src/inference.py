import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace

import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
import yaml
from torch.cuda.amp import autocast

from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from src.models import get_norm_layer
from src.tta import apply_simple_tta
from src.utils import reload_ckpt_bis

parser = argparse.ArgumentParser(description='Brats validation and testing dataset inference')
parser.add_argument('--config', default='', type=str, metavar='PATH', # path to 装有config的训练好的model和参数啥的，yaml文件
                    help='path(s) to the trained models config yaml you want to use', nargs="+")
parser.add_argument('--devices', required=True, type=str,  # 记得要传个参数 0
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--on', default="val", choices=["val","train","test"])  # 默认是validation
parser.add_argument('--tta', action="store_true")
parser.add_argument('--seed', default=16111990)


def main(args):
    # setup
    random.seed(args.seed)
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeWarning("This will not be able to run on CPU only")
    print(f"Working with {ngpus} GPUs")
    # print("设置的args: ", args.config)

    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    save_folder = pathlib.Path(f"./preds/{current_experiment_time}") # 记住在preds文件夹下面!!
    save_folder.mkdir(parents=True, exist_ok=True)

    with (save_folder / 'args.txt').open('w') as f:
        print(vars(args), file=f)         # 把参数设置写进文件

    args_list = []
    for config in args.config:
        config_file = pathlib.Path(config).resolve()  # 搞成绝对路径
        print("config file: ",config_file)
        ckpt = config_file.with_name("model_best.pth.tar")  # 换了一个文件名
        with config_file.open("r") as file:
            old_args = yaml.safe_load(file)  # 打开yaml文件
            old_args = SimpleNamespace(**old_args, ckpt=ckpt)  # 把args从yaml文件load进来
            # set default normalisation
            if not hasattr(old_args, "normalisation"):    # 如果training 的参数里没有指定Normalization就用默认minmax
                old_args.normalisation = "minmax"
        print("old args: ",old_args)
        args_list.append(old_args)

    if args.on == "test":
        args.pred_folder = save_folder / f"test_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    elif args.on == "val":
        args.pred_folder = save_folder / f"validation_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    else:
        args.pred_folder = save_folder / f"training_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)

    # Create model

    models_list = []
    normalisations_list = []
    for model_args in args_list:  # model_args 是从yaml文件里load进来的
        # print("Model used: ",model_args.arch)   # 用了哪一个model
        model_maker = getattr(models, model_args.arch)

        model = model_maker(
            4, 3,
            width=model_args.width, deep_supervision=model_args.deep_sup,
            norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
        print(f"Creating {model_args.arch}")

        reload_ckpt_bis(str(model_args.ckpt), model)  # 把 best ckpt load进来
        models_list.append(model)
        normalisations_list.append(model_args.normalisation)
        print("reload best weights")
        # print("model info: ", model)

    dataset_minmax = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="minmax")

    dataset_zscore = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="zscore")

    loader_minmax = torch.utils.data.DataLoader(
        dataset_minmax, batch_size=1, num_workers=2)

    loader_zscore = torch.utils.data.DataLoader(
        dataset_zscore, batch_size=1, num_workers=2)

    print("Val dataset number of batch: ", len(loader_minmax)) # 几个batch, 也就是几个脑子

    # 这个generate_segmentations函数是下面的这个，不是utils.py里面的
    generate_segmentations((loader_minmax, loader_zscore), models_list, normalisations_list, args)


def generate_segmentations(data_loaders, models, normalisations, args):
    # TODO: try reuse the function used for train... # 试着把这个和utils.py里面的函数合体
    # 它可以这样是因为dataloader没有shuffle,还设置了random seed
    for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
        patient_id = batch_minmax["patient_id"][0]
        ref_img_path = batch_minmax["seg_path"][0]
        crops_idx_minmax = batch_minmax["crop_indexes"]
        crops_idx_zscore = batch_zscore["crop_indexes"]
        inputs_minmax = batch_minmax["image"]
        inputs_zscore = batch_zscore["image"]
        inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
        inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
        model_preds = []
        last_norm = None
        for model, normalisation in zip(models, normalisations):
            if normalisation == last_norm:
                pass
            elif normalisation == "minmax":
                inputs = inputs_minmax.cuda()
                pads = pads_minmax
                crops_idx = crops_idx_minmax
            elif normalisation == "zscore":
                inputs = inputs_zscore.cuda()
                pads = pads_zscore
                crops_idx = crops_idx_zscore
            model.cuda()  # go to gpu
            with autocast():
                with torch.no_grad():
                    if args.tta:  # 如果需要进行test time augmentation:
                        # apply_simple_tta 函数在tta.py 文件里可以找到
                        pre_segs = apply_simple_tta(model, inputs, True) # 看tta.py文件！！
                        # model_preds.append(pre_segs)  # 这儿好像作者写错了，有bug
                    else: # deep supervision 时不需要tta
                        if model.deep_supervision:
                            pre_segs, _ = model(inputs)
                        else:
                            pre_segs = model(inputs)
                        pre_segs = pre_segs.sigmoid_().cpu()
                    # remove pads 移除padding
                    maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - \
                                       pads[2]
                    pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
                    # print("pre_segs size", pre_segs.shape)
                    segs = torch.zeros((1, 3, 155, 240, 240))
                    segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
                    # print("segs size", segs.shape)

                    model_preds.append(segs)
            model.cpu()  # free for the next one
        pre_segs = torch.stack(model_preds).mean(dim=0)

        segs = pre_segs[0].numpy() > 0.5

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)
        ref_img = sitk.ReadImage(ref_img_path)
        labelmap.CopyInformation(ref_img)
        print(f"Writing new segmentation as {str(args.pred_folder)}/{patient_id}.nii.gz")   # 还是一个合体的
        sitk.WriteImage(labelmap, f"{str(args.pred_folder)}/{patient_id}.nii.gz")


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
