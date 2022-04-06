import argparse
import os
import pathlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
# from ranger import Ranger
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import determinist_collate  # 重分配
from src.loss import EDiceLoss
from src.models import get_norm_layer, DataAugmenter
from src.utils import save_args, AverageMeter, ProgressMeter, reload_ckpt, save_checkpoint, reload_ckpt_bis, \
    count_parameters, WeightSWA, save_metrics, generate_segmentations

parser = argparse.ArgumentParser(description='Brats Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='Atten_Unet',
                    help='model architecture (default: Unet)')
parser.add_argument('--width', default=48, help='base number of features for Unet (x2 per downsampling)', type=int)
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')  # 这个用在重新启动，200 epoch 后面训练的那150个epoch !!
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')  # dest 就是把获取的值放到哪个变量里的意思
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
# Warning: untested option!!
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint. Warning: untested option')
# parser.add_argument('--devices', required=True, type=str,
#                     help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--debug', action="store_true")
parser.add_argument('--deep_sup', action="store_true")  # 深度监督
parser.add_argument('--no_fp16', action="store_true")
parser.add_argument('--seed', default=16111990, help="seed for train/val split")
parser.add_argument('--warm', default=3, type=int, help="number of warming up epochs")

parser.add_argument('--val', default=3, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")  # 这个应该是交叉验证
parser.add_argument('--norm_layer', default='group')
parser.add_argument('--swa', action="store_true", help="perform stochastic weight averaging at the end of the training")
parser.add_argument('--swa_repeat', type=int, default=5, help="how many warm restarts to perform")
# parser.add_argument('--optim', choices=['adam', 'sgd', 'ranger', 'adamw'], default='ranger')
parser.add_argument('--optim', choices=['adam', 'sgd', 'adamw'], default='adam')
parser.add_argument('--com', help="add a comment to this run!")
parser.add_argument('--dropout', type=float, help="amount of dropout to use", default=0.)
parser.add_argument('--warm_restart', action='store_true',
                    help='use scheduler warm restarts with period of 30')  # 这个可能得用到
parser.add_argument('--full', action='store_true', help='Fit the network on the full training set')  # 用整个训练集训练！

#
# 加一个device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 这是我加的
print("Training device used: ", device)

# setup
ngpus = torch.cuda.device_count()
# if ngpus == 0:
#     raise RuntimeWarning("This will not be able to run on CPU only")

# print(f"Working with {ngpus} GPUs")


#    if args.optim.lower() == "ranger":
#        # No warm up if ranger optimizer
#        args.warm = 0


def main(args):
    """ The main training function.

    Only works for single node (be it single or multi-GPU)

    Parameters
    ----------
    args :
        Parsed arguments
    """

    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    args.exp_name = f"{'debug_' if args.debug else ''}{current_experiment_time}_" \
                    f"_fold{args.fold if not args.full else 'FULL'}" \
                    f"_{args.arch}_{args.width}" \
                    f"_batch{args.batch_size}" \
                    f"_optim{args.optim}" \
                    f"_{args.optim}" \
                    f"_lr{args.lr}-wd{args.weight_decay}_epochs{args.epochs}_deepsup{args.deep_sup}" \
                    f"_{'fp16' if not args.no_fp16 else 'fp32'}" \
                    f"_warm{args.warm}_" \
                    f"_norm{args.norm_layer}{'_swa' + str(args.swa_repeat) if args.swa else ''}" \
                    f"_dropout{args.dropout}" \
                    f"_warm_restart{args.warm_restart}" \
                    f"{'_' + args.com.replace(' ', '_') if args.com else ''}"
    args.save_folder = pathlib.Path(f"./runs/{args.exp_name}")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.save_folder / "segs"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    args.save_folder = args.save_folder.resolve()  # 变成绝对路径
    save_args(args)  # utils.py 里面有save_args这个函数
    t_writer = SummaryWriter(str(args.save_folder))  # Tensorboard 里面的

    # Create model
    print(f"Creating {args.arch} model...\n")

    model_maker = getattr(models, args.arch)  # 大概意思就是获取Unet,  EquiUnet 或者 Att_EquiUnet

    model = model_maker(
        4, 3,  # 问？？这个4，3 是干什么的？？？？？
        width=args.width, deep_supervision=args.deep_sup,
        norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)  # get_norm_layer函数在model下面的 layers.py里

    print(f"total number of trainable parameters {count_parameters(model)}\n")  # count_parameters 函数在utils.py里面



    if args.swa:  # 其中一个参数，stochastic weight averaging，训练后的随机梯度平均，看论文
        # Create the average model
        swa_model = model_maker(
            4, 3,
            width=args.width, deep_supervision=args.deep_sup,
            norm_layer=get_norm_layer(args.norm_layer))  # swa_model 和model的区别就是没有dropout
        for param in swa_model.parameters():
            param.detach_()
        # swa_model = swa_model.cuda()
        swa_model = swa_model.to(device)  # model 放到 GPU 上
        swa_model_optim = WeightSWA(swa_model)  # WeightSWA 在utils.py 里面

    if ngpus > 1:  # 其实这里用不着多GPU 训练
        # model = torch.nn.DataParallel(model).cuda()
        model = torch.nn.DataParallel(model).to(device)
    else:
        # model = model.cuda()
        model = model.to(device)
    # print("Model info: ", model)  # 这里有个打印 model ！！ 可以先不要！！！打出来乱七八糟的！！
    model_file = args.save_folder / "model.txt"  # 在runs 目录下面
    with model_file.open("w") as f:
        print(model, file=f)

    # criterion = EDiceLoss().cuda()   # loss是这个！！
    criterion = EDiceLoss().to(device)  # 这个是用来弄loss的 # EDiceLoss 和 metric 函数在loss下面的dice.py 里面
    metric = criterion.metric  # 这个是用来衡量结果的

    rangered = False  # needed because LR scheduling scheme is different for this optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay, eps=1e-4)

    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9,
                                    nesterov=True)

    elif args.optim == "adamw":  # adamw 没有weight decay
        print(f"Optimiser used:　adamw. Weight decay argument will not be used. Default is 11e-2")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    #    elif args.optim == "ranger":
    #        optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #        rangered = True

    # optionally resume from a checkpoint
    if args.resume:  # 这个是使用现有的ckpt, resume是args的一个参数，传入的是ckpt的文件路径
        reload_ckpt(args, model, optimizer)  # reload_ckpt函数在utils.py 里面可以找到

    if args.debug:  # 可以用来测试
        args.epochs = 3
        args.warm = 0
        args.val = 1

    if args.full:
        train_dataset, bench_dataset = get_datasets(args.seed, args.debug,
                                                    full=True)  # get_datasets函数在dataset下面的brats.py 里面能找到

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)

        bench_loader = torch.utils.data.DataLoader(
            bench_dataset, batch_size=1, num_workers=args.workers)

    else:
        # 这里竟然有seed,也不知道干什么的
        train_dataset, val_dataset, bench_dataset = get_datasets(args.seed, args.debug, fold_number=args.fold)

        train_loader = torch.utils.data.DataLoader(  # train占五分之四？validation占五分之一？？
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False,
            pin_memory=False, num_workers=args.workers, collate_fn=determinist_collate)  # 这里没太看懂，去看论文！！
        # determinist_collate在dataset的batch_utils.py 里面

        bench_loader = torch.utils.data.DataLoader(
            bench_dataset, batch_size=1, num_workers=args.workers)
        print("Val dataset number of batch:", len(val_loader))

    print("Train dataset number of batch:", len(train_loader))

    # create grad scaler
    scaler = GradScaler()  # torch.cuda.amp 里的，主要用来完成梯度缩放
    # torch.cuda.amp.autocast：主要用作上下文管理器或者装饰器，来确定使用混合精度的范围。

    # Actual Train loop
    best = np.inf  # ？？？？这是干啥的？？？
    print("start warm-up now! 3 epochs")  # 看不懂！！　cur_iter 是什么？？然后warm up 怎么弄的，看论文！！
    if args.warm != 0:  # 默认有3个epoch的最初warm
        tot_iter_train = len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,  # 这是啥玩意儿？？
                                                      lambda cur_iter: (1 + cur_iter) / (tot_iter_train * args.warm))

    patients_perf = []

    if not args.resume:
        for epoch in range(args.warm):  # 这里这是最开始的warm，还没开始正式训练！！
            ts = time.perf_counter()  # 计算训练时间的
            model.train()
            training_loss = step(train_loader, model, criterion, metric, args.deep_sup, optimizer, epoch, t_writer,
                                 scaler, scheduler, save_folder=args.save_folder,
                                 no_fp16=args.no_fp16, patients_perf=patients_perf)
            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0 and not args.full:  # 默认是每３个epoch做一次validation
                model.eval()
                with torch.no_grad():
                    validation_loss = step(val_loader, model, criterion, metric, args.deep_sup, optimizer, epoch,
                                           t_writer, save_folder=args.save_folder,
                                           no_fp16=args.no_fp16)

                t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss,
                                    epoch)  # validation_loss 比training_loss 大

    if args.warm_restart:  # 算了吧，不用了。总epoch数得是30的倍数！！
        print('Total number of epochs should be divisible by 30, else it will do odd things')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, eta_min=1e-7)
    else:  # 如果没有warm_restart又没用rangered, 就epoch+30
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               args.epochs + 30 if not rangered else round(
                                                                   args.epochs * 0.5))

    # 这里才开始了正式训练！！！！！！！！
    print("start training now!...")
    if args.swa:  # Stochastic weight averaging 训练后的随机梯度平均
        # c = 15, k=3, repeat = 5 # 这是干啥的？？？看论文！！！c, k repeat 是干啥的？？？？？？？c估计是总共swa epoch数，k是每几个epoch 更新一次
        c, k, repeat = 30, 3, args.swa_repeat
        epochs_done = args.epochs
        reboot_lr = 0
        if args.debug:
            c, k, repeat = 2, 1, 2

    for epoch in range(args.start_epoch + args.warm, args.epochs + args.warm):
        print(f"Epoch {epoch} start training: ")
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()
            model.train()
            training_loss = step(train_loader, model, criterion, metric, args.deep_sup, optimizer, epoch, t_writer,
                                 scaler, save_folder=args.save_folder,
                                 no_fp16=args.no_fp16, patients_perf=patients_perf)
            te = time.perf_counter()
            print(f"Train Epoch {epoch} done in {te - ts} s\n")

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0 and not args.full:
                print(f"Epoch {epoch} start validation: ")
                model.eval()
                with torch.no_grad():
                    validation_loss = step(val_loader, model, criterion, metric, args.deep_sup, optimizer,
                                           epoch,
                                           t_writer,
                                           save_folder=args.save_folder,
                                           no_fp16=args.no_fp16, patients_perf=patients_perf)

                t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss, epoch)

                if validation_loss < best:
                    best = validation_loss
                    model_dict = model.state_dict()  # 包含bias和weight
                    save_checkpoint(
                        dict(
                            epoch=epoch, arch=args.arch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ),
                        save_folder=args.save_folder, )

                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s\n")

            if args.swa:  # 看论文！！！！c 是干啥的？？？？这里干啥的没看懂！！！
                if (args.epochs - epoch - c) == 0:
                    reboot_lr = optimizer.param_groups[0]['lr']

            if not rangered:  # 没使用rangered
                scheduler.step()
                print("scheduler stepped!")
            else:
                if epoch / args.epochs > 0.5:
                    scheduler.step()
                    print("scheduler stepped!")

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")  # ?????????干啥的？？？
            break

    if args.swa:  # 用了随机梯度平均 ！！！！！！
        swa_model_optim.update(model)
        print("SWA Model initialised!")
        for i in range(repeat):  # repeat 就是重复了多少次的swa的30个epochs
            optimizer = torch.optim.Adam(model.parameters(), args.lr / 2,
                                         weight_decay=args.weight_decay)  # Adam 用到了weight_decay
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c + 10)  # c 是啥玩意儿来着？？看论文！！
            for swa_epoch in range(c):
                # do_epoch for one epoch
                ts = time.perf_counter()
                model.train()
                swa_model.train()
                current_epoch = epochs_done + i * c + swa_epoch  # epochs_done这里默认应该是200
                training_loss = step(train_loader, model, criterion, metric, args.deep_sup, optimizer,
                                     current_epoch, t_writer,
                                     scaler, no_fp16=args.no_fp16, patients_perf=patients_perf)
                te = time.perf_counter()
                print(f"Train Epoch done in {te - ts} s\n")

                t_writer.add_scalar(f"SummaryLoss/train", training_loss, current_epoch)
                #  怎么这TM才有Traning_loss?? 之前都是overfit

                # update every k epochs and val: # 每k个epoch 更新一次
                print(f"cycle number: {i}, swa_epoch: {swa_epoch}, total_cycle_to_do {repeat}\n")
                if (swa_epoch + 1) % k == 0:
                    swa_model_optim.update(model)
                    if not args.full:
                        model.eval()
                        swa_model.eval()
                        with torch.no_grad():  # 这怎么还两个loss呢？？
                            validation_loss = step(val_loader, model, criterion, metric, args.deep_sup, optimizer,
                                                   current_epoch,
                                                   t_writer, save_folder=args.save_folder, no_fp16=args.no_fp16)
                            swa_model_loss = step(val_loader, swa_model, criterion, metric, args.deep_sup, optimizer,
                                                  current_epoch,
                                                  t_writer, swa=True, save_folder=args.save_folder,
                                                  no_fp16=args.no_fp16)

                        t_writer.add_scalar(f"SummaryLoss/val", validation_loss, current_epoch)
                        t_writer.add_scalar(f"SummaryLoss/swa", swa_model_loss, current_epoch)
                        t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss, current_epoch)
                        t_writer.add_scalar(f"SummaryLoss/overfit_swa", swa_model_loss - training_loss, current_epoch)
                scheduler.step()
        epochs_added = c * repeat  # 最后增加的epochs
        save_checkpoint(
            dict(
                epoch=args.epochs + epochs_added, arch=args.arch,
                state_dict=swa_model.state_dict(),
                optimizer=optimizer.state_dict()
            ),
            save_folder=args.save_folder, )
    else:  # 这里没用随机梯度平均！！！就是普通model
        save_checkpoint(
            dict(
                epoch=args.epochs, arch=args.arch,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            ),
            save_folder=args.save_folder, )

    try:  # 用训练结束的模型ckpt来生成segmentation！！！
        df_individual_perf = pd.DataFrame.from_records(patients_perf)  # patients performance吗？？
        # print("DataFrame records df_individual_perf: ",df_individual_perf,"\n")
        df_individual_perf.to_csv(f'{str(args.save_folder)}/patients_indiv_perf.csv')
        reload_ckpt_bis(f'{str(args.save_folder)}/model_best.pth.tar', model)  # reload_ckpt_bis函数在utils.py文件里可以找到
        generate_segmentations(bench_loader, model, t_writer, args)  # generate_segmentations函数在utils.py文件里可以找到
        # 这是个很大的函数！！里面还要调用一个calculate_metrics的函数，很长很麻烦！！
    except KeyboardInterrupt:
        print("Stopping right now!")


# 注意这里有个step函数！！
def step(data_loader, model, criterion: EDiceLoss, metric, deep_supervision, optimizer, epoch, writer, scaler=None,
         scheduler=None, swa=False, save_folder=None, no_fp16=False, patients_perf=None):

    # 这里在Tensorborad 里面增加了model的图
    writer.add_graph(model)
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')  # utils.py 里有
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # TODO monitor teacher loss
    mode = "train" if model.training else "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []
    # print(f"fp 16 True or False ?? : {not no_fp16}")
    # TODO: not recreate data_aug for each epoch...
    # data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cuda()
    data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).to(device)
    # DataAugmenter 在models下面的 augmentation_blocks.py 里面

    for i, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # targets = batch["label"].cuda(non_blocking=True)
        if ngpus == 0:  # label就是seg那些
            targets = batch["label"].to(device)
        else:
            targets = batch["label"].cuda(non_blocking=True)

        # inputs = batch["image"].cuda()
        inputs = batch["image"].to(device)
        patient_id = batch["patient_id"]

        with autocast(enabled=not no_fp16):  # 用来自动model训练和算loss的
            # data augmentation step # 数据增广，只有train模式下才数据增广
            if mode == "train":
                inputs = data_aug(inputs)
            if deep_supervision:  # 有deep supervision
                segs, deeps = model(inputs)  # 待看model具体是怎么写的？？
                if mode == "train":  # revert the data aug #如果是deep supervision, 在train的时候就不数据增广
                    segs, deeps = data_aug.reverse([segs, deeps])
                loss_ = torch.stack(  # 注意看这里！！如果有deep supervision就会是两种loss的和
                    [criterion(segs, targets)] + [criterion(deep, targets) for
                                                  deep in deeps])
                print(f"main loss: {loss_}")
                loss_ = torch.mean(loss_)
            else:  # 无deep supervision
                segs = model(inputs)  # inputs已经转置旋转过了，训练
                if mode == "train":
                    segs = data_aug.reverse(segs)  # 训练完了为了算loss再把数据转置旋转回来！！这样才能算loss
                loss_ = criterion(segs, targets)
            if patients_perf is not None:  # 记住这里针对每个病人序号单独地加了loss！！
                patients_perf.append(
                    dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
                )

            writer.add_scalar(f"Loss/{mode}{'_swa' if swa else ''}",  # 在tensorboard里面添加了有关loss的东西
                              loss_.item(),
                              global_step=batch_per_epoch * epoch + i)

            # measure accuracy and record loss_ 衡量准确度，记录loss
            if not np.isnan(loss_.item()):
                losses.update(loss_.item())  # 这是个AverageMeter ！！！可以update
            else:
                print("NaN in model loss!!")

            if not model.training:  # 如果是validation模式，就不用criterion，用metric, 算DSC !!
                metric_ = metric(segs, targets)  # 算出来的dice: 2*(inputs 和targets都=1的数量) / (inputs=1的数量 + targets=1的数量)
                metrics.extend(metric_)

        # compute gradient and do SGD step 计算梯度 loss.backward 和step!!
        if model.training:
            scaler.scale(loss_).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)  # 这是个AverageMeter!!
        end = time.perf_counter()
        # Display progress
        progress.display(i)  # 这是个ProgressMeter !!!

    if not model.training:  # 这个应该从model里找！！
        save_metrics(epoch, metrics, swa, writer, epoch, False, save_folder)  # save_metrics函数在utils.py 可以找到
        # teacher是False!!

    if mode == "train":  # 训练模式
        writer.add_scalar(f"SummaryLoss/train", losses.avg, epoch)  # 这些在Tensorboard 都可以看到！！
    else:  # validation模式
        writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    return losses.avg


if __name__ == '__main__':
    arguments = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
