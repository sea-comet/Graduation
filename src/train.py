import argparse
import time
import pathlib
from datetime import datetime
import pandas as pd
import numpy as np

import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.dataset import get_datasets
from src.dataset.batch_utils import determinist_collate  # 重分配
from src import models
from src.models import get_norm_layer, DataAugmenter
from src.loss import EDiceLoss

from src.utils import AverageMeter, save_args, ProgressMeter, save_checkpoint, reload_ckpt, reload_ckpt_bis, \
    count_parameters, save_metrics, generate_segmentations


parser = argparse.ArgumentParser(description='Brats Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='Atten_Unet',
                    help='model architecture (default: Atten_Unet)')
parser.add_argument('--width', default=48, help='base number of features for Unet (x2 per downsampling)', type=int)
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')  # 这个用在重新启动，200 epoch 后面训练的那150个epoch !!
parser.add_argument('--epochs', default=150, type=int, metavar='N',
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

parser.add_argument('--debug', action="store_true")

parser.add_argument('--seed', default=16111990, help="seed for train/val split")
parser.add_argument('--warm', default=3, type=int, help="number of warming up epochs")

parser.add_argument('--val', default=3, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")  # 这个应该是交叉验证
parser.add_argument('--norm_layer', default='group')

parser.add_argument('--optim', choices=['adam', 'sgd', 'adamw'], default='adam')

parser.add_argument('--dropout', type=float, help="amount of dropout to use", default=0.)

parser.add_argument('--full', action='store_true', help='Fit the network on the full training set')  # 用整个训练集训练！

#
# Add a device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training device used: ", device)

# setup
ngpus = torch.cuda.device_count()
if ngpus == 0:
     raise RuntimeWarning("This could only be run on GPU environment")

print(f"Working with {ngpus} GPUs")


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
                    f"_lr{args.lr}-wd{args.weight_decay}_epochs{args.epochs}" \
                    f"_warm{args.warm}_" \
                    f"_dropout{args.dropout}"


    args.save_folder = pathlib.Path(f"./runs/{args.exp_name}")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.save_folder / "segs"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    args.save_folder = args.save_folder.resolve()  # 变成绝对路径
    save_args(args)  # utils.py 里面有save_args这个函数
    t_writer = SummaryWriter(str(args.save_folder))  # Tensorboard 里面的

    # Create model
    print(f"Creating {args.arch} model...\n")

    model_maker = getattr(models, args.arch)  # 大概意思就是获取Unet, 或 Atten_Unet

    model = model_maker(
        4, 3,  # inplanes: 4 代表4个modality 相当于channels,  num_class: 3 代表ET, TC, WT
        width=args.width,
        norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)  # get_norm_layer函数在model下面的 layers.py里

    model = model.to(device)

    # print("Model info: ", model)  # 打印 model 信息，layer信息
    model_file = args.save_folder / "model.txt"  # 在runs 目录下面
    with model_file.open("w") as f:
        print(model, file=f)

    # criterion = EDiceLoss().cuda()   # loss是这个！！
    criterion = EDiceLoss().to(device)  # 这个是用来弄loss的 # EDiceLoss 和 metric 函数在loss下面的dice.py 里面
    metric = criterion.metric  # 这个是用来衡量结果的

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
            training_loss = step(train_loader, model, criterion, metric, optimizer, epoch, t_writer,
                                 scaler, scheduler, save_folder=args.save_folder,
                                  patients_perf=patients_perf)
            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0 and not args.full:  # 默认是每３个epoch做一次validation
                model.eval()
                with torch.no_grad():
                    validation_loss = step(val_loader, model, criterion, metric, optimizer, epoch,
                                           t_writer, save_folder=args.save_folder)

                t_writer.add_scalar(f"SummaryLoss/overfit", validation_loss - training_loss,
                                    epoch)  # validation_loss 比training_loss 大



    # 这里才开始了正式训练！！！！！！！！
    print("start training now!...")

    for epoch in range(args.start_epoch + args.warm, args.epochs + args.warm):
        print(f"Epoch {epoch} start training: ")
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()
            model.train()
            training_loss = step(train_loader, model, criterion, metric, optimizer, epoch, t_writer,
                                 scaler, save_folder=args.save_folder,
                                 patients_perf=patients_perf)
            te = time.perf_counter()
            print(f"Train Epoch {epoch} done in {te - ts} s\n")

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0 and not args.full:
                print(f"Epoch {epoch} start validation: ")
                model.eval()
                with torch.no_grad():
                    validation_loss = step(val_loader, model, criterion, metric, optimizer,
                                           epoch,
                                           t_writer,
                                           save_folder=args.save_folder,
                                           patients_perf=patients_perf)

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

            scheduler.step()
            print("scheduler stepped!")


        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")  # ?????????干啥的？？？
            break


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
        # 这是个很大的函数！！里面还要调用一个calculate_metrics的函数，很长很麻烦！！还得写入val.txt !!
    except KeyboardInterrupt:
        print("Stopping right now!")


# 注意这里有个step函数！！
def step(data_loader, model, criterion: EDiceLoss, metric, optimizer, epoch, writer, scaler=None,
         scheduler=None, save_folder=None, patients_perf=None):

    # Setup
    batch_time = AverageMeter('BatchTime', ':6.3f')  # utils.py 里有
    data_time = AverageMeter('DataTime', ':6.3f')
    losses = AverageMeter('Loss', ':6.4f')
    Acc = AverageMeter('Acc', ':6.4f')
    # TODO monitor teacher loss
    mode = "train" if model.training else "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses, Acc],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []

    # TODO: not recreate data_aug for each epoch...
    # data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cuda()
    # data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).to(device)
    # 这个Augmentation就很奇怪，是在step里根据每个脑子生成一个随机数来决定要不要augment的，后面可以改成按dataset来弄
    # 这里暂时改成drop channel 是False了
    data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=False).to(device)
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

        with autocast(enabled=True):  # 用来自动model训练和算loss的
            # data augmentation step # 数据增广，只有train模式下才数据增广
            if mode == "train":
                inputs = data_aug(inputs)

            segs = model(inputs)  # inputs已经转置旋转过了，训练
            if mode == "train":
                segs = data_aug.reverse(segs)  # 训练完了为了算loss再把数据转置旋转回来！！这样才能算loss
            loss_ = criterion(segs, targets)

            if patients_perf is not None:  # 记住这里针对每个病人序号单独地加了loss！！
                patients_perf.append(
                    dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
                )

            writer.add_scalar(f"Loss/{mode}",  # 在tensorboard里面添加了有关loss的东西
                              loss_.item(),
                              global_step=batch_per_epoch * epoch + i)

            # measure accuracy and record loss_ 衡量准确度，记录loss
            if not np.isnan(loss_.item()):
                losses.update(loss_.item())
                Acc.update(1. - loss_.item())# 这是个AverageMeter ！！！可以update
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
        progress.display(i+1)  # 这是个ProgressMeter !!!

    if not model.training:  # 这个应该从model里找！！
        save_metrics(epoch, metrics, writer, epoch, False, save_folder)  # save_metrics函数在utils.py 可以找到
        # teacher是False!!

    if mode == "train":  # 训练模式
        writer.add_scalar(f"SummaryLoss/train", losses.avg, epoch)  # 这些在Tensorboard 都可以看到！！
        writer.add_scalar(f"SummaryAcc/train", 1. - losses.avg, epoch) # 在tensorboard加了一个关于accuracy的scaler
    else:  # validation模式
        writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)
        writer.add_scalar(f"SummaryAcc/val", 1. - losses.avg, epoch)

    return losses.avg


if __name__ == '__main__':
    arguments = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
