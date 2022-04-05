import pathlib

import SimpleITK as sitk
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset

from src.config import get_brats_folder
from src.dataset.image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise


class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, debug=False, data_aug=False,
                 no_seg=False, normalisation="minmax"):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.debug = debug
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg # no_seg 就代表用的是validation的数据集，没有seg.nii
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        if not no_seg: # 说明用的training的数据集
            self.patterns += ["_seg"]
        for patient_dir in patients_dir:
            patient_id = patient_dir.name # 取出来病人代表的数字
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
            )
            self.datas.append(patient)  #datas里面放了数据集里所有病人的各模态和seg数据的path,每个病人有个字典

    def __getitem__(self, idx):
        _patient = self.datas[idx]  # _patient 为某病人id和4模态以及seg的path字典, load_nii 函数在下面呢！！是个static 函数，传参为.nii文件的地址
        # patient_image：某病人4个模态的array，组成的字典
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax": # 用最大最小值来normalize
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
            # irm_min_max_preprocess函数在dataset下面的image_utils.py 里面有
        elif self.normalisation == "zscore": # 用均值和标准差来normalize
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
            # zscore_normalise函数在dataset的image_utils.pye文件里有
        patient_image = np.stack([patient_image[key] for key in patient_image]) # 4个模态摞在一起，增维度,数字为4
        if _patient["seg"] is not None:
            et = patient_label == 4   # ET标签是4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1) # NET标签是1，TC = ET+NET = 标签4+标签1
            wt = np.logical_or(tc, patient_label == 2) # edema 标签是2，WT = ET+NET+edema = 标签4+标签1+标签2
            patient_label = np.stack([et, tc, wt])  # 增维度，是3个class，数字为3
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training: # 训练！！
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # 整个脑子的四个模态合体后，取出所有不等于0的坐标，即，肿瘤存在地方的坐标
            # Add 1 pixel in each side #　就在最小值和最大值上加了一个pixel !!,牛逼！！切出最小能包括整个肿瘤的框！！
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            # 用元组一起循环也可以啊！！
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128　　　　＃　pad_or_crop_image函数在dataset下面的image_utils.py 里面有
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:  # validation or testing,inference 好像有专门的数据预处理，所以不用先crop,而且validation时还得生成整个脑子的彩色图才行
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]   #因为每个肿瘤的大小框都不一定一样，所以可能很多要pad！
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]

        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label,
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present, # 这是干啥用的？？
                    supervised=True,  # 这是干啥用的？？
                    )

    @staticmethod
    def load_nii(path_folder): # 传入的参数为nii地址
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas) if not self.debug else 3


def get_datasets(seed, debug, no_seg=False, on="train", full=False,
                 fold_number=0, normalisation="minmax"):                # 默认是minmax normalization
    base_folder = pathlib.Path(get_brats_folder(on)).resolve()  # get_brats_folder 函数在config.py 里面
    print(f"{on} dataset path : {base_folder}\n")
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()]) # 取出数据集里各个数据的地址，并排序
    if full: # 如果用了整个数据集,就是,没有用 5 fold 交叉验证
        train_dataset = Brats(patients_dir, training=True, debug=debug,
                              normalisation=normalisation)
        bench_dataset = Brats(patients_dir, training=False, benchmarking=True, debug=debug,
                              normalisation=normalisation)
        return train_dataset, bench_dataset
    if no_seg:
        return Brats(patients_dir, training=False, debug=debug,
                     no_seg=no_seg, normalisation=normalisation)
    kfold = KFold(5, shuffle=True, random_state=seed)  # 5折验证，这里用了seed !!为什么呢???
    splits = list(kfold.split(patients_dir))
    # print("打印出来path 用5 fold产出来的splits看看是什么玩意儿：", splits)
    train_idx, val_idx = splits[fold_number] # 这返回的应该是path的id
    # print("first idx of train", train_idx[0])
    # print("first idx of test", val_idx[0])
    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_idx]
    # return patients_dir # 用了 5 fold 交叉验证的, 用了五分之四的数据来训练
    train_dataset = Brats(train, training=True,  debug=debug,
                          normalisation=normalisation)
    val_dataset = Brats(val, training=False, data_aug=False,  debug=debug,
                        normalisation=normalisation)
    bench_dataset = Brats(val, training=False, benchmarking=True, debug=debug,  # 这个benchmark 到底是个什么玩意儿？
                          normalisation=normalisation)                          # 就标了一个benchmarking 的flag 啥也没干！
    return train_dataset, val_dataset, bench_dataset
