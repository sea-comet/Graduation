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
        self.validation = no_seg # no_seg 用于validation和testing，no need for seg.nii
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        if not no_seg: # 用于training
            self.patterns += ["_seg"]
        for patient_dir in patients_dir:
            patient_id = patient_dir.name # get patient Number
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
            )
            self.datas.append(patient)  # every patient--> a dict ,path to 4 modalities nii.gz and seg.nii.gz

    def __getitem__(self, idx):
        _patient = self.datas[idx]  # _patient--> one patinet's dict # load_nii down, 传参为.nii file path
        # patient_image：4 modalities t1, t1ce, t2, flair array，dict
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax": # minmax normalize
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
            # irm_min_max_preprocess --> image_utils.py
        elif self.normalisation == "zscore": # zscore normalize
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
            # zscore_normalise函数在dataset的image_utils.py文件里
        patient_image = np.stack([patient_image[key] for key in patient_image]) # 4个模态stack，增维度为4
        if _patient["seg"] is not None: # has seg
            et = patient_label == 4   # ET label 4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1) # NET label 1，TC = ET + NET = label 4 + label 1
            wt = np.logical_or(tc, patient_label == 2) # edema label 2，WT = ET + NET + edema = label 4 + 1 + 2
            patient_label = np.stack([et, tc, wt])  # 增维度，3个分类class
        else:
            patient_label = np.zeros(patient_image.shape)
            et_present = 0
        if self.training: # 训练！！
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # 整个脑子的四个模态合体后，取出所有不等于0的坐标，即，肿瘤存在地方的坐标
            # Add 1 pixel in each side #　在最小值和最大值上add 1, minimal bounding box
            # 使用Minimal bounding box
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]

            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128　　　　＃　pad_or_crop_image --> image_utils.py
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:  # validation or testing,inference

            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            # pad
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]

        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label,
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder): # 传入的参数为nii地址
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas) if not self.debug else 3


def get_datasets(seed, debug, no_seg=False, on="train", full=False,
                 fold_number=0, normalisation="minmax"):                # 默认: minmax normalization
    base_folder = pathlib.Path(get_brats_folder(on)).resolve()  # get_brats_folder --> config.py
    print(f"{on} dataset path : {base_folder}\n")
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()]) # extract path, sort
    if full: # full dataset, no 5 fold
        train_dataset = Brats(patients_dir, training=True, debug=debug,
                              normalisation=normalisation)
        bench_dataset = Brats(patients_dir, training=False, benchmarking=True, debug=debug,
                              normalisation=normalisation)
        return train_dataset, bench_dataset
    if no_seg:
        return Brats(patients_dir, training=False, debug=debug,
                     no_seg=no_seg, normalisation=normalisation)
    kfold = KFold(5, shuffle=True, random_state=seed)  #  need seed, every time the same, 5 fold
    splits = list(kfold.split(patients_dir))
    # print("打印出来path 用5 fold产出来的splits:", splits)
    train_idx, val_idx = splits[fold_number] # return path id.
    # print("first idx of train", train_idx[0])
    # print("first idx of test", val_idx[0])
    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_idx]
    # return patients_dir # 4/5 train, 1/5 validate.
    train_dataset = Brats(train, training=True,  debug=debug,
                          normalisation=normalisation)
    val_dataset = Brats(val, training=False, data_aug=False,  debug=debug,
                        normalisation=normalisation)         # 记住这儿training设置的是False,一开始data没有切成128，128，128
    bench_dataset = Brats(val, training=False, benchmarking=True, debug=debug,
                          normalisation=normalisation)
    return train_dataset, val_dataset, bench_dataset
