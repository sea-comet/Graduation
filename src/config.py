import os
from pathlib import Path

# user = "YOU"
# BRATS_TRAIN_FOLDERS = f"/home/{user}/Datasets/brats2020/training"
# BRATS_VAL_FOLDER = f"/home/{user}/Datasets/brats2020/MICCAI_BraTS2020_ValidationData"
# BRATS_TEST_FOLDER = f"/home/{user}/Datasets/brats2020/MICCAI_BraTS2020_TestingData"

BRATS_TRAIN_FOLDERS = "./../Dataset_small/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
BRATS_VAL_FOLDER = "./../Dataset_small/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
BRATS_TEST_FOLDER = "./../Dataset_small/BraTS2020_TestingData/MICCAI_BraTS2020_TestingData"

'''
用于colab的路径
BRATS_TRAIN_FOLDERS = "./../drive/My Drive/Dataset_small/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
BRATS_VAL_FOLDER = "./../drive/My Drive/Dataset_small/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
BRATS_TEST_FOLDER = "./../drive/My Drive/Dataset_small/BraTS2020_TestingData/MICCAI_BraTS2020_TestingData"
'''

def get_brats_folder(on="val"):
    if on == "train":
        return os.environ['BRATS_FOLDERS'] if 'BRATS_FOLDERS' in os.environ else BRATS_TRAIN_FOLDERS
    elif on == "val":
        return os.environ['BRATS_VAL_FOLDER'] if 'BRATS_VAL_FOLDER' in os.environ else BRATS_VAL_FOLDER
    elif on == "test":
        return os.environ['BRATS_TEST_FOLDER'] if 'BRATS_TEST_FOLDER' in os.environ else BRATS_TEST_FOLDER
