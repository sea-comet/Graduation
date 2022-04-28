# Brain tumor segmentation
# Environment 
Linux platform with GPU
Code can be run on Google Colab
# Installation

```bash
pip install -r requirements.txt
```

# Training

First change your data folder by modifying values in `src/config.py`

```python
# Used for 5 fold-Training and Validation
BRATS_TRAIN_FOLDERS = "your-Path_to/brats2020/MICCAI_BraTS_2020_Data_Training"
# Used for Testing
BRATS_VAL_FOLDER = "your-Path_to/brats2020/MICCAI_BraTS_2020_Data_Valdation"
```

Then, start training:

```bash
!python -m src.train --width 48 --arch Atten_Unet --epochs 150 --optim adam # Use Atten_Unet
```

There are other parameters that can be changed. More details on the available options for train.py:
```bash
python -m src.train -h
```

Note that this the batch size should be remain at 1, for data augmentation is done one by one.

After training, you will have a `runs` folder created containing a directory for each run you have done.

For each run, a yaml file with the option used for the runs, and 
a `segs` folder containing the generated .nii.gz segmentations for the validation fold used.

```
- src
    - runs
        - 20201127_34335135__fold_etc
            202020201127_34335135__fold_etc.ymal
            - segs
            model.txt # the printed model
            model_best.pth.tar # model weights
            patients_indiv_perf.csv # a log of training patient segmentation performance
            events.out.. # Tensorboard log
```
Drag the segmentation result in .nii.gz file in fold 'seg' into the software ITK-Snap, along with the orginal nii.gz file from any of the four modalities t1, t1ce, t2, flair, then the segmentation result on the brain can be visualized.

The yaml file is required to perform inference on the validation and train set

# Testing

The script to perform testing is... `test.py` !!

```
python -m src.inference -h 
usage: inference.py [-h] [--config PATH [PATH ...]] --devices DEVICES
                    [--on {val,train,test}] [--tta] [--seed SEED]

Brats testing dataset inference # you should change the path to the yaml file yourself!!
```bash
!python -m src.test --config "/content/Graduation/runs/×××××.yaml" --devices 0 --mode val 
```

