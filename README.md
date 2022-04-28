# Graduation Project: Brain tumor segmentation -- Dong Wanqi 
# Environment 
* Linux platform with GPU.
* Code can be run on Google Colab
# Fast running on Google Colab

I also prepared colab notebooks that allow you to run the algorithm on Google Colab. Upload the file to Google Colab and then start running !

*  [Graduation_Google Colab](https://github.com/sea-comet/Graduation/blob/master/Graduation_Google%20Colab.ipynb) 
# Installation

```bash
pip install -r requirements.txt
```

# Training

First change your data folder by modifying values in `src/config.py` `BRATS_TRAIN_FOLDERS` is for 5-fold training and validation, which use dataset with ground truth labels seg.nii. `BRATS_VAL_FOLDER` is for testing, which use dataset with no labels. Both Datasets come from BraTS2020.

```python
# Used for 5 fold-Training and Validation
BRATS_TRAIN_FOLDERS = "your-Path_to/BraTS2020_TrainingData/MICCAI_BraTS_2020_Data_Training"
# Used for Testing
BRATS_VAL_FOLDER = "your-Path_to/BraTS2020_ValidationData/MICCAI_BraTS_2020_Data_Valdation"
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
a `segs` folder containing the generated .nii.gz segmentations for the validation folder used.

```
- src
    - runs
        - 20220407_154427__fold0_etc 
            20220407_154427__fold0_etc.ymal
            - segs
            model.txt # the printed model
            model_best.pth.tar # model weights
            patients_indiv_perf.csv # a log of training patient segmentation performance
            events.out.. # Tensorboard log file
```
Drag the segmentation result in .nii.gz file in folder 'seg' into the software ITK-Snap, along with the orginal nii.gz file from any of the four modalities t1, t1ce, t2, flair, then the segmentation result on the brain can be visualized.

The yaml file is required to perform inference on testing dataset 

# Testing

The script to perform testing is `test.py` !!  You should change the path to the yaml file yourself!!

```bash
!python -m src.test --config "/content/Graduation/runs/×××××.yaml" --devices 0 --mode val 

```
For other options:
```
python -m src.test -h 
```




