# Graduation Project: Brain tumor segmentation -- Dong Wanqi 
# Environment 
* Linux platform with GPU.
* Code can be run on Google Colab
# Fast running on Google Colab

I also prepared colab notebooks that allow you to run the algorithm on Google Colab. Upload the file to Google Colab and then start training & validation or testing

* runnning on Google Colab: [Graduation_Google Colab notebook](https://github.com/sea-comet/Graduation/blob/master/Graduation_Google%20Colab.ipynb) 
# Installation

```python
pip install -r requirements.txt
```

# Training

First change your data folder by modifying the dataset path in `src/config.py` 
* `BRATS_TRAIN_FOLDERS` is for 5-fold training and validation, which is the path to dataset with ground truth label "seg.nii" file. 
* `BRATS_VAL_FOLDER` is for seperate validation or testing, which is the path to dataset with no ground truth labels. Both Datasets come from BraTS2020.

```python
# Used for 5 fold-Training and Validation
BRATS_TRAIN_FOLDERS = "your-Path_to/BraTS2020_TrainingData/MICCAI_BraTS_2020_Data_Training"
# Used for seperate validation or Testing
BRATS_VAL_FOLDER = "your-Path_to/BraTS2020_ValidationData/MICCAI_BraTS_2020_Data_Valdation"
```

Then, start training:

run the instruction in command line tools

```python
python -m src.train --width 48 --arch Atten_Unet --epochs 150 # Use Atten_Unet
```

There are other parameters that can be used. More details on the available options for train.py:
```python
python -m src.train -h
```

Note that this the batch size should be set to 1, for data augmentation is done volume by volume.

After training, you will have a `runs` folder created containing a directory for each run you have done.

* For each run, a yaml file with the option used for the runs will be generated
* A `segs` folder containing the generated .nii.gz files. These are the segmentation results.
* First drag the orginal nii.gz file from any of the four modalities t1, t1ce, t2, flair into the software `ITK-Snap`[[Download link]](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3)(A specilaized software used for medical data visualization) to show the brain background, then drag the segmentation result (.nii.gz file) in folder `segs` into the software ITK-Snap, too, then the segmentation result on the brain can be visualized.
* You can also use the "display nii data.ipynb" to see the segmentation

The content in the `runs` folder. 

```
- src
    - runs
        - 20220407_154427__fold0_etc 
            20220407_154427__fold0_etc.ymal # Used for testing
            - segs
              - BraTS20_Training_ID.nii
            model.txt     # model structure
            model_best.pth.tar     # model weights, used for validation and testing
            patients_indiv_perf.csv    # a log of every patient's segmentation performance in traning 
            results.csv     # Validation results for each patient and for each metric
            val.txt    # validation result for each epoch
            events.out.. # Tensorboard log file
```

# Testing

* The `yaml` file is required to perform inference on testing dataset. `model_best.pth.tar` should also be under the same directory
* The script to perform testing is `test.py` !!  You should use "--config option" change the path to the yaml file you created in the training procedure yourself!!

Run the test.py

```python
python -m src.test --config "/content/Graduation/runs/×××××.yaml" --devices 0 --mode val 

```
For other options:
```python
python -m src.test -h 
```

# Results for Training & Validation & Testing & Segmentation 
### 毕设训练结果.zip[ Can be found in the link below ]


* 链接: [百度网盘](https://pan.baidu.com/s/1k_6mCowWd16sU8yR2jxQpw)
* `提取码: dg8v `


It contains three seperate results for training & validation and testing. 

* Mainly look at the results in folder `3 run_Atten Unet_drop 0.2_noise (0.9,1.1)_150 epoch`, `3 pred_Atten Unet_drop 0.2_noise (0.9,1.1)_150 epoch`, and `3 Tensorboard 图像` because they use the final chosen parameters and model. The results in folder name start with 4 used U-Net structure as comparison, and the results in folder name start with 5 choose to use no channel dropiing.
 
* The explanation of files in the folder `3 run_Atten Unet_drop 0.2_noise (0.9,1.1)_150 epoch` can be found in `Training` section, since they are actually the `runs` folder

# Model 
*  Model proposed in this project: Atten_Unet.
*  It is a varient from 3D-Unet + CBAM block. Each block was modified.
*  The architecture for Atten_Unet
![image](https://github.com/sea-comet/Graduation/blob/master/images/model.png)

# Results
* For segmentation result: 
* Here is the comparison between the predicted segmentation result and the ground truth labels for patient 9
### predicted segmentation result
![image](https://github.com/sea-comet/Graduation/blob/master/images/patient%209_Pred%20seg.png)
### ground truth labels
![image](https://github.com/sea-comet/Graduation/blob/master/images/patient%209_Ground%20truth.png)

