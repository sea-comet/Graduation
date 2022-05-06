# Graduation Project: 3D MRI Brain tumor segmentation <br>(Dong Wanqi)

# Report & PPT
Report and PPT and be found here:
* [Report](https://github.com/sea-comet/Graduation/blob/master/report%20and%20PPT/Dong%20Wanqi_2018213196_FinalReport.pdf)
* [PPT](https://github.com/sea-comet/Graduation/blob/master/report%20and%20PPT/Dong%20Wanqi_2018213196_FinalViva.pdf)

# Environment 
* Platform with GPU.
* Code can be run on Google Colab

# Fast running on Google Colab

I also prepared colab notebooks that allow you to run the algorithm on Google Colab. Upload the file to Google Colab and then start training & validation or testing

* runnning on Google Colab: [Graduation_Google Colab notebook](https://github.com/sea-comet/Graduation/blob/master/src/Graduation_Google%20Colab.ipynb) 
# Installation

```python
pip install -r requirements.txt
```

# Training & Validation

First change your data folder by modifying the dataset path in `src/config.py` 
* `BRATS_TRAIN_FOLDERS` is for 5-fold training and validation, which is the path to dataset with ground truth label "seg.nii" file. 
* `BRATS_VAL_FOLDER` is for independent validation or testing, which is the path to dataset with no ground truth labels. <br>Both Datasets come from BraTS2020.

```python
# Used for 5 fold-Training and Validation
BRATS_TRAIN_FOLDERS = "your-Path_to/BraTS2020_TrainingData/MICCAI_BraTS_2020_Data_Training"
# Used for independent validation or Testing
BRATS_VAL_FOLDER = "your-Path_to/BraTS2020_ValidationData/MICCAI_BraTS_2020_Data_Valdation"
```

Then, start training:

Run the following instruction in command line tools.

```python
python -m src.train --width 48 --arch Atten_Unet --epochs 150 # Use Atten_Unet
```

There are other parameters that can be used. More details on the available options for train.py:
```python
python -m src.train -h
```

Note that the batch size should be set to 1, for data augmentation is done volume by volume.

After training, you will have a `runs` folder created containing a directory for each run you have done.

* For each run, a yaml file with the option used for the runs will be generated

* A `segs` folder containing all the generated .nii.gz files for validation samples. These are the segmentation results.
* First drag the orginal nii.gz file from any of the four modalities t1, t1ce, t2, flair into software `ITK-Snap`[[Download link]](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3)(A specilaized software used for medical data visualization) to show the brain background, then drag the segmentation result `.nii` file in folder `segs` into the software ITK-Snap, too, then the segmentation result on the brain can be visualized.
* You can also use the "display nii data.ipynb" file to see the segmentation


The contents in the `runs` folder and their corresponding explanations.<br> Many files are created during training and validation containing the records and results.

```
- src
    - runs
        - 20220407_154427__fold0_etc 
            20220407_154427__fold0_etc.ymal # Used for testing
            - segs
              - BraTS20_Training_ID.nii
              - ....
            model.txt     # printed model structure
            model_best.pth.tar     # best model weights, used for validation and testing
            patients_indiv_perf.csv    # a log of every patient's segmentation performance in training 
            results.csv     # validation results for each patient in each metric
            val.txt    # validation result at each epoch
            events.out.. # Tensorboard log file, containing lots of graph during training 
```

# Testing

* The `yaml` file is required to perform inference on testing dataset. `model_best.pth.tar` should also be under the same directory.
* The script to perform testing is `test.py`.  You should use "--config" option to change the path to the yaml file you created in the training procedure yourself!!

Run the test.py

```python
python -m src.test --config "/content/Graduation/runs/×××××.yaml" --devices 0 --mode val 

```
For other available options:
```python
python -m src.test -h 
```

# Results for Training & Validation & Testing & Segmentation 
### 毕设训练结果.zip[ Can be found in the link below ]


* 链接: [百度网盘](https://pan.baidu.com/s/1k_6mCowWd16sU8yR2jxQpw)
* `提取码: dg8v `

如果上面链接失效，请用下面新的链接

* 链接2: [百度网盘2](https://pan.baidu.com/s/1wyR8uKIQ4-0mD8sfMkpVOw)
* `提取码: ls1w`

It contains three seperate results for training & validation and testing. 

Please mainly look at the results in folder `3 run_Atten Unet_drop 0.2_noise (0.9,1.1)_150 epoch`, `3 pred_Atten Unet_drop 0.2_noise (0.9,1.1)_150 epoch`, and `3 Tensorboard 图像` because they use the ultimate chosen parameters and model. The results in folder with a name start with 4 used U-Net structure as comparison, while the results in folder with a name start with 5 used no channel dropiing.
 
The explanation of each file in folder `3 run_Atten Unet_drop 0.2_noise (0.9,1.1)_150 epoch` can be found in `Training` section, since they are actually the `runs` folder

# Model 
Model proposed in this project: Atten_Unet. <br>
It is a varient from 3D-Unet + CBAM block. Each block was modified. <br>
The architecture of Atten_Unet:

![image](https://github.com/sea-comet/Graduation/blob/master/images/model.png)

# Results
Segmentation results: <br>
Here is the comparison between the predicted segmentation result and the ground truth labels for patient No. 9

### predicted segmentation result
![image](https://github.com/sea-comet/Graduation/blob/master/images/patient%209_Pred%20seg.png)
### ground truth labels
![image](https://github.com/sea-comet/Graduation/blob/master/images/patient%209_Ground%20truth.png)
<br>
They look very similar, isn't it?<br>
The qualitative results are also excellent. Please see the detailed explanation in the report.

