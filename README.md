# SOAP: Vision-Centric 3D Semantic Scene Completion with Scene-Adaptive Decoder and Occluded Region-Aware View Projection
This repo is the official Code of SOAP: Vision-Centric 3D Semantic Scene Completion with Scene-Adaptive Decoder and Occluded Region-Aware View Projection (**CVPR 2025**).
## News
**[2053/04/01]**  Code and demo release.


## Prepare Dataset

### a. SemanticKITTI
To prepare the SemanticKITTI dataset, first download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color and calibration files), along with the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download).

To generate the pseudo-depth maps, you can either follow the preprocessing steps from [VoxFormer](https://github.com/NVlabs/VoxFormer/tree/main/preprocess) or download them directly from [stereodepth]().

Place all `.zip` files under `SOAP/data/SemanticKITTI` and unzip them. After extraction, your dataset directory should look like this:
```
SOAP
├── data/
│   ├── SemanticKITTI/
│   │   ├── dataset/
│   │   │   ├── sequences
│   │   │   │   ├── 00
│   │   │   │   │   ├── calib.txt
│   │   │   │   │   ├── image_2/
│   │   │   │   │   ├── voxels/
│   │   │   │   ├── 01
│   │   │   │   ├── 02
│   │   │   │   ├── ...
│   │   │   │   ├── 21
|   |   |   ├── stereodepth/
│   │   │   │   ├── 00
│   │   │   │   │   ├── depth/
│   │   │   │   ├── ...
│   │   │   │   ├── 21
```

### b. SSCBench-KITTI360 
Download the dataset from [SSCBench-KITTI-360](https://github.com/ai4ce/SSCBench), and prepare the depth maps by following the preprocessing steps from [VoxFormer](https://github.com/NVlabs/VoxFormer/tree/main/preprocess), or download them directly from [stereodepth_bench]().  After setup, your dataset directory should look like this:
```
SOAP
├── data/
│   ├── SSCBench-KITTI360/
│   │   ├── data_2d_raw/
│   │   │   ├── 2013_05_28_drive_0000_sync/
|   │   │   │   ├── image_00/
|   │   │   │   ├── voxels/
|   │   │   │   ├── poses.txt
│   │   │   ├── 2013_05_28_drive_0002_sync/
│   │   │   ├── ...
│   │   │   ├── 2013_05_28_drive_0010_sync/
│   │   ├── stereodepth_bench/
│   │   │   ├── 2013_05_28_drive_0000_sync/
|   │   │   │   ├── depth/
│   │   │   ├── ...
│   │   │   ├── 2013_05_28_drive_0010_sync/

```


## Training & Evaluation
For most of our experiments, we trained the model using 4 RTX A6000 GPUs.  
Before starting training, please download the corresponding pretrained weights for [EfficientNetB7](https://github.com/zhangyp15/OccFormer/releases/download/assets/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth) and [ResNet50](https://github.com/fregu856/deeplabv3/blob/master/pretrained_models/resnet/resnet50-19c8e357.pthh).

### Training
```
#SemanticKITTI with resnet
bash tools/dist_train.sh projects/configs/soap_resnet_kitti.py {num_gpus}
#SemanticKITTI with efficientnet
bash tools/dist_train.sh projects/configs/soap_kitti.py {num_gpus}
#SSCBench-KITTI360 with resnet
bash tools/dist_train.sh projects/configs/soap_resnet_bench.py {num_gpus}
#SSCBench-KITTI360 with efficientnet
bash tools/dist_train.sh projects/configs/soap_bench.py {num_gpus}
```
During the training process, the model is evaluated on the validation set after every epoch. The checkpoint with best performance will be saved. The output logs and checkpoints will be available at work_dirs/$CONFIG.

### Evaluation
```
#SemanticKITTI with resnet
bash tools/dist_test.sh projects/configs/soap_resnet_kitti.py {weights_file.pth} {num_gpus}
#SemanticKITTI with efficientnet
bash tools/dist_test.sh projects/configs/soap_kitti.py {weights_file.pth} {num_gpus}
#SSCBench-KITTI360 with resnet
bash tools/dist_test.sh projects/configs/soap_resnet_bench.py {weights_file.pth} {num_gpus}
#SSCBench-KITTI360 with efficientnet
bash tools/dist_test.sh projects/configs/soap_bench.py{weights_file.pth} {num_gpus}
```

### Test submissions for SemanticKITTI
To generate submission files for the test set, please follow the steps below:

1.  In either `SOAP/projects/configs/soap_kitti.py` or `SOAP/projects/configs/soap_resnet_kitti.py`,  
    change **line 275** to:   `split='test'` 
2.   In `SOAP/tools/dist_test.sh`, uncomment **line 14** by removing the `#` symbol before:  `--test-save=results` 
3.  Run the following command to generate submission files
 ```
 #SemanticKITTI with resnet
bash tools/dist_test.sh projects/configs/soap_resnet_kitti.py {weights_file.pth} {num_gpus}
#SemanticKITTI with efficientnet
bash tools/dist_test.sh projects/configs/soap_kitti.py {weights_file.pth} {num_gpus}
 ```
4. Compress the predictions:
```
cd results && zip -r {ZIP_FILE} sequences
```
5. Finally, submit the `{ZIP_FILE}` to the [competition site](https://codalab.lisn.upsaclay.fr/competitions/7170#participate) and check the results.
