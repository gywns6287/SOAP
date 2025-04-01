## Prepare Dataset

### a. SemanticKITTI
To prepare the SemanticKITTI dataset, first download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color and calibration files), along with the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download).

To generate the pseudo-depth maps, you can follow the preprocessing steps from [VoxFormer](https://github.com/NVlabs/VoxFormer/tree/main/preprocess).

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
│   │   │   │   │   |   ├── 000000.npy
│   │   │   │   │   |   ├── ...
│   │   │   │   ├── ...
│   │   │   │   ├── 21
```

### b. SSCBench-KITTI360 
Download the dataset from [SSCBench-KITTI-360](https://github.com/ai4ce/SSCBench), and prepare the depth maps by following the preprocessing steps from [VoxFormer](https://github.com/NVlabs/VoxFormer/tree/main/preprocess).  After setup, your dataset directory should look like this:
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
│   │   │   │   │   ├── 000000.npy
│   │   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── 2013_05_28_drive_0010_sync/

```
