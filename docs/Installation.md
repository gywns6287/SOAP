## Installation

**a. Create a conda virtual environment**
```shell
conda create -n soap python=3.7 -y
conda activate soap
```

**(optional) Install gcc-6.2**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**b. Install pytorch and torch vision**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
or 
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**c. Install mmcv, mmdet, and mmseg**
```shell
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

**d. Install mmdet3d 0.17.1**
This is a custom version of MMDetection3D provided by OccFormer, which  includes additional operations like bev-pooling, voxel pooling. 

```shell
cd mmdetection3d
pip install -r requirements/runtime.txt
python setup.py install
cd ..
```

**e. build multi-scal deformable attention**

```shell
cd projects/mmdet3d_plugin/soap/necks/ops/
python setup.py install
cd -
```


**f. Install other dependencies, like timm, einops, torchmetrics, etc.**
```shell
pip install -r requirements.txt
```
