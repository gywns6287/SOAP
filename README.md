# SOAP: Vision-Centric 3D Semantic Scene Completion with Scene-Adaptive Decoder and Occluded Region-Aware View Projection
![demo](https://github.com/gywns6287/SOAP/blob/main/assets/best_prediction.gif)

This repo is the official Code of SOAP: Vision-Centric 3D Semantic Scene Completion with Scene-Adaptive Decoder and Occluded Region-Aware View Projection (**[CVPR 2025](link)**).

## Method
![model](https://github.com/gywns6287/SOAP/blob/main/assets/model.png)

Overview of the proposed SOAP. Given multi-scale image features, an occluded region-aware view projection (OAP) transforms  them into voxel space, refining duplicated features in occluded voxel regions using historical information. A scene-adaptive decoder enriches query embeddings with diverse semantic contexts through a semantic repository, which is constructed and iteratively updated by  a repository builder.

## News
**[2053/04/01]**  Code and demo release.

## Getting Started
[1] Check [installation](https://github.com/gywns6287/SOAP/blob/main/docs/Installation.md) for installation.

[2] Check [data_preparation](https://github.com/gywns6287/SOAP/blob/main/docs/Dataset.md) for preparing SemanticKITTI and SSCBench datasets.

[3] Check [train_eval_test](https://github.com/gywns6287/SOAP/blob/main/docs/train_eval_test.md) for training, evaluation, and test submission.


## Model Zoo
We provide the pretrained weights on SemanticKITTI and SSCBench-KITTI360 datasets. 

| Dataset             | Backbone     | Val IoU / mIoU | Test IoU / mIoU | Model Weights |
|---------------------|--------------|----------------|------------------|----------------|
| SemanticKITTI       | EfficientNetB7 | 47.2 / 19.2    | 46.1 / 19.1       | [Download](https://drive.google.com/file/d/1MQt6FoVI7xRoseC97UWmOt0zhp3nNrgU/view?usp=drive_link) |
| SemanticKITTI       | ResNet50 | 48.1 / 18.8    | 47.5 / 18.7       | [Download](https://drive.google.com/file/d/13HeZdzJNb0ld-i2-L_BCth7z9M9FswQW/view?usp=drive_link) |
| SSCBench-KITTI360 | EfficientNetB7 | - / -    | 48.2 / 21.2       | [Download](https://drive.google.com/file/d/1A72mzz-I5E5heOaQt3w35TFPSoeigodJ/view?usp=drive_link) |
| SSCBench-KITTI360       | ResNet50 | -/-  | 48.5 / 20.3       | [Download](https://drive.google.com/file/d/1OqH8Rbiq5m_mMmOaaPMWsLPtd1rn3R30/view?usp=drive_link) |


## Acknowledgement
This project is developed based on the following open-source projects: [VoxFormer](https://github.com/NVlabs/VoxFormer), [OccFormer](https://github.com/zhangyp15/OccFormer), and [CGFormer](https://github.com/pkqbajng/CGFormer).
We would like to thank the authors of these works for their excellent contributions.
