

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
