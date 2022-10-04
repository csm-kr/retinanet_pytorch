# Retinanet pytorch

re-implementation of Retinanet detection : https://arxiv.org/abs/1708.02002

### Requirements

- Python 3.7
- Numpy
- pytorch >= 1.5.0 

### Training Setting

```
- batch size : 16
- optimizer : SGD
- epoch : 30 
- initial learning rate 0.01
- weight decay : 5e-4
- momentum : 0.9
- scheduler : cosineannealing LR (min : 5e-5)
```

### TODO List

- [x] Dataset
- [x] COCO 
- [x] VOC 
- [x] Model
- [x] Loss (Focal loss and smooth l1 loss)
- [x] Coder
- [x] Distributed training (distributed data parallel)

### Results

- VOC
```
start..evaluation
90.25% = aeroplane AP 
84.79% = bicycle AP 
87.98% = bird AP 
76.75% = boat AP 
63.18% = bottle AP 
90.86% = bus AP 
92.45% = car AP 
95.38% = cat AP 
66.37% = chair AP 
91.59% = cow AP 
78.50% = diningtable AP 
92.67% = dog AP 
90.68% = horse AP 
86.94% = motorbike AP 
84.95% = person AP 
59.69% = pottedplant AP 
89.97% = sheep AP 
81.69% = sofa AP 
89.28% = train AP 
83.85% = tvmonitor AP 
mAP = 83.89%
```

- COCO
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.369
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.181
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.383
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.493
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.307
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.464
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.488
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.291
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.669
mAP :  0.3446613792728923
mean Loss :  0.3393706703444406
Eval Time : 253.8811
```
|methods     | Traning Dataset   |    Testing Dataset     | Resolution | AP        |AP50     |AP75    |Time | Fps  |
|------------|-------------------| ---------------------- | ---------- | --------- |---------|--------| ----| ---- |
|papers      | COCOtrain2017     |  COCO test-dev         | 600 x 600  |  34.0     |52.5     |36.5    |98   |10.20 |
|papers      | COCOtrain2017     |  COCOval2017(minival)  | 600 x 600  |  34.3     |53.2     |36.9    |98   |10.20 |
|ours        | COCOtrain2017     |  COCO test-dev         | 600 x 600  |  -     |-     |-    |-   |- |
|ours        | COCOtrain2017     |  COCOval2017(minival)  | 600 x 600  |  34.5     |53.3     |36.9    |-   |- |

### Distributed learning

- Distributed Data Parallel for fully using the gpu memory and decreasing training time
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 72%   69C    P2   328W / 350W |  14395MiB / 24268MiB |     98%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:21:00.0 Off |                  N/A |
| 65%   65C    P2   325W / 350W |  13394MiB / 24268MiB |     99%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
 2800~2900 s/epoch 
 
--------------

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 70%   69C    P2   298W / 350W |  11154MiB / 24268MiB |     98%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:21:00.0 Off |                  N/A |
| 62%   63C    P2   291W / 350W |   8764MiB / 24268MiB |     83%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
 240x s/epoch
training time : 285x s/epoch -> 240x s/epoch (improvement about 15 %)

### Quick Start

- training
```
main.py --config ./configs/retinanet_coco_train.txt
main.py --config ./configs/retinanet_voc_train.txt
```

- testing & demo

1- download "VOC, COCO" best weights from [here](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/ETi9zFxZ1E9Hnr63z4Azu3EBMJJsLzeNHR5IEHFDJScVbg?e=6mSW8T) and [here](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/Ee2ebGMjDCNAmMFWJvEFZoABWiDiSHSruCErn1Jg4NOHsA?e=ZGodv2)

2- and set saves file like bellows:
```
logs
    |-- retinanet_coco
          |-- saves
                |-- retinanet_coco.best.pth.tar
    |-- retinanet_voc
          |-- saves
                |-- retinanet_voc.best.pth.tar
```

3- and run script like bellows:

(test)

```
test.py --config ./configs/retinanet_coco_test.txt
test.py --config ./configs/retinanet_voc_test.txt
```

(demo)

```
demo.py --config ./configs/retinanet_coco_demo.txt
demo.py --config ./configs/retinanet_voc_demo.txt
```

### Reference
```
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
https://github.com/NVIDIA/retinanet-examples
https://github.com/yhenon/pytorch-retinanet
https://github.com/liangheming/retinanetv1
```

### Citation
If you found this implementation and pretrained model helpful, please consider citation
```
@misc{csm-kr_retinanet_pytorch,
  author={Sungmin, Cho},
  publisher = {GitHub},
  title={retinanet_pytorch},
  url={https://github.com/csm-kr/retinanet_pytorch},
  year={2022},
}
```