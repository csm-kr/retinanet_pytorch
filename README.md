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
- epoch : 13 
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
85.76% = aeroplane AP 
80.25% = bicycle AP 
81.58% = bird AP 
68.81% = boat AP 
54.17% = bottle AP 
86.81% = bus AP 
89.26% = car AP 
91.04% = cat AP 
57.68% = chair AP 
70.44% = cow AP 
75.21% = diningtable AP 
83.24% = dog AP 
75.26% = horse AP 
82.89% = motorbike AP 
81.87% = person AP 
50.89% = pottedplant AP 
80.59% = sheep AP 
77.87% = sofa AP 
85.40% = train AP 
76.09% = tvmonitor AP 
mAP = 76.76%
it takes 117.71sec.
0.7675624038181124
Eval Time : 379.8731
```

- COCO


```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.307
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.467
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.670
0.346526727016717

```


|methods     | Traning Dataset   |    Testing Dataset     | Resolution | AP        |AP50     |AP75    |Time | Fps  |
|------------|-------------------| ---------------------- | ---------- | --------- |---------|--------| ----| ---- |
|papers      | COCOtrain2017     |  COCO test-dev         | 600 x 600  |  34.0     |52.5     |36.5    |98   |10.20 |
|papers      | COCOtrain2017     |  COCOval2017(minival)  | 600 x 600  |  34.3     |53.2     |36.9    |98   |10.20 |
|ours        | COCOtrain2017     |  COCO test-dev         | 600 x 600  |  -     |-     |-    |-   |- |
|ours        | COCOtrain2017     |  COCOval2017(minival)  | 600 x 600  |  -     |-     |-    |-   |- |

- VOC


- trained weight can get at [here](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EUDJTzLdWyxNjoGfYapaGCUBwsrjK6R5yr77Uk4YnHubBw?e=nQRtMH)


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

- testing
```
test.py --config ./configs/retinanet_coco_test.txt
test.py --config ./configs/retinanet_voc_test.txt
```

- demo
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