# Retinanet pytorch

re-implementation of Retinanet detection : https://arxiv.org/abs/1708.02002

### Setting

- Python 3.7
- Numpy
- pytorch >= 1.5.0 

### TODO List

- [x] Dataset
- [x] Model
- [x] Loss (Focal loss and smooth l1 loss)
- [x] Coder
- [ ] Distributed training

### Experiment
- [x] COCO 
- [ ] VOC 

COCO

|methods     | Traning Dataset   |    Testing Dataset     | Resolution | AP        |AP50     |AP75    |Time | Fps  |
|------------|-------------------| ---------------------- | ---------- | --------- |---------|--------| ----| ---- |
|papers      | COCOtrain2017     |  COCO test-dev         | 600 x 600  |  34.0     |52.5     |36.5    |98   |10.20 |
|papers      | COCOtrain2017     |  COCOval2017(minival)  | 600 x 600  |  34.3     |53.2     |36.9    |98   |10.20 |
|ours        | COCOtrain2017     |  COCOval2017(minival)  | 600 x 600  |  32.5     |50.8     |34.6    |-    |-     |

<!-- |ours*       | COCOtrain2017     |  COCO test-dev         | 600 x 600  |**34.7**   |**53.6** |**37.3**|67   |14.85 | -->
<!-- |ours*       | COCOtrain2017     |  COCOval2017(minival)  | 600 x 600  |**34.7**   |**53.5** |**37.1**|67   |14.85 | -->
<!-- dkljdkfj -->

### scheduler

- we use step LR scheduler scheme.

- whole training epoch is 13 and learning rate decay is at 9, 11

```
papers trained for 90k iterations with 16 batch.
when it comes to 1 epoch, number of training image(117266) / batch(16) = 7329.125 iterations (7.3k)
so 90k is about 13 epoch due to 7.3K * 13 = 94.9k 
and at 60k, 80k learning rate is divided by 10 to 1e-3, 1e-4

In this repo, for convinience of calculation to epochs, 
whole training epoch 13 (about 94.9k iterations)
learning rate decay at 9 (65k), 12 (87k) epochs

 paper     | this repo  | Learning Rate  
0k ~ 60K   | 0K ~ 65k   | 1e-2
60K ~ 80K  | 65k ~ 87k  | 1e-3
80K ~ 90K  | 87k ~ 95k  | 1e-4
``` 

<!-- - whole training epoch is 60 and learning rate decay is at 30, 45 -->

### training options

- batch : 16
- scheduler : step LR
- loss : focal loss and smooth l1 loss
- dataset : coco
- epoch : 13
- gpu : nvidia geforce rtx 3090 * 2EA
- lr : 1e-2

### training

- dataset

    train : trainval35k == train2017

    test : minval2014 == val2017

- data augmentation

    only use horizontal flipping same as papers.
    
<!-- 2. this repo uses data augmentation (random crop, expand, flip, photometric distortion, resize) refers to https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py -->

- results

    minival eval
    
    
```
Evaluate annotation type *bbox*
DONE (t=89.60s).
Accumulating evaluation results...
DONE (t=13.50s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.535
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.181
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.532
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.605
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.710
```

### Reference

ssd tutorial : data augmentation and detection structure

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

retina net pytorch

https://github.com/NVIDIA/retinanet-examples

https://github.com/yhenon/pytorch-retinanet

### Start Guide


