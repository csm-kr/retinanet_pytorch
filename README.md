# Retinanet pytorch

re-implementation of Retinanet detection : https://arxiv.org/abs/1708.02002

### Setting

- Python 3.7
- Numpy
- pytorch >= 1.5.0 

### TODO List

- [x] Dataset
- [x] Model
- [x] Loss
- [X] Coder

### Experiment
- [ ] VOC 실험하기
- [x] COCO 실험하기

COCO

|methods        | Traning Dataset        |    Testing Dataset     | Resolution | AP        |AP50   |AP75    |Time(ms) | Fps  |
|---------------|------------------------| ---------------------- | ---------- | --------- |-------|--------|:------ :| ---- |
|papers         | COCOtrain2017          |  COCO test-dev         | 600 x 600  |  34.0     |52.5   |36.5    |98       |10.20 |
|papers         | COCOtrain2017          |  COCOval2017(minival)  | 600 x 600  |  34.3     |53.2   |36.9    |98       |10.20 |
|our repo       | COCOtrain2017          |  COCO test-dev         | 600 x 600  |-          |-      |-       |-        |-     |
|our repo       | COCOtrain2017          |  COCOval2017(minival)  | 600 x 600  |***34.7*** |53.5   |37.1    |67       |14.85 |

best epoch : 58

### scheduler

- we use step LR scheduler scheme.

- whole training epoch is 60 and learning rate decay is at 30, 45

### training options

- batch : 16
- scheduler : step LR
- loss : focal loss and smooth l1 loss
- dataset : coco
- epoch : 60
- gpu : nvidia geforce rtx 3090
- lr : 1e-2

### training

- dataset

train : trainval35k == train2017

test : minval2014 == val2017

- data augmentation

1. papers use only horizontal image flipping for data augmentation

2. this repo uses 

or
test-dev 

```

batch : 16 
[8, 12]/14 - 32.14 mAP/600
[8, 18]/20 - 
[8, 12]/14 - 34.07 mAP / 800


cf)
총 90K 돈다.
   paper   |    epoch   |   iter   |  accum        |
0 ~ 60K    | (7329 * 8) |  58,632  |  58,632 (59k) | ~ 8 epoch [0, 7]    8
1e-2
60K ~ 80K  | (7329 * 3) |  21,987  |  80,619 (81K) | ~ 11 epoch [8, 11] 12
1e-3
80K ~ 90K  | (7329 * 2) |  14,685  |  95,304 (95K) | ~ 13 epoch         14
1e-4

iteration : 90K
batch * iteraion = 16 * 90,000 = 1,440,000 / 117266

117266 / 16 -> 7329 iteration (7K)
1 epoch 에 약 7K
10 epoch 은 약 70K 
20 epoch 은 약 140K


```

### experiments

```
1. 
```
```
2. 
```
### Start Guide


