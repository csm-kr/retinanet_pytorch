import cv2
import math
import torch
import numpy as np
import torch.nn as nn
from torchvision.ops import nms
from utils import cxcy_to_xy, decode
from models.anchor import create_anchors
from models.modules import ClsModule, Resnet50, RegModule, FPN


class RetinaNet(nn.Module):
    def __init__(self, num_classes, img_size):
        super(RetinaNet, self).__init__()

        # anchor - https://discuss.pytorch.org/t/module-cuda-not-moving-module-tensor/74582/5
        # self.register_buffer('anchors', create_anchors(img_size))
        self.anchors = create_anchors(img_size)

        # module
        self.backbone = Resnet50(pretrained=True)
        self.fpn = FPN()
        self.cls_module = ClsModule(num_classes)
        self.reg_module = RegModule()
        self.initialize_sub_modules()
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_sub_modules(self):
        i = 0
        for c in self.cls_module.features.children():
            if isinstance(c, nn.Conv2d):
                if i == 8:  # final layer
                    pi = 0.01
                    b = - math.log((1 - pi) / pi)
                    nn.init.constant_(c.bias, b)
                    nn.init.normal_(c.weight, std=0.01)

                else:
                    nn.init.normal_(c.weight, std=0.01)
                    nn.init.constant_(c.bias, 0)
            i += 1
        for c in self.reg_module.features.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

    # https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/model.py
    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        features = self.fpn(c3, c4, c5)  # [p3, p4, p5, p6, p7 ]
        cls = torch.cat([self.cls_module(feature) for feature in features], dim=1)
        reg = torch.cat([self.reg_module(feature) for feature in features], dim=1)
        return cls, reg

    def predict(self, cls, reg, center_anchor, opts):
        pred_cls = cls
        pred_reg = reg

        pred_cls = pred_cls.squeeze()                                                   # for batch 1, [67995, num_classes]
        pred_bbox = cxcy_to_xy(decode(pred_reg.squeeze(), center_anchor)).clamp(0, 1)   # for batch 1, [67995, 4]
        pred_bbox = pred_bbox.clamp(min=0, max=1)

        bbox, label, score = self._suppress(pred_bbox, pred_cls, opts)
        return bbox, label, score

    def _suppress(self, raw_cls_bbox, raw_prob, opts):
        bbox = list()
        label = list()
        score = list()

        # skip cls_id = 0 because it is the background class
        for l in range(1, opts.num_classes):
            # cls_bbox_l = raw_cls_bbox.reshape((-1, self.num_classes, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > opts.thres
            cls_bbox_l = raw_cls_bbox[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, iou_threshold=0.3)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


if __name__ == '__main__':
    img = torch.randn([2, 3, 600, 600]).cuda()
    model = RetinaNet(num_classes=81, img_size=600).cuda()
    output = model(img)
    print(output[0].size())
    print(output[1].size())
