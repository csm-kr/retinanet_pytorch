import numpy as np
import torch
from math import sqrt
from abc import ABCMeta, abstractmethod
from config import device
# from util.utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap
from utils import find_jaccard_overlap, xy_to_cxcy, xy_to_cxcy2
from collections import OrderedDict
from anchor import RETINA_Anchor
import torch.nn.functional as F
from utils import cxcy_to_xy


class Coder(metaclass=ABCMeta):

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass


class RETINA_Coder(Coder):
    def __init__(self, opts):
        super().__init__()
        self.data_type = opts.data_type
        self.center_anchor = RETINA_Anchor('retina').create_anchors(img_size=opts.resize)
        self.num_classes = opts.num_classes

        # standard variance for encoding and decoding.It is small since the boxes are more accurate.
        # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
        # See https://github.com/weiliu89/caffe/issues/155

        # self.variance = torch.tensor(data=[0.1, 0.1, 0.2, 0.2], requires_grad=False)
        # degrade performance from 0.317 to 0.288

    def assign_anchors_to_device(self):
        self.center_anchor = self.center_anchor.to(device)

    def assign_anchors_to_cpu(self):
        self.center_anchor = self.center_anchor.to('cpu')

    def encode(self, cxcy):
        """
        for loss, gt(cxcy) to gcxcy
        """
        gcxcy = (cxcy[:, :2] - self.center_anchor[:, :2]) / self.center_anchor[:, 2:]       # (box cxy-anc cxy)/anc wh
        gwh = torch.log(cxcy[:, 2:] / self.center_anchor[:, 2:])                            # log(box wh / anc wh)
        # return torch.cat([gcxcy, gwh], dim=1) / self.variance.to(cxcy.get_device())
        return torch.cat([gcxcy, gwh], dim=1)

    def decode(self, gcxgcy):
        """
        for test and demo, gcxcy to gt
        """
        # gcxgcy *= self.variance.to(gcxgcy.get_device())
        cxcy = gcxgcy[:, :2] * self.center_anchor[:, 2:] + self.center_anchor[:, :2]
        wh = torch.exp(gcxgcy[:, 2:]) * self.center_anchor[:, 2:]
        return torch.cat([cxcy, wh], dim=1)

    # IT - IoU Threshold == 0.5
    def build_target(self, gt_boxes, gt_labels, IT=None):
        """
        gt_boxes : [B, ]
        gt_labels : [B, ]
        """
        batch_size = len(gt_labels)
        n_priors = self.center_anchor.size(0)
        device_ = gt_labels[0].get_device()

        # ----- 1. make container
        gt_locations = torch.zeros((batch_size, n_priors, 4), dtype=torch.float, device=device_)
        gt_classes = -1 * torch.ones((batch_size, n_priors, self.num_classes), dtype=torch.float, device=device_)

        anchor_identifier = -1 * torch.ones((batch_size, n_priors), dtype=torch.float32, device=device_)
        # if anchor is positive -> 1,
        #              negative -> 0,
        #              ignore   -> -1

        # ----- 2. make corner anchors
        self.center_anchor = self.center_anchor.to(device_)
        corner_anchor = cxcy_to_xy(self.center_anchor)

        for i in range(batch_size):
            boxes = gt_boxes[i]     # xy coord
            labels = gt_labels[i]

            # ----- 3. find iou between anchors and boxes
            iou = find_jaccard_overlap(corner_anchor, boxes)
            IoU_max, IoU_argmax = iou.max(dim=1)

            # ----- 4. build gt_classes
            # if iou < 0.4 -> negative class and set 0, anchor_identifier is zero
            negative_indices = IoU_max < 0.4
            gt_classes[i][negative_indices, :] = 0
            anchor_identifier[i][negative_indices] = 0

            if IT is not None:
                # if iou > 0.5 -> positive class and set label, anchor_identifier is 1
                positive_indices = IoU_max >= 0.5  # [120087] - binary values
            else:
                # if a anchor has max iou -> just one positive anchor.
                _, IoU_argmax_per_object = iou.max(dim=0)    # [num_object] - 어떤 anchor 가 obj 의 max 인지 알려줌
                positive_indices = torch.zeros_like(IoU_max)
                positive_indices[IoU_argmax_per_object] = 1
                positive_indices = positive_indices.type(torch.bool)

            argmax_labels = labels[IoU_argmax]
            gt_classes[i][positive_indices, :] = 0
            gt_classes[i][positive_indices, argmax_labels[positive_indices].long()] = 1.  # objects
            anchor_identifier[i][positive_indices] = 1                                    # original masking \in {0, 1}
            # anchor_identifier[i][positive_indices] = IoU_max[positive_indices]           # iou masking \in [0, 1]

            # ----- 4. build gt_locations
            argmax_locations = boxes[IoU_argmax]
            center_locations = xy_to_cxcy(argmax_locations)  # [67995, 4] 0 ~ 1 사이이다. boxes 가
            gt_gcxcywh = self.encode(center_locations)
            gt_locations[i] = gt_gcxcywh

        return gt_classes, gt_locations, anchor_identifier

    def post_processing(self, pred, is_demo=False):

        if is_demo:
            self.assign_anchors_to_cpu()
            pred_cls = pred[0].to('cpu')
            pred_loc = pred[1].to('cpu')
        else:
            pred_cls = pred[0]
            pred_loc = pred[1]

        n_priors = self.center_anchor.size(0)
        assert n_priors == pred_loc.size(1) == pred_cls.size(1)

        # decode 에서 나온 bbox 는 center coord
        pred_bboxes = cxcy_to_xy(self.decode(pred_loc.squeeze())).clamp(0, 1)        # for batch 1, [67995, 4]
        pred_scores = pred_cls.squeeze()                                             # for batch 1, [67995, num_classes]

        # corner coordinates 를 x1y1x2y2 를 0 ~ 1 로 scaling 해줌
        # 0.3109697496017331 -> 0.3115717185294685 로 오름

        return pred_bboxes, pred_scores


if __name__ == '__main__':
    retina_coder = RETINA_Coder()
    retina_coder.assign_anchors_to_device()
    print(retina_coder.center_anchor)