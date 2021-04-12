import numpy as np
import torch
from math import sqrt
from abc import ABCMeta, abstractmethod
from config import device
# from util.utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap
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
        self.center_anchor = RETINA_Anchor('retina').create_anchors(img_size=opts.resize).to(device)
        self.num_classes = opts.num_classes

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
        return torch.cat([gcxcy, gwh], dim=1)

    def decode(self, gcxgcy):
        """
        for test and demo, gcxcy to gt
        """

        cxcy = gcxgcy[:, :2] * self.center_anchor[:, 2:] + self.center_anchor[:, :2]
        wh = torch.exp(gcxgcy[:, 2:]) * self.center_anchor[:, 2:]
        return torch.cat([cxcy, wh], dim=1)

    def post_processing(self, pred, is_demo=False):

        if is_demo:
            self.assign_anchors_to_cpu()
            pred_loc = pred[0].to('cpu')
            pred_cls = pred[1].to('cpu')
        else:
            pred_loc = pred[0]
            pred_cls = pred[1]

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