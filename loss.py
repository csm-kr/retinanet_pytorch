import torch
import torch.nn as nn
from utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap
from config import device
from model import RetinaNet
import torch.nn.functional as F


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class Focal_Loss(nn.Module):
    def __init__(self, coder):
        super().__init__()

        self.coder = coder
        self.priors_cxcy = self.coder.center_anchor
        self.priors_xy = cxcy_to_xy(self.priors_cxcy)
        self.num_classes = self.coder.num_classes
        self.bce = nn.BCELoss(reduction='none')
        self.smooth_l1 = SmoothL1Loss()
        # self.smooth_l1 = nn.SmoothL1Loss(reduction=None)

    def forward(self, pred, b_boxes, b_labels):
        """
        Forward propagation.
        :param pred (loc, cls) prediction tuple (N, 67995, 4) / (N, 67995, num_classes) or [120087] anchors
        :param labels: true object labels, a list of N tensors
        """
        pred_loc = pred[0]
        pred_cls = pred[1]

        batch_size = pred_loc.size(0)
        n_priors = self.priors_xy.size(0)

        assert n_priors == pred_loc.size(1) == pred_cls.size(1)  # 67995 --> 120087

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)                        # (N, 67995, 4)
        true_classes = -1 * torch.ones((batch_size, n_priors, self.num_classes), dtype=torch.float).to(device)  # (N, 67995, num_classes)
        depth = -1 * torch.ones((batch_size, n_priors), dtype=torch.bool).to(device)                            # (N, 67995)

        for i in range(batch_size):
            boxes = b_boxes[i]  # xy coord
            labels = b_labels[i]

            ###################################################
            #           match strategies  -> make target      #
            ###################################################
            iou = find_jaccard_overlap(self.priors_xy, boxes)  # [67995, num_objects]
            IoU_max, IoU_argmax = iou.max(dim=1)               # [67995]

            negative_indices = IoU_max < 0.4

            # =======================  make true classes ========================
            true_classes[i][negative_indices, :] = 0           # make negative

            depth[i][negative_indices] = 0

            positive_indices = IoU_max >= 0.5                  # iou 가 0.5 보다 큰 아이들 - [67995]
            argmax_labels = labels[IoU_argmax]                 # assigned_labels

            # class one-hot encoding
            # 0 으로 만들고 이후에 1 을 넣어주기
            true_classes[i][positive_indices, :] = 0
            true_classes[i][positive_indices, argmax_labels[positive_indices].long()] = 1.  # objects

            depth[i][positive_indices] = 1

            # ===========================  make true locs ===========================
            true_locs_ = xy_to_cxcy(boxes[IoU_argmax])                               # [67995, 4] 0~1 사이이다. boxes 가
            true_locs_ = self.coder.encode(true_locs_)
            true_locs[i] = true_locs_

        # ------------------------------------------ cls loss ------------------------------------------
        alpha = 0.25
        gamma = 2

        alpha_factor = torch.ones_like(true_classes).to(device) * alpha                    # alpha
        a_t = torch.where((true_classes == 1), alpha_factor, 1. - alpha_factor)            # a_t
        p_t = torch.where(true_classes == 1, pred_cls, 1 - pred_cls)                       # p_t
        bce = self.bce(pred_cls, true_classes)
        cls_loss = a_t * (1 - p_t) ** gamma * bce                                          # focal loss

        cls_mask = (depth >= 0).unsqueeze(-1).expand_as(cls_loss)                          # both fore and back ground
        num_of_pos = (depth > 0).sum().float().clamp(min=1)                                # only foreground (min=1)
        cls_loss = (cls_loss * cls_mask).sum() / num_of_pos                                # batch 의 bce loss
                                                                                           # / batch 의 object 총갯수

        # ------------------------------------------ loc loss ------------------------------------------
        loc_mask = (depth > 0).unsqueeze(-1).expand_as(true_locs)                          # only foreground
        loc_loss = self.smooth_l1(pred_loc, true_locs)  # (), scalar
        loc_loss = (loc_mask * loc_loss).sum() / num_of_pos
        # loc_loss *= 2                                                                      # balance values

        total_loss = (cls_loss + loc_loss)
        return total_loss, (loc_loss, cls_loss)


if __name__ == '__main__':
    from anchor import RETINA_Anchor
    from coder import RETINA_Coder
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')
    parser.add_argument('--resize', type=int, default=600, help='image_size')
    parser.add_argument('--num_classes', type=int, default=20)
    loss_opts = parser.parse_args()

    test_image = torch.randn([2, 3, 600, 600]).to(device)
    model = RetinaNet().to(device)
    reg, cls = model(test_image)
    print("cls' size() :", cls.size())
    print("reg's size() :", reg.size())

    gt = [torch.Tensor([[0.426, 0.158, 0.788, 0.997], [0.0585, 0.1597, 0.8947, 0.8213]]).to(device),
          torch.Tensor([[0.002, 0.090, 0.998, 0.867], [0.3094, 0.4396, 0.4260, 0.5440]]).to(device)]

    label = [torch.Tensor([14, 15]).to(device),
             torch.Tensor([12, 14]).to(device)]

    coder = RETINA_Coder(loss_opts)
    loss = Focal_Loss(coder=coder)
    print(loss((reg, cls), gt, label))