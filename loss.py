import torch
import torch.nn as nn
from config import device
from model import RetinaNet


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1/9):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred_cls, gt_cls):
        alpha_factor = torch.ones_like(gt_cls).to(device) * self.alpha              # alpha
        a_t = torch.where((gt_cls == 1), alpha_factor, 1. - alpha_factor)           # a_t
        p_t = torch.where(gt_cls == 1, pred_cls, 1 - pred_cls)                      # p_t
        bce = self.bce(pred_cls, gt_cls)                                            # [B, num_anchors, 1]
        cls_loss = a_t * (1 - p_t) ** self.gamma * bce                              # [B, num_anchors, 1]
        return cls_loss


class RetinaLoss(nn.Module):
    def __init__(self, coder):
        super().__init__()
        self.coder = coder
        self.focal_loss = FocalLoss()
        self.smooth_l1_loss = SmoothL1Loss()

    def forward(self, pred, gt_boxes, gt_labels):
        pred_cls = pred[0]
        pred_loc = pred[1]

        n_priors = self.coder.center_anchor.size(0)
        assert n_priors == pred_loc.size(1) == pred_cls.size(1)  # 67995 --> 120087

        # build targets
        gt_cls, gt_locs, depth = self.coder.build_target(gt_boxes, gt_labels, IT=0.5)

        # make mask & num_of_pos
        num_of_pos = (depth > 0).sum().float()  # only foreground
        cls_mask = (depth >= 0).unsqueeze(-1).expand_as(gt_cls)  # both fore and back ground
        loc_mask = (depth > 0).unsqueeze(-1).expand_as(gt_locs)                      # boolean

        # cls loss
        cls_loss = self.focal_loss(pred_cls, gt_cls)

        # loc loss
        loc_loss = self.smooth_l1_loss(pred_loc, gt_locs)

        # masking
        cls_loss = (cls_loss * cls_mask).sum() / num_of_pos
        loc_loss = (loc_mask * loc_loss).sum() / num_of_pos
        total_loss = cls_loss + loc_loss
        return total_loss, (cls_loss, loc_loss)


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
    cls, reg = model(test_image)
    print("cls' size() :", cls.size())
    print("reg's size() :", reg.size())

    gt = [torch.Tensor([[0.426, 0.158, 0.788, 0.997], [0.0585, 0.1597, 0.8947, 0.8213]]).to(device),
          torch.Tensor([[0.002, 0.090, 0.998, 0.867], [0.3094, 0.4396, 0.4260, 0.5440]]).to(device)]

    label = [torch.Tensor([14, 15]).to(device),
             torch.Tensor([12, 14]).to(device)]

    coder = RETINA_Coder(loss_opts)
    loss = RetinaLoss(coder=coder)
    print(loss((cls, reg), gt, label))