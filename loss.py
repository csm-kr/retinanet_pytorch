import torch
import torch.nn as nn
from utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap, encode


class TargetMaker(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.num_classes = opts.num_classes

    def forward(self, gt_boxes, gt_labels, center_anchor, IT=0.5):
        """
        gt_boxes : [B, ]
        gt_labels : [B, ]
        """
        batch_size = len(gt_labels)
        n_priors = center_anchor.size(0)
        device_ = gt_labels[0].get_device()

        # ----- 1. make container
        gt_locations = torch.zeros((batch_size, n_priors, 4), dtype=torch.float, device=device_)
        gt_classes = -1 * torch.ones((batch_size, n_priors, self.num_classes), dtype=torch.float, device=device_)

        anchor_identifier = -1 * torch.ones((batch_size, n_priors), dtype=torch.float32, device=device_)
        # if anchor is positive -> 1,
        #              negative -> 0,
        #              ignore   -> -1

        # ----- 2. make corner anchors
        center_anchor = center_anchor.to(device_)
        corner_anchor = cxcy_to_xy(center_anchor)

        for i in range(batch_size):
            boxes = gt_boxes[i]  # xy coord
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
                if positive_indices.sum() == 0:
                    _, IoU_argmax_per_object = iou.max(dim=0)  # [num_object] - 어떤 anchor 가 obj 의 max 인지 알려줌
                    positive_indices = torch.zeros_like(IoU_max)
                    positive_indices[IoU_argmax_per_object] = 1
                    positive_indices = positive_indices.type(torch.bool)
            else:
                _, IoU_argmax_per_object = iou.max(dim=0)  # [num_object] - 어떤 anchor 가 obj 의 max 인지 알려줌
                positive_indices = torch.zeros_like(IoU_max)
                positive_indices[IoU_argmax_per_object] = 1
                positive_indices = positive_indices.type(torch.bool)

            # set background 0
            argmax_labels = labels[IoU_argmax] + 1

            gt_classes[i][positive_indices, :] = 0
            gt_classes[i][positive_indices, argmax_labels[positive_indices].long()] = 1.  # objects
            anchor_identifier[i][positive_indices] = 1  # original masking \in {0, 1}
            # anchor_identifier[i][positive_indices] = IoU_max[positive_indices]           # iou masking \in [0, 1]

            # ----- 4. build gt_locations
            argmax_locations = boxes[IoU_argmax]
            center_locations = xy_to_cxcy(argmax_locations)  # [67995, 4] 0 ~ 1 사이이다. boxes 가
            gt_gcxcywh = encode(center_locations, center_anchor)
            gt_locations[i] = gt_gcxcywh

        return gt_classes, gt_locations, anchor_identifier


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
        alpha_factor = torch.ones_like(gt_cls) * self.alpha                         # alpha
        a_t = torch.where((gt_cls == 1), alpha_factor, 1. - alpha_factor)           # a_t
        p_t = torch.where(gt_cls == 1, pred_cls, 1 - pred_cls)                      # p_t
        bce = self.bce(pred_cls, gt_cls)                                            # [B, num_anchors, 1]
        cls_loss = a_t * (1 - p_t) ** self.gamma * bce                              # [B, num_anchors, 1]

        return cls_loss


class RetinaLoss(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.smooth_l1_loss = SmoothL1Loss()
        self.target_maker = TargetMaker(opts)

    def forward(self, pred, gt_boxes, gt_labels, center_anchor):
        pred_cls = pred[0]
        pred_loc = pred[1]

        # print(self.coder.center_anchor.size(0))
        n_priors = int(center_anchor.size(0))
        assert n_priors == pred_loc.size(1) == pred_cls.size(1)  # 67995 --> 120087

        # build targets
        gt_cls, gt_locs, depth = self.target_maker(gt_boxes, gt_labels, center_anchor, IT=0.5)

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


# if __name__ == '__main__':
#     from anchor import RETINA_Anchor
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')
#     parser.add_argument('--resize', type=int, default=600, help='image_size')
#     parser.add_argument('--num_classes', type=int, default=20)
#     loss_opts = parser.parse_args()
#
#     test_image = torch.randn([2, 3, 600, 600]).to(device)
#     model = RetinaNet().to(device)
#     cls, reg = model(test_image)
#     print("cls' size() :", cls.size())
#     print("reg's size() :", reg.size())
#
#     gt = [torch.Tensor([[0.426, 0.158, 0.788, 0.997], [0.0585, 0.1597, 0.8947, 0.8213]]).to(device),
#           torch.Tensor([[0.002, 0.090, 0.998, 0.867], [0.3094, 0.4396, 0.4260, 0.5440]]).to(device)]
#
#     label = [torch.Tensor([14, 15]).to(device),
#              torch.Tensor([12, 14]).to(device)]
#
#     coder = RETINA_Coder(loss_opts)
#     loss = RetinaLoss(coder=coder)
#     print(loss((cls, reg), gt, label))