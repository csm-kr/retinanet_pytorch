import os
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import nms as torchvision_nms
import numpy as np
import math

# for voc label
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
voc_label_map = {k: v for v, k in enumerate(voc_labels)}
voc_label_map['background'] = 20
voc_rev_label_map = {v: k for k, v in voc_label_map.items()}  # Inverse mapping
np.random.seed(0)
voc_color_array = np.random.randint(256, size=(21, 3)) / 255  # In plt, rgb color space's range from 0 to 1

# for coco label
coco_labels = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_label_map = {k: v for v, k in enumerate(coco_labels)}  # {0 ~ 79 : 'person' ~ 'toothbrush'}
coco_label_map['background'] = 80                                # {80 : 'background'}
coco_rev_label_map = {v: k for k, v in coco_label_map.items()}  # Inverse mapping
np.random.seed(1)
coco_color_array = np.random.randint(256, size=(81, 3)) / 255  # In plt, rgb color space's range from 0 to 1


def bar_custom(current, total, width=30):
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d byte]" % (current / total * 100, percent_bar, current, total)
    return progress


def cxcy_to_xy(cxcy):

    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=1)


def xy_to_cxcy(xy):

    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=1)


def xy_to_cxcy2(xy):
    wh = xy[..., 2:] - xy[..., :2]
    cxcy = xy[..., :2] + 0.5 * wh
    return torch.cat([cxcy, wh], dim=1)


def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps  # (n1, n2)

    return intersection / union  # (n1, n2)


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)  # 0 혹은 양수로 만드는 부분
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)  # 둘다 양수인 부분만 존재하게됨!


def encode(gt_cxywh, anc_cxywh):
    tg_cxy = (gt_cxywh[:, :2] - anc_cxywh[:, :2]) / anc_cxywh[:, 2:]
    tg_wh = torch.log(gt_cxywh[:, 2:] / anc_cxywh[:, 2:])
    tg_cxywh = torch.cat([tg_cxy, tg_wh], dim=1)
    return tg_cxywh


def decode(tcxcy, center_anchor):
    cxcy = tcxcy[:, :2] * center_anchor[:, 2:] + center_anchor[:, :2]
    wh = torch.exp(tcxcy[:, 2:]) * center_anchor[:, 2:]
    cxywh = torch.cat([cxcy, wh], dim=1)
    return cxywh


def resume(opts, model, optimizer, scheduler):
    if opts.start_epoch != 0:
        # take pth at epoch - 1

        f = os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(opts.start_epoch - 1))
        device = torch.device('cuda:{}'.format(opts.gpu_ids[opts.rank]))
        checkpoint = torch.load(f=f,
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])                              # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                      # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])                      # load sched state dict
        if opts.rank == 0:
            print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))
    else:
        if opts.rank == 0:
            print('\nNo check point to resume.. train from scratch.\n')
    return model, optimizer, scheduler
