import torch
import random
from utils import find_jaccard_overlap
from torch.nn.functional import interpolate
import torchvision.transforms.functional as F


def detection_resize(image,
                     boxes,
                     labels,
                     size,
                     max_size=None,
                     box_normalization=True):
    """
    detection 을 위한 resize function
    :param image: PIL image
    :param boxes: target tensor : [N, 4]
    :param size: resized size (2개 -> resize됨, 1개 -> 작은값 기준으로 resize)
    :param max_size: (1개 & max_size 큰 값 기준으로 resize)
    :return: resize image, scaled boxes
    """
    # 1. get original size
    # w, h = image.size
    h = image.size(1)
    w = image.size(2)

    # ----------check get aspect ratio ------------
    # min_size_ = float(min((h, w)))
    # max_size_ = float(max((h, w)))
    # aspect_ratio = max_size_ / min_size_
    # max = 0
    # if aspect_ratio > max:
    #     max = aspect_ratio
    #     print(aspect_ratio)

    # 2. get resize size
    if isinstance(size, (list, tuple)):
        size = size
    else:
        if max_size is not None:
            min_original_size = float(min((h, w)))
            max_original_size = float(max((h, w)))

            # e.g) 800 ~ 1333
            # 작은값을 800으로 맞추었을때의 큰값이 1333 을 넘으면,
            if size / min_original_size * max_original_size > max_size:
                # 큰 값을 1333 으로 맞추었을때의 작은값을 size로 정한다. (더 작아짐)
                size = int(round(max_size / max_original_size * min_original_size))

        # 3. get aspect_ratio
        if (w <= h and w == size) or (h <= w and h == size):
            size = (h, w)
        else:
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            size = (oh, ow)

    rescaled_image = F.resize(image, size)

    new_h, new_w = size
    new_h, new_w = float(new_h), float(new_w)
    old_h, old_w = h, w
    old_h, old_w = float(old_h), float(old_w)
    ratio_height = new_h / old_h
    ratio_width = new_w / old_w

    scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height]).unsqueeze(0)
    if box_normalization:
        scaled_boxes /= torch.as_tensor([new_w, new_h, new_w, new_h]).unsqueeze(0)
    return rescaled_image.squeeze(0), scaled_boxes, labels


def detection_hflip(image, boxes, labels):
    flipped_image = F.hflip(image)
    w, h = image.size
    boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
    return flipped_image, boxes, labels


def detection_zoomout(image, boxes, labels, max_scale):
    # original_w, original_h = image.size
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = max_scale
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    mean = torch.mean(image, (1, 2))
    filler = torch.FloatTensor(mean)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)

    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_image, new_boxes, labels


def detection_zoonin(image, boxes, labels, max_trials):
    original_h = image.size(1)
    original_w = image.size(2)
    while True:
        min_overlap = random.choice([.1, .3, .5, .7, .9, None])  # 'None' refers to no cropping
        if min_overlap is None:
            return image, boxes, labels

        for _ in range(max_trials):
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)
            overlap = overlap.squeeze(0)

            if overlap.max().item() < min_overlap:
                continue

            new_image = image[:, top:bottom, left:right]
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * \
                              (bb_centers[:, 1] > top) * (bb_centers[:, 1] < bottom)

            if not centers_in_crop.any():
                continue

            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def detection_photometric_distort(image, boxes, labels):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image
    distortions = [F.adjust_brightness,
                   F.adjust_contrast,
                   F.adjust_saturation,
                   F.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if d.__name__ is 'adjust_hue':
            adjust_factor = random.uniform(-18 / 255., 18 / 255.)
        else:
            adjust_factor = random.uniform(0.5, 1.5)
        new_image = d(new_image, adjust_factor)
    return new_image, boxes, labels


class DetCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class DetToTensor(object):
    def __call__(self, image, boxes, labels):
        return F.to_tensor(image), boxes, labels


# ######################################## Before Tensor() ########################################
class DetRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            return detection_hflip(image, boxes, labels)
        return image, boxes, labels


class DetRandomPhotoDistortion(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            return detection_photometric_distort(image, boxes, labels)
        return image, boxes, labels


# ######################################## After Tensor() ########################################
class DetRandomZoomOut(object):
    def __init__(self, p=0.5, max_scale=3):
        self.p = p
        self.max_scale = max_scale

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            return detection_zoomout(image, boxes, labels, self.max_scale)
        return image, boxes, labels


class DetRandomZoomIn(object):
    def __init__(self, p=0.5, max_trials=50):
        self.p = p
        self.max_trials = max_trials

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            return detection_zoonin(image, boxes, labels, self.max_trials)
        return image, boxes, labels


class DetNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, boxes, labels):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, boxes, labels


class DetResize(object):

    def __init__(self, size, max_size=None, box_normalization=True):
        """

        :param size:
        :param max_size:
        :param norm_boxes: normalized boxes if True : [0 ~ 1]
        """
        self.size = size
        self.max_size = max_size
        self.box_normalization = box_normalization

    def __call__(self, image, boxes, labels):
        return detection_resize(image=image, boxes=boxes, labels=labels,
                                size=self.size,
                                max_size=self.max_size,
                                box_normalization=self.box_normalization)


def detection_resize_only_image(image, size, max_size):

    h = image.size(1)
    w = image.size(2)

    # 2. get resize size
    if isinstance(size, (list, tuple)):
        size = size
    else:
        if max_size is not None:
            min_original_size = float(min((h, w)))
            max_original_size = float(max((h, w)))

            # e.g) 800 ~ 1333
            # 작은값을 800으로 맞추었을때의 큰값이 1333 을 넘으면,
            if size / min_original_size * max_original_size > max_size:
                # 큰 값을 1333 으로 맞추었을때의 작은값을 size로 정한다. (더 작아짐)
                size = int(round(max_size / max_original_size * min_original_size))

        # 3. get aspect_ratio
        if (w <= h and w == size) or (h <= w and h == size):
            size = (h, w)
        else:
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            size = (oh, ow)

    rescaled_image = F.resize(image, size)
    return rescaled_image


class FRCNNResizeOnlyImage(object):
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, image):
        return detection_resize_only_image(image=image,
                                           size=self.size,
                                           max_size=self.max_size)