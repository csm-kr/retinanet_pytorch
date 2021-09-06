import numpy as np
import torch
from abc import ABCMeta, abstractmethod
from config import device
from utils import cxcy_to_xy, xy_to_cxcy


class Anchor(metaclass=ABCMeta):
    def __init__(self, model_name='yolo'):
        self.model_name = model_name.lower()
        assert model_name in ['yolo', 'ssd', 'retina']

    @abstractmethod
    def create_anchors(self):
        pass


class RETINA_Anchor(Anchor):
    def create_anchors(self, img_size):

        print('make retina anchors')
        pyramid_levels = np.array([3, 4, 5, 6, 7])
        feature_maps = [(img_size + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]  # convolutional  filter

        # eg.
        # 600 -  [ 75, 38, 19, 10, 5], stride - [8, 15, 31, 60, 120], area - [32, 60, 124, 240, 480]
        # 800 -  [100, 50, 25, 13, 7], stride - [8, 16, 32, 61, 114], area - [32, 64, 128, 244, 456]
        # 1024 - [128, 64, 32, 16, 8], stride - [8, 16, 32, 64, 128], area - [32, 64, 128, 256, 512]

        aspect_ratios = np.array([1.0, 2.0, 0.5])
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        # refer to https://github.com/NVIDIA/retinanet-examples/blob/master/retinanet/box.py
        strides = [img_size//f for f in feature_maps]
        areas = [s * 4 for s in strides]  # 600 기준 - [32, 60, 124, 240, 480]

        # original paper areas
        # https://arxiv.org/abs/1708.02002 chapter 4's anchor
        areas = [32, 64, 128, 256, 512]

        center_anchors = []
        for f_map, area, stride in zip(feature_maps, areas, strides):
            for i in range(f_map):
                for j in range(f_map):
                    c_x = (j + 0.5) / f_map
                    c_y = (i + 0.5) / f_map

                    # if consider
                    # refer to https://github.com/yhenon/pytorch-retinanet
                    # c_x = (j + 0.5) * stride / img_size
                    # c_y = (i + 0.5) * stride / img_size

                    for aspect_ratio in aspect_ratios:
                        for scale in scales:
                            w = (area / img_size) * np.sqrt(aspect_ratio) * scale
                            h = (area / img_size) / np.sqrt(aspect_ratio) * scale

                            anchor = [c_x,
                                      c_y,
                                      w,
                                      h]
                            center_anchors.append(anchor)

        center_anchors = np.array(center_anchors).astype(np.float32)
        center_anchors = torch.FloatTensor(center_anchors).to(device)

        visualization = False
        if visualization:

            # original
            corner_anchors = cxcy_to_xy(center_anchors)

            # center anchor clamp 방식!
            corner_anchors = cxcy_to_xy(center_anchors).clamp(0, 1)
            center_anchors = xy_to_cxcy(corner_anchors)

            from matplotlib.patches import Rectangle
            import matplotlib.pyplot as plt

            size = 300
            img = torch.ones([size, size, 3], dtype=torch.float32)

            plt.imshow(img)
            axes = plt.axes()
            axes.set_xlim([- 1/3 * size, size])
            axes.set_ylim([- 1/3 * size, size])

            for anchor in corner_anchors[:10]:
                x1 = anchor[0] * size
                y1 = anchor[1] * size
                x2 = anchor[2] * size
                y2 = anchor[3] * size

                plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                              width=x2 - x1,
                                              height=y2 - y1,
                                              linewidth=1,
                                              edgecolor=[0, 1, 0],
                                              facecolor='none'
                                              ))
            plt.show()

        return center_anchors


if __name__ == '__main__':
    retina_anchor = RETINA_Anchor(model_name='retina')
    anchor = retina_anchor.create_anchors(img_size=600)

    center_anchor = anchor
    print(center_anchor)
    # corner_anchor = cxcy_to_xy(anchor) * 600
    # print(corner_anchor[:100, :])
    # print(anchor.size())


