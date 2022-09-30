import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import platform
import os
import wget
import glob
import zipfile
from utils import bar_custom, coco_color_array


def download_coco(root_dir='D:\data\\coco', remove_compressed_file=True):
    # for coco 2017
    coco_2017_train_url = 'http://images.cocodataset.org/zips/train2017.zip'
    coco_2017_val_url = 'http://images.cocodataset.org/zips/val2017.zip'
    coco_2017_test_url = 'http://images.cocodataset.org/zips/test2017.zip'
    coco_2017_trainval_anno_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    os.makedirs(root_dir, exist_ok=True)

    img_dir = os.path.join(root_dir, 'images')
    anno_dir = os.path.join(root_dir, 'annotations')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    """Download the VOC data if it doesn't exit in processed_folder already."""

    # if (os.path.exists(os.path.join(img_dir, 'train2017')) and
    #         os.path.exists(os.path.join(img_dir, 'val2017')) and
    #         os.path.exists(os.path.join(img_dir, 'test2017'))):
    #
    if (os.path.exists(os.path.join(img_dir, 'train2017')) and
            os.path.exists(os.path.join(img_dir, 'val2017'))):

        print("Already exist!")
        return

    print("Download...")

    # image download
    wget.download(url=coco_2017_train_url, out=img_dir, bar=bar_custom)
    print('')
    wget.download(url=coco_2017_val_url, out=img_dir, bar=bar_custom)
    print('')
    # wget.download(url=coco_2017_test_url, out=img_dir, bar=bar_custom)
    # print('')

    # annotation download
    wget.download(coco_2017_trainval_anno_url, out=root_dir, bar=bar_custom)
    print('')

    print("Extract...")

    # image extract
    with zipfile.ZipFile(os.path.join(img_dir, 'train2017.zip')) as unzip:
        unzip.extractall(os.path.join(img_dir))
    with zipfile.ZipFile(os.path.join(img_dir, 'val2017.zip')) as unzip:
        unzip.extractall(os.path.join(img_dir))
    # with zipfile.ZipFile(os.path.join(img_dir, 'test2017.zip')) as unzip:
    #     unzip.extractall(os.path.join(img_dir))

    # annotation extract
    with zipfile.ZipFile(os.path.join(root_dir, 'annotations_trainval2017.zip')) as unzip:
        unzip.extractall(os.path.join(root_dir))

    # remove zips
    if remove_compressed_file:
        root_zip_list = glob.glob(os.path.join(root_dir, '*.zip'))  # in root_dir remove *.zip
        for anno_zip in root_zip_list:
            os.remove(anno_zip)

        img_zip_list = glob.glob(os.path.join(img_dir, '*.zip'))  # in img_dir remove *.zip
        for img_zip in img_zip_list:
            os.remove(img_zip)
        print("Remove *.zips")

    print("Done!")


# COCO_Dataset
class COCO_Dataset(Dataset):
    def __init__(self,
                 root='D:\Data\coco',
                 split='train',
                 download=True,
                 transform=None,
                 visualization=False):
        super().__init__()

        if platform.system() == 'Windows':
            matplotlib.use('TkAgg')  # for window

        # -------------------------- set root --------------------------
        self.root = root

        # -------------------------- set split --------------------------
        assert split in ['train', 'val', 'test']
        self.split = split
        self.set_name = split + '2017'

        # -------------------------- download --------------------------
        self.download = download
        if self.download:
            download_coco(root_dir=root)

        # -------------------------- transform --------------------------
        self.transform = transform

        # -------------------------- visualization --------------------------
        self.visualization = visualization

        self.img_path = glob.glob(os.path.join(self.root, 'images', self.set_name, '*.jpg'))
        self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_' + self.set_name + '.json'))

        self.img_id = list(self.coco.imgToAnns.keys())
        # self.ids = self.coco.getImgIds()

        self.coco_ids = sorted(self.coco.getCatIds())  # list of coco labels [1, ...11, 13, ... 90]  # 0 ~ 79 to 1 ~ 90
        self.coco_ids_to_continuous_ids = {coco_id: i for i, coco_id in enumerate(self.coco_ids)}  # 1 ~ 90 to 0 ~ 79
        # int to int
        self.coco_ids_to_class_names = {category['id']: category['name'] for category in
                                        self.coco.loadCats(self.coco_ids)}  # len 80
        # int to string
        # {1 : 'person', 2: 'bicycle', ...}
        '''
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        '''

    def __getitem__(self, index):

        # --------------------------- load data ---------------------------
        # 1. image_id
        img_id = self.img_id[index]

        # 2. load image
        img_coco = self.coco.loadImgs(ids=img_id)[0]
        file_name = img_coco['file_name']
        file_path = os.path.join(self.root, 'images', self.set_name, file_name)

        # eg. 'D:\\Data\\coco\\images\\val2017\\000000289343.jpg'
        image = Image.open(file_path).convert('RGB')

        # 3. load anno
        anno_ids = self.coco.getAnnIds(imgIds=img_id)  # img id 에 해당하는 anno id 를 가져온다.
        anno = self.coco.loadAnns(ids=anno_ids)        # anno id 에 해당하는 annotation 을 가져온다.

        det_anno = self.make_det_annos(anno)           # anno -> [x1, y1, x2, y2, c] numpy 배열로

        boxes = torch.FloatTensor(det_anno[:, :4])     # numpy to Tensor
        labels = torch.LongTensor(det_anno[:, 4])

        # --------------------------- for transform ---------------------------
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)
        # print("boxes:", boxes)

        if self.visualization:
            # ----------------- visualization -----------------
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # tensor to img
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            img_vis *= std
            img_vis += mean
            img_vis = np.clip(img_vis, 0, 1)

            plt.figure('input')
            plt.imshow(img_vis)
            print('num objects : {}'.format(len(boxes)))

            for i in range(len(boxes)):

                new_h_scale = new_w_scale = 1
                # box_normalization of DetResize
                if self.transform.transforms[-2].box_normalization:
                    new_h_scale, new_w_scale = image.size()[1:]

                x1 = boxes[i][0] * new_w_scale
                y1 = boxes[i][1] * new_h_scale
                x2 = boxes[i][2] * new_w_scale
                y2 = boxes[i][3] * new_h_scale

                # print(boxes[i], ':', self.coco_ids_to_class_names[self.coco_ids[labels[i]]])

                # class
                plt.text(x=x1 - 5,
                         y=y1 - 5,
                         s=str(self.coco_ids_to_class_names[self.coco_ids[labels[i]]]),
                         bbox=dict(boxstyle='round4',
                                   facecolor=coco_color_array[labels[i]],
                                   alpha=0.9))

                # bounding box
                plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                              width=x2 - x1,
                                              height=y2 - y1,
                                              linewidth=1,
                                              edgecolor=coco_color_array[labels[i]],
                                              facecolor='none'))

            plt.show()

        return image, boxes, labels

    def make_det_annos(self, anno):

        annotations = np.zeros((0, 5))
        for idx, anno_dict in enumerate(anno):

            if anno_dict['bbox'][2] < 1 or anno_dict['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = anno_dict['bbox']

            annotation[0, 4] = self.coco_ids_to_continuous_ids[anno_dict['category_id']]  # 원래 category_id가 18이면 들어가는 값은 16
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, difficulties, img_name and
        additional_info
        """
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)
        return images, boxes, labels

    def __len__(self):
        return len(self.img_id)


if __name__ == '__main__':

    device = torch.device('cuda:0')
    import torchvision.transforms as transforms
    import dataset.detection_transforms as det_transforms

    transform_train = det_transforms.DetCompose([
        # ------------- for Tensor augmentation -------------
        det_transforms.DetRandomPhotoDistortion(),
        det_transforms.DetRandomHorizontalFlip(),
        det_transforms.DetToTensor(),
        # ------------- for Tensor augmentation -------------
        det_transforms.DetRandomZoomOut(max_scale=3),
        det_transforms.DetRandomZoomIn(),
        det_transforms.DetResize(size=(600, 600), box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    transform_test = det_transforms.DetCompose([
        det_transforms.DetToTensor(),
        det_transforms.DetResize(size=800, max_size=1333, box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    coco_dataset = COCO_Dataset(root="D:/data/coco",
                                split='train',
                                download=True,
                                transform=transform_test,
                                visualization=True)

    train_loader = torch.utils.data.DataLoader(coco_dataset,
                                               batch_size=1,
                                               collate_fn=coco_dataset.collate_fn,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    for i, data in enumerate(train_loader):
        images = data[0]
        boxes = data[1]
        labels = data[2]

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        # print(labels)
