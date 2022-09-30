import os
import wget
import glob
import torch
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
from xml.etree.ElementTree import parse
from matplotlib.patches import Rectangle
from utils import bar_custom, voc_color_array


def download_voc(root_dir='D:\data\\voc', remove_compressed_file=True):

    voc_2012_train_url = 'https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
    voc_2007_train_url = 'https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar'
    voc_2007_test_url = 'https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar'

    os.makedirs(root_dir, exist_ok=True)

    """Download the VOC data if it doesn't exist in processed_folder already."""
    if (os.path.exists(os.path.join(root_dir, 'VOCtrainval_11-May-2012')) and
        os.path.exists(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007')) and
        os.path.exists(os.path.join(root_dir, 'VOCtest_06-Nov-2007'))):
        print("Already exist!")
        return

    print("Download...")

    wget.download(url=voc_2012_train_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(url=voc_2007_train_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(url=voc_2007_test_url, out=root_dir, bar=bar_custom)
    print('')

    os.makedirs(os.path.join(root_dir, 'VOCtrainval_11-May-2012'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'VOCtest_06-Nov-2007'), exist_ok=True)

    print("Extract...")

    with tarfile.open(os.path.join(root_dir, 'VOCtrainval_11-May-2012.tar')) as tar:
        tar.extractall(os.path.join(root_dir, 'VOCtrainval_11-May-2012'))
    with tarfile.open(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007.tar')) as tar:
        tar.extractall(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007'))
    with tarfile.open(os.path.join(root_dir, 'VOCtest_06-Nov-2007.tar')) as tar:
        tar.extractall(os.path.join(root_dir, 'VOCtest_06-Nov-2007'))

    # remove tars
    if remove_compressed_file:
        root_zip_list = glob.glob(os.path.join(root_dir, '*.tar'))  # in root_dir remove *.zip
        for root_zip in root_zip_list:
            os.remove(root_zip)
        print("Remove *.tars")

    print("Done!")


class VOC_Dataset(data.Dataset):

    # not background for coco
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    """
    voc dataset
    """
    def __init__(self,
                 root='D:\data\\voc',
                 split='train',
                 download=True,
                 transform=None,
                 visualization=True):
        super(VOC_Dataset, self).__init__()

        self.root = root
        # -------------------------- set split --------------------------
        assert split in ['train', 'test']
        self.split = split

        # -------------------------- download --------------------------
        self.download = download
        if self.download:
            download_voc(root_dir=root)

        # -------------------------- transform --------------------------
        self.transform = transform

        # -------------------------- visualization --------------------------
        self.visualization = visualization

        # -------------------------- data setting --------------------------
        self.data_list = []
        self.img_list = []
        self.anno_list = []

        for i in os.listdir(self.root):
            if i.find('.tar') == -1 and i.find(self.split) != -1:  # split 이 포함된 data - except .tar 제외
                self.data_list.append(i)

        for data_ in self.data_list:

            self.img_list.extend(glob.glob(os.path.join(os.path.join(self.root, data_), '*/*/JPEGImages/*.jpg')))
            self.anno_list.extend(glob.glob(os.path.join(os.path.join(self.root, data_), '*/*/Annotations/*.xml')))
            # only voc 2007
            # self.img_list.extend(glob.glob(os.path.join(os.path.join(self.root, data_), '*/VOC2007/JPEGImages/*.jpg')))
            # self.anno_list.extend(glob.glob(os.path.join(os.path.join(self.root, data_), '*/VOC2007/Annotations/*.xml')))

        # for debug
        self.img_list = sorted(self.img_list)
        self.anno_list = sorted(self.anno_list)

        self.class_idx_dict = {class_name: i for i, class_name in enumerate(self.class_names)}     # class name : idx
        self.idx_class_dict = {i: class_name for i, class_name in enumerate(self.class_names)}     # idx : class name

    def __getitem__(self, idx):

        # load img
        image = Image.open(self.img_list[idx]).convert('RGB')
        # print(self.img_list[idx])
        # load labels
        boxes, labels = self.parse_voc(self.anno_list[idx])
        # print(boxes)
        # print(self.img_list[idx])
        # issue : diff 1인 친구들

        # load img name for string
        img_name = os.path.basename(self.anno_list[idx]).split('.')[0]
        img_name_to_ascii = [ord(c) for c in img_name]

        # load img width and height
        img_width, img_height = float(image.size[0]), float(image.size[1])

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)  # 0 ~ 19
        info = {}
        info['name'] = img_name
        info['original_wh'] = [int(img_width), int(img_height)]
        # --------------------------- for transform ---------------------------
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.visualization:
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

                # class
                plt.text(x=x1 - 5,
                         y=y1 - 5,
                         s=str(self.idx_class_dict[labels[i].item()]),   # FIXME
                         bbox=dict(boxstyle='round4',
                                   facecolor=voc_color_array[labels[i]],
                                   alpha=0.9))

                # bounding box
                plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                              width=x2 - x1,
                                              height=y2 - y1,
                                              linewidth=1,
                                              edgecolor=voc_color_array[labels[i]],
                                              facecolor='none'))

            plt.show()
        if self.split == "test":
            return image, boxes, labels, info

        return image, boxes, labels

    def __len__(self):
        return len(self.img_list)

    def parse_voc(self, xml_file_path):

        tree = parse(xml_file_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.iter("object"):

            # if int(obj.find('difficult').text) == 1:
            #     # difficult.append(int(obj.find('difficult').text))
            #     continue

            # 'name' tag 에서 멈추기
            name = obj.find('./name')
            class_name = name.text.lower().strip()
            labels.append(self.class_idx_dict[class_name])

            # bbox tag 에서 멈추기
            bbox = obj.find('./bndbox')
            x_min = bbox.find('./xmin')
            y_min = bbox.find('./ymin')
            x_max = bbox.find('./xmax')
            y_max = bbox.find('./ymax')

            # from str to int
            x_min = float(x_min.text) - 1
            y_min = float(y_min.text) - 1
            x_max = float(x_max.text) - 1
            y_max = float(y_max.text) - 1

            boxes.append([x_min, y_min, x_max, y_max])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        boxes = list()
        labels = list()
        info = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            if self.split == "test":
                info.append(b[3])

        images = torch.stack(images, dim=0)
        if self.split == "test":
            return images, boxes, labels, info
        return images, boxes, labels


if __name__ == "__main__":
    import torchvision.transforms as transforms
    device = torch.device('cuda:0')
    import dataset.detection_transforms as det_transforms

    # train_transform
    # ubuntu_root = "/home/cvmlserver3/Sungmin/data/voc"
    window_root = 'D:\data\\voc'
    # for test
    # window_root = r'C:\Users\csm81\Desktop\\voc_temp'
    root = window_root

    transform_train = det_transforms.DetCompose([
        # ------------- for Tensor augmentation -------------
        # det_transforms.DetRandomPhotoDistortion(),
        det_transforms.DetRandomHorizontalFlip(),
        det_transforms.DetToTensor(),
        # ------------- for Tensor augmentation -------------
        # det_transforms.DetRandomZoomOut(max_scale=3),
        # det_transforms.DetRandomZoomIn(),
        det_transforms.DetResize(size=600, max_size=1333, box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    transform_test = det_transforms.DetCompose([
        det_transforms.DetToTensor(),
        det_transforms.DetResize(size=800, max_size=1333, box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    train_set = VOC_Dataset(root,
                            split='test',
                            download=False,
                            transform=transform_train,
                            visualization=True)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=1,
                                               collate_fn=train_set.collate_fn,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    print(len(train_loader))

    for i, data in enumerate(train_loader):

        images = data[0]
        boxes = data[1]
        labels = data[2]

        # images = images.to(device)
        # boxes = [b.to(device) for b in boxes]
        # labels = [l.to(device) for l in labels]
        print(i)
        print(labels)
        print(boxes)


