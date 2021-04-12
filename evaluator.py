import os
import json
import tempfile
from voc_eval import voc_eval
from pycocotools.cocoeval import COCOeval


class Evaluator(object):
    def __init__(self, data_type='voc'):
        self.data_type = data_type

        # for VOC
        self.det_img_name = list()
        self.det_additional = list()
        self.det_boxes = list()
        self.det_labels = list()
        self.det_scores = list()

        # for COCO
        self.results = list()
        self.img_ids = list()

    def get_info(self, info):
        if self.data_type == 'voc':

            (pred_boxes, pred_labels, pred_scores, img_names, additional_info) = info

            self.det_img_name.append(img_names)  # 4952 len list # [1] - img_name_length [B, k]
            self.det_additional.append(additional_info)  # 4952 len list # [2] -  w, h   [B, 2]
            self.det_boxes.append(pred_boxes.cpu())  # 4952 len list # [obj, 4]
            self.det_labels.append(pred_labels.cpu())  # 4952 len list # [obj]
            self.det_scores.append(pred_scores.cpu())  # 4952 len list # [obj]

        elif self.data_type == 'coco':

            (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids) = info

            self.img_ids.append(img_id)

            # convert coco_results coordination
            pred_boxes[:, 2] -= pred_boxes[:, 0]  # x2 to w
            pred_boxes[:, 3] -= pred_boxes[:, 1]  # y2 to h

            w = img_info['width']
            h = img_info['height']

            pred_boxes[:, 0] *= w
            pred_boxes[:, 2] *= w
            pred_boxes[:, 1] *= h
            pred_boxes[:, 3] *= h

            for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                if int(pred_label) == 80:  # background label is 80
                    print('background label :', int(pred_label))
                    continue

                coco_result = {
                    'image_id': img_id,
                    'category_id': coco_ids[int(pred_label)],
                    'score': float(pred_score),
                    'bbox': pred_box.tolist(),
                }
                self.results.append(coco_result)

    def evaluate(self, dataset):
        if self.data_type == 'voc':

            test_root = os.path.join(dataset.root, 'VOCtest_06-Nov-2007', 'VOCdevkit', 'VOC2007', 'Annotations')
            mAP = voc_eval(test_root, self.det_img_name, self.det_additional, self.det_boxes, self.det_scores,
                           self.det_labels)

        elif self.data_type == 'coco':

            _, tmp = tempfile.mkstemp()
            json.dump(self.results, open(tmp, "w"))

            cocoGt = dataset.coco
            cocoDt = cocoGt.loadRes(tmp)

            # https://github.com/argusswift/YOLOv4-pytorch/blob/master/eval/cocoapi_evaluator.py
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.

            coco_eval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType='bbox')
            coco_eval.params.imgIds = self.img_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            mAP = coco_eval.stats[0]
            mAP_50 = coco_eval.stats[1]
        return mAP