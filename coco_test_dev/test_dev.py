import os
import time
import torch
from utils import detect
from evaluator import Evaluator
import json


def test_dev(epoch, test_loader, model, coder, opts):

    # testing
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch),
                             map_location=device)
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict, strict=True)

    tic = time.time()

    is_coco = hasattr(test_loader.dataset, 'coco')  # if True the set is COCO else VOC
    if is_coco:
        print('COCO dataset evaluation...')

    evaluator = Evaluator(data_type=opts.data_type)

    with torch.no_grad():

        for idx, datas in enumerate(test_loader):

            images = datas[0]
            images = images.to(device)
            pred = model(images)
            # eval
            pred_boxes, pred_labels, pred_scores = detect(pred=pred,
                                                          coder=coder,
                                                          opts=opts)

            img_id = test_loader.dataset.img_id[idx]
            img_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
            coco_ids = test_loader.dataset.coco_ids
            info = (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

            evaluator.get_info(info)
            toc = time.time()

            if idx % 1000 == 0 or idx == len(test_loader) - 1:
                print('Step: [{0}/{1}]\t'
                      'Test Time : {time:.4f}\t'
                      .format(idx,
                              len(test_loader),
                              time=toc - tic))

        json.dump(evaluator.results, open('detections_{}_{}_results.json'.format('test-dev2019', opts.submit_name), "w"))


if __name__ == '__main__':

    from coco_test_dev.coco_test_dev_dataset import COCO_Test_Dev_Dataset
    from config import device, device_ids
    from model import RetinaNet
    from coder import RETINA_Coder
    import argparse

    # 1. argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=58)
    parser.add_argument('--save_path', type=str, default='../saves')
    parser.add_argument('--save_file_name', type=str, default='retina_res_50_coco')
    parser.add_argument('--conf_thres', type=float, default=0.05)

    parser.add_argument('--data_root', type=str, default='D:\data\coco')
    #parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/coco')

    parser.add_argument('--resize', type=int, default=600)
    parser.add_argument('--visualization', type=bool, default=False)
    parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')
    parser.add_argument('--num_classes', type=int, default=80)

    parser.add_argument('--submit_name', type=str, default='sm')  # FIXME
    test_opts = parser.parse_args()
    print(test_opts)

    # 3. visdom
    vis = None

    # 4. data set
    test_set = COCO_Test_Dev_Dataset(root=test_opts.data_root, set_name='test2017', split='test', download=True, resize=test_opts.resize)
    test_opts.n_classes = 80

    # 5. data loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=0)

    # 6. network
    model = RetinaNet(num_classes=test_opts.num_classes).to(device)
    model = torch.nn.DataParallel(module=model, device_ids=device_ids)
    coder = RETINA_Coder(test_opts)

    test_dev(epoch=test_opts.epoch,
             test_loader=test_loader,
             model=model,
             coder=coder,
             opts=test_opts,
             )