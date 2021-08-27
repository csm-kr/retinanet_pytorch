import os
import time
import torch
from utils import detect
from evaluator import Evaluator
from config import device, device_ids


def test(epoch, vis, test_loader, model, criterion, coder, opts):

    # ---------- load ----------
    print('Validation of epoch [{}]'.format(epoch))
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch),
                             map_location=device)
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)

    tic = time.time()
    sum_loss = 0

    is_coco = hasattr(test_loader.dataset, 'coco')  # if True the set is COCO else VOC
    if is_coco:
        print('COCO dataset evaluation...')
    else:
        print('VOC dataset evaluation...')

    evaluator = Evaluator(data_type=opts.data_type)

    with torch.no_grad():

        for idx, datas in enumerate(test_loader):

            images = datas[0]
            boxes = datas[1]
            labels = datas[2]

            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # ---------- loss ----------
            pred = model(images)
            loss, (cls_loss, loc_loss) = criterion(pred, boxes, labels)

            sum_loss += loss.item()

            # ---------- eval ----------
            pred_boxes, pred_labels, pred_scores = detect(pred=pred,
                                                          coder=coder,
                                                          opts=opts)

            if opts.data_type == 'voc':
                img_name = datas[3][0]
                img_info = datas[4][0]
                info = (pred_boxes, pred_labels, pred_scores, img_name, img_info)

            elif opts.data_type == 'coco':
                img_id = test_loader.dataset.img_id[idx]
                img_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
                coco_ids = test_loader.dataset.coco_ids
                info = (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

            evaluator.get_info(info)

            toc = time.time()

            # ---------- print ----------
            if idx % 1000 == 0 or idx == len(test_loader) - 1:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(test_loader),
                              loss=loss,
                              time=toc - tic))

        mAP = evaluator.evaluate(test_loader.dataset)
        mean_loss = sum_loss / len(test_loader)

        print(mAP)
        print("Eval Time : {:.4f}".format(time.time() - tic))

        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss',
                               legend=['test Loss', 'mAP']))


if __name__ == "__main__":

    from dataset.voc_dataset import VOC_Dataset
    from dataset.coco_dataset import COCO_Dataset
    from loss import RetinaLoss
    from model import Resnet50, RetinaNet
    from coder import RETINA_Coder
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=58)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='retina_res_50_coco')
    parser.add_argument('--conf_thres', type=float, default=0.05)

    # parser.add_argument('--data_root', type=str, default='D:\data\\voc')
    # parser.add_argument('--data_root', type=str, default='D:\data\coco')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/voc')
    parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/coco')

    parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')
    parser.add_argument('--resize', type=int, default=600, help='image_size')
    parser.add_argument('--num_classes', type=int, default=80)

    test_opts = parser.parse_args()
    print(test_opts)

    # 2. device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = None

    # 4. data set
    if test_opts.data_type == 'voc':
        test_set = VOC_Dataset(root=test_opts.data_root, split='test', resize=600)
        test_opts.num_classes = 20

    if test_opts.data_type == 'coco':
        test_set = COCO_Dataset(root=test_opts.data_root, set_name='val2017', split='test', resize=600)
        test_opts.num_classes = 80

    # 5. data loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=0)
    # 6. network
    model = RetinaNet(num_classes=test_opts.num_classes).to(device)
    model = torch.nn.DataParallel(module=model, device_ids=device_ids)
    coder = RETINA_Coder(opts=test_opts)

    # 7. loss
    criterion = RetinaLoss(coder)

    test(epoch=test_opts.epoch,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         coder=coder,
         opts=test_opts)







