import os
import time
import torch
import numpy as np
from tqdm import tqdm
from evaluation.evaluator import Evaluator


@torch.no_grad()
def test_and_eval(epoch, device, vis, test_loader, model, criterion, opts, xl_log_saver=None, result_best=None):

    if opts.rank == 0:

        # 0. evaluator
        evaluator = Evaluator(data_type=opts.data_type)
        print('Validation of epoch [{}]'.format(epoch))
        model.eval()
        local_gpu_id = int(opts.gpu_ids[opts.rank])

        # 1. load .pth
        checkpoint = torch.load(f=os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(epoch)),
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        tic = time.time()
        sum_loss = []

        # 2.
        if opts.data_type == 'coco':
            print('COCO dataset evaluation...')
        else:
            print('VOC dataset evaluation...')

        for idx, data in enumerate(tqdm(test_loader)):

            images = data[0]
            boxes = data[1]
            labels = data[2]

            # ---------- cuda ----------
            images = images.to(local_gpu_id)
            boxes = [b.to(local_gpu_id) for b in boxes]
            labels = [l.to(local_gpu_id) for l in labels]
            anchors = model.module.anchors.to(local_gpu_id)

            # ---------- loss ----------
            pred = model(images)
            loss, (cls_loss, loc_loss) = criterion(pred, boxes, labels, anchors)
            sum_loss.append(loss.item())

            # ---------- predict ----------
            pred_boxes, pred_labels, pred_scores = model.module.predict(pred[0], pred[1], anchors, opts)

            if opts.data_type == 'voc':

                info = data[3][0]  # [{}]
                info = (pred_boxes, pred_labels, pred_scores, info['name'], info['original_wh'])

            elif opts.data_type == 'coco':

                img_id = test_loader.dataset.img_id[idx]
                img_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
                coco_ids = test_loader.dataset.coco_ids
                info = (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

            evaluator.get_info(info)
            toc = time.time()

            # ---------- print ----------
            if idx % opts.vis_step == 0 or idx == len(test_loader) - 1:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(test_loader),
                              loss=loss,
                              time=toc - tic))

        mAP = evaluator.evaluate(test_loader.dataset)
        mean_loss = np.array(sum_loss).mean()
        print("mAP : ", mAP)
        print("mean Loss : ", mean_loss)
        print("Eval Time : {:.4f}".format(time.time() - tic))
        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss and map for {}'.format(opts.name),
                               legend=['test Loss', 'mAP']))

        if xl_log_saver is not None:
            xl_log_saver.insert_each_epoch(contents=(epoch, mAP, mean_loss))

        # save best.pth.tar
        if result_best is not None:
            if result_best['mAP'] < mAP:
                print("update best model")
                result_best['epoch'] = epoch
                result_best['mAP'] = mAP
                torch.save(checkpoint, os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.best.pth.tar'))

            return result_best


if __name__ == "__main__":
    from dataset.build import build_dataloader
    from models.build import build_model
    from loss import RetinaLoss
    import configargparse
    from config import get_args_parser

    parser = configargparse.ArgumentParser('Retinanet testing', parents=[get_args_parser()])
    opts = parser.parse_args()

    # 2. device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = None

    # 4. dataloader
    _, test_loader = build_dataloader(opts)

    # 5. network
    model = build_model(opts).to(device)

    # 6. loss
    criterion = RetinaLoss(opts)

    # 7. loss
    test_and_eval(epoch=opts.test_epoch,
                  device=device,
                  vis=vis,
                  test_loader=test_loader,
                  model=model,
                  criterion=criterion,
                  opts=opts)







