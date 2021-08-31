import torch
import sys
from dataset.voc_dataset import VOC_Dataset
from dataset.coco_dataset import COCO_Dataset
from loss import RetinaLoss
import visdom
from train import train
from test import test
from torch.optim.lr_scheduler import MultiStepLR
from model import RetinaNet
import os
from config import device, device_ids, parse
from coder import RETINA_Coder

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def main():

    # 1. argparser
    opts = parse(sys.argv[1:])
    print(opts)

    # 3. visdom
    vis = visdom.Visdom(port=opts.port)
    # 4. data set
    train_set = None
    test_set = None

    if opts.data_type == 'voc':
        train_set = VOC_Dataset(root=opts.data_root, split='train', resize=opts.resize)
        test_set = VOC_Dataset(root=opts.data_root, split='test', resize=opts.resize)
        opts.num_classes = 20

    elif opts.data_type == 'coco':
        train_set = COCO_Dataset(root=opts.data_root, set_name='train2017', split='train', resize=opts.resize)
        test_set = COCO_Dataset(root=opts.data_root, set_name='val2017', split='test', resize=opts.resize)
        opts.num_classes = 80

    # 5. data loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=opts.batch_size,
                                               collate_fn=train_set.collate_fn,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True)

    # 6. network
    model = RetinaNet(num_classes=opts.num_classes).to(device)
    model = torch.nn.DataParallel(module=model, device_ids=device_ids)
    coder = RETINA_Coder(opts=opts)  # there is center_anchor in coder.

    # 7. loss
    criterion = RetinaLoss(coder=coder)

    # 8. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=opts.momentum,
                                weight_decay=opts.weight_decay)

    # optimizer = torch.optim.Adam(params=model.parameters(),
    #                              lr=opts.lr,
    #                              weight_decay=opts.weight_decay)

    # 9. scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[14, 18], gamma=0.1)

    # 10. resume
    if opts.start_epoch != 0:

        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1), map_location=device)        # 하나 적은걸 가져와서 train
        model.load_state_dict(checkpoint['model_state_dict'])                              # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                      # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])                      # load sched state dict
        print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))

    else:

        print('\nNo check point to resume.. train from scratch.\n')

    # for statement
    for epoch in range(opts.start_epoch, opts.epoch):

        # 11. train
        train(epoch=epoch,
              vis=vis,
              train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              opts=opts)

        # 12. test
        test(epoch=epoch,
             vis=vis,
             test_loader=test_loader,
             model=model,
             criterion=criterion,
             coder=coder,
             opts=opts)

        scheduler.step()


if __name__ == "__main__":
    main()



