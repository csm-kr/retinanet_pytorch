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

# for distributed_training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# use cudnn auto-tuner for own hard-ware
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def main_worker(rank, world_size, DDP_=True):
    """
    rank : gpu 번호
    world_size : gpu 총 갯수
    """
    # 1. argparser
    opts = parse(sys.argv[1:])
    print(opts)
    # start_gpu_num = 2
    gpu_id = int(opts.gpu_ids[rank])
    opts.gpu_id = gpu_id
    if DDP_:
        # 2. rank setting
        opts.rank = rank
        torch.cuda.set_device(gpu_id)
        if opts.rank is not None:
            print("Use GPU: {} for training".format(gpu_id))
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:3456',
                                world_size=world_size,
                                rank=rank)
        torch.distributed.barrier()

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

    # for DDP
    if DDP_:
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=int(opts.batch_size/world_size),
                                                   collate_fn=train_set.collate_fn,
                                                   shuffle=False,
                                                   num_workers=int(opts.num_workers/world_size),
                                                   pin_memory=True,
                                                   sampler=DistributedSampler(dataset=train_set, shuffle=True))
    else:
        # for DP
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=opts.batch_size,
                                                   collate_fn=train_set.collate_fn,
                                                   shuffle=True,
                                                   num_workers=opts.num_workers,
                                                   pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True)

    # 6. network

    # IF DDP
    if DDP_:
        model = RetinaNet(num_classes=opts.num_classes)
        model = model.cuda(gpu_id)
        model = DDP(module=model,
                    device_ids=[gpu_id],
                    find_unused_parameters=True)

    else:
        # IF DP
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

    # 9. scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[8, 11], gamma=0.1)   # 8, 11

    # 10. resume
    if opts.start_epoch != 0:

        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1), map_location=torch.device('cuda:{}'.format(gpu_id)))
        # 하나 적은걸 가져와서 train
        model.load_state_dict(checkpoint['model_state_dict'])                              # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                      # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])                      # load sched state dict
        if opts.rank == 0:
            print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))

    else:
        if opts.rank == 0:
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


def main():
    # for DP
    # main_worker(0, 2, DDP_=False)

    # for DDP
    world_size = torch.cuda.device_count()   # 4
    # world_size = 3
    mp.spawn(main_worker,
             args=(world_size, ),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()



