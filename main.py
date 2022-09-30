import os
import torch
import visdom

# dataset
from dataset.build import build_dataloader

# model
from models.build import build_model

# loss
from loss import RetinaLoss

from train import train_one_epoch
from test import test_and_eval
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

# log
from log import XLLogSaver
# resume
from utils import resume

# for distributed_training
import torch.distributed as dist
import torch.multiprocessing as mp

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import configargparse
from config import get_args_parser


def main_worker(rank, opts):
    """
    rank : gpu 번호
    world_size : gpu 총 갯수
    """
    # 1. config
    print(opts)

    # 2. distributed
    if opts.distributed:
        init_for_distributed(rank, opts)

    # 3. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 4. visdom
    vis = visdom.Visdom(port=opts.visdom_port)

    # 5. data set / loader
    train_loader, test_loader = build_dataloader(opts)

    # 6. network
    model = build_model(opts)

    # 7. loss
    criterion = RetinaLoss(opts)

    # 8. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=opts.momentum,
                                weight_decay=opts.weight_decay)

    # 9. scheduler
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=opts.epoch, eta_min=0.00005)

    # 9. logger
    xl_log_saver = None
    if opts.rank == 0:
        xl_log_saver = XLLogSaver(xl_folder_name=os.path.join(opts.log_dir, opts.name),
                                  xl_file_name=opts.name,
                                  tabs=('epoch', 'mAP', 'val_loss'))

    # 10. resume
    model, optimizer, scheduler = resume(opts, model, optimizer, scheduler)

    # set best results
    result_best = {'epoch': 0, 'mAP': 0., 'val_loss': 0.}

    # for statement
    for epoch in range(opts.start_epoch, opts.epoch):

        if opts.distributed:
            train_loader.sampler.set_epoch(epoch)

        # 11. train
        train_one_epoch(epoch=epoch,
                        vis=vis,
                        train_loader=train_loader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        opts=opts)

        # # 12. test
        result_best = test_and_eval(epoch=epoch,
                                    device=device,
                                    vis=vis,
                                    test_loader=test_loader,
                                    model=model,
                                    criterion=criterion,
                                    opts=opts,
                                    xl_log_saver=xl_log_saver,
                                    result_best=result_best)
        scheduler.step()


def init_for_distributed(rank, opts):

    # 1. setting for distributed training
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=opts.world_size,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print(opts)
    return


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


if __name__ == "__main__":

    parser = configargparse.ArgumentParser('Retinanet training', parents=[get_args_parser()])
    opts = parser.parse_args()

    if len(opts.gpu_ids) > 1:
        opts.distributed = True

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    if opts.distributed:
        mp.spawn(main_worker,
                 args=(opts,),
                 nprocs=opts.world_size,
                 join=True)
    else:
        main_worker(opts.rank, opts)



