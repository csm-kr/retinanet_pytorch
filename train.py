import time
import os
import torch
import torch.distributed as dist
from config import device


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if hasattr(param.grad, 'data'):
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts):
    if opts.rank == 0:
        print('Training of epoch [{}]'.format(epoch))
    tic = time.time()
    model.train()
    model.module.freeze_bn()  # as attach module, we use data parallel

    for idx, datas in enumerate(train_loader):

        images = datas[0]
        boxes = datas[1]
        labels = datas[2]

        images = images.to(opts.rank)
        boxes = [b.to(opts.rank) for b in boxes]
        labels = [l.to(opts.rank) for l in labels]

        pred = model(images)
        loss, (cls_loss, loc_loss) = criterion(pred, boxes, labels)

        # sgd
        optimizer.zero_grad()
        loss.backward()
        if model.__class__.__name__ == 'DistributedDataParallel':
            # https://tutorials.pytorch.kr/intermediate/dist_tuto.html
            average_gradients(model)
        optimizer.step()

        toc = time.time()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if (idx % opts.vis_step == 0 or idx == len(train_loader) - 1) and opts.rank == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Cls_loss: {cls_loss:.4f}\t'
                  'Loc_loss: {loc_loss:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader),
                          loss=loss,
                          cls_loss=cls_loss,
                          loc_loss=loc_loss,
                          lr=lr,
                          time=toc - tic))

            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 3)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, cls_loss, loc_loss]).unsqueeze(0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss', 'Cls Loss', 'Loc Loss']))

    # # 각 epoch 마다 저장
    if opts.rank == 0:
        if not os.path.exists(opts.save_path):
            os.mkdir(opts.save_path)

        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}

        torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))



