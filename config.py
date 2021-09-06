import torch
import argparse

device_ids = [0, 1]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')


def parse(args):
    # 1. arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=13)                  # 13
    parser.add_argument('--port', type=str, default='2015')
    parser.add_argument('--lr', type=float, default=1e-2)                 # 1e-2
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)       # 0.0001
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--vis_step', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)

    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='retina_res_50_coco')                         # FIXME

    parser.add_argument('--conf_thres', type=float, default=0.05)
    parser.add_argument('--start_epoch', type=int, default=0)

    # FIXME choose your dataset root
    # parser.add_argument('--data_root', type=str, default='D:\data\\voc')
    # parser.add_argument('--data_root', type=str, default='D:\data\coco')
    parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/coco')
    parser.add_argument('--img_path', type=str, default='/home/cvmlserver5/Sungmin/data/coco/images/val2017')

    parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')              # FIXME
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--resize', type=int, default=600)                                               # FIXME

    parser.set_defaults(visualization=False)
    parser.add_argument('--vis', dest='visualization', action='store_true')

    opts = parser.parse_args(args)
    return opts