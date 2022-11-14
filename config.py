import configargparse


def get_args_parser():
    # config
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--name", type=str, help='experiment name')                             # TODO

    # visdom
    parser.add_argument('--visdom_port', type=int)                                              # TODO
    parser.add_argument('--vis_step', type=int, default=100)

    # data
    parser.add_argument('--data_root', type=str)                                                # TODO
    parser.add_argument('--data_type', type=str)                                                # TODO
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resize', type=int, default=600)

    # model
    parser.add_argument('--num_classes', type=int)

    # train
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default='./logs')

    # test
    parser.add_argument('--test_epoch', type=str, default='best')
    parser.add_argument('--thres', type=float, default=0.05)
    parser.add_argument('--top_k', type=int, default=200, help='set top k for after nms')

    # demo
    parser.add_argument('--demo_epoch', type=str, default='best')
    parser.add_argument('--demo_root', type=str)                                               # TODO
    parser.add_argument('--demo_image_type', type=str)                                         # TODO
    parser.set_defaults(demo_vis=False)
    parser.add_argument('--demo_vis_true', dest='demo_vis', action='store_true')

    # distributed
    parser.add_argument('--distributed_true', dest='distributed', action='store_true')
    parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    opts = parser.parse_args()
    print(opts)
