import configargparse


def get_args_parser():
    parser = configargparse.ArgumentParser(add_help=False)

    # config
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--name", type=str, help='experiment name')

    # visualization
    parser.add_argument('--visdom_port', type=int, default=9999)
    parser.add_argument('--vis_step', type=int, default=100)

    # log
    parser.add_argument("--log_dir", type=str, default='./logs')

    # resume
    parser.add_argument("--start_epoch", type=int, default=0)

    # data
    parser.add_argument("--batch_size", type=int)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--num_workers', type=int, default=0)

    # model
    parser.add_argument('--resize', type=int)
    parser.add_argument('--num_classes', type=int)

    # train [optimizer (SGD)]
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # test
    parser.add_argument('--thres', type=float, default=0.05)
    parser.add_argument('--test_epoch', type=str, default='best')

    # demo
    parser.add_argument('--demo_root', type=str)
    parser.add_argument('--demo_epoch', type=str, default='best')
    parser.add_argument('--demo_image_type', type=str)
    parser.set_defaults(demo_vis=False)
    parser.add_argument('--demo_vis_true', dest='demo_vis', action='store_true')

    # distributed
    parser.set_defaults(distributed=False)
    parser.add_argument('--distributed_true', dest='distributed', action='store_true')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--gpu_ids', nargs="+")
    parser.add_argument('--world_size', type=int, default=0)

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    opts = parser.parse_args()
    print(opts)
