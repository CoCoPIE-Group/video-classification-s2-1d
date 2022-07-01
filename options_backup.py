import argparse

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ADMM pruning for 3D CNNs')
    parser.add_argument('--logger', action='store_true', default=True,
                        help='whether to use logger')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--seed', type=int, default=2019, metavar='S',
                        help='random seed (default: 2019)')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='for multi-gpu training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--load-path', default='',
                        help='pretrained model path')

    parser.add_argument('--arch', type=str, default='r2+1d',
                        help='[c3d, r2+1d-pretrained, s3d]')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--admm-epochs', type=int, default=10, metavar='N',
                        help='number of interval epochs to update admm (default: 10)')

    parser.add_argument('--optmzr', type=str, default='sgd', metavar='OPTMZR',
                        help='optimizer used (default: adam)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr-decay', type=int, default=30, metavar='LR_decay',
                        help='how many every epoch before lr drop (default: 30)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--sparsity-type', type=str, default='blk-kgs',
                        help ='define sparsity_type: [blk-vanilla, blk-kgs]')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--config-file', type=str, default='c3d',
                        help ='config file name')
    parser.add_argument('--admm', action='store_true', default=False,
                        help='for admm training')
    parser.add_argument('--masked-retrain', action='store_true', default=False,
                        help='for masked retrain')
    parser.add_argument('--combine-progressive', action='store_true', default=False,
                        help='for progressive pruning')
    parser.add_argument('--rho', type=float, default = 0.0001,
                        help ='define rho for ADMM')
    parser.add_argument('--rho-num', type=int, default = 4,
                        help ='define how many rhos for ADMM training')
    parser.add_argument('--data_location', default='/raid10/ms/dataset',
                        help='training and testing data location')

    # Tricks
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        help='define lr scheduler')
    parser.add_argument('--warmup', action='store_true', default=False,
                        help='warm-up scheduler')
    parser.add_argument('--warmup-lr', type=float, default=0.00001, metavar='M',
                        help='warmup-lr, smaller than original lr')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='M',
                        help='number of epochs for lr warmup')
    parser.add_argument('--mixup', action='store_true', default=False,
                        help='ce mixup')
    parser.add_argument('--alpha', type=float, default=0.0, metavar='M',
                        help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
    parser.add_argument('--smooth', action='store_true', default=False,
                        help='lable smooth')
    parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                        help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
    parser.add_argument('--no-tricks', action='store_true', default=False,
                        help='disable all training tricks and restore original classic training process')

    ### customize
    parser.add_argument('--dataset',
                        default='ucf101')
    parser.add_argument('--resume',  action='store_true', default=False,
                        help='resume from last epoch if model exists')
    parser.add_argument('--block-size',
                        type=int, default=8,
                        help='block size in block circulant weight matrix')
    parser.add_argument('--connectivity-block-size', '-conn-blk',
                        nargs='*', type=int, default=[8, 4],
                        help='block size for connectivity pruning')

    # distillation
    parser.add_argument('--distill', action='store_true', default=False ,
                        help='if distillation')
    parser.add_argument('--teacharch', default=None,
                        help='teacher model architecture name')
    parser.add_argument('--teacher-path', default='',
                        help='the path of teacher model')
    parser.add_argument('--temperature', default=3, type=float,
                        help='temperature of distillation')
    parser.add_argument('--kd-coefficient', default=0.5, type=float,
                        help='loss coefficient of knowledge distillation')
    return parser

