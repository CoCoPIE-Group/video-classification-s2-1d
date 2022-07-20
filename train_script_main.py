import os
import sys
import time
import json
from time import strftime
from tqdm import tqdm  # progress bar
import numpy as np
# from custom_profile import profile_prune

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms

from network import c3d, s3d
from network.scaling_s21d import r2plus1d_scaling

from video_utils.utils import build_dataflow, get_augmentor, my_collate
from video_utils.video_dataset import VideoDataSet, VideoDataSetLMDB
from video_utils.dataset_config import get_dataset_config

from utils import *

import argparse

from xgen_tools import *
from xgen_tools import xgen_record, xgen_init, xgen_load, XgenArgs,xgen
from co_lib import Co_Lib as CL


parser = argparse.ArgumentParser(description='3D net training in PyTorch')
parser.add_argument('--arch', '-a',
                    choices=['c3d', 'r2+1d-pretrained'],
                    default='c3d')
parser.add_argument('--dataset', '-d',
                    choices=['ucf101', 'hmdb51', 'kinetics', 'mini-kinetics'],
                    default='ucf101')
parser.add_argument('--batch-size', '-b',
                    type=int, default=32)
parser.add_argument('--lr',
                    help='learning rate: default 5e-3 for sgd optimizer, 1e-4 for adam',
                    type=float, default=5e-3)
parser.add_argument('--optim',
                    help='optimizer',
                    choices=['sgd', 'adam'],
                    default='sgd')
parser.add_argument('--lr-scheduler',
                    help='learning rate scheduler',
                    choices=[None, 'cosine'],
                    default=None)
parser.add_argument('--lr-milestones',
                    help='epochs to decay learning rate by 10',
                    type=int, nargs='+', default=[20, 30])
parser.add_argument('--epochs',
                    help='num of epochs for training',
                    type=int, default=35)
parser.add_argument('--resume',
                    help='resume from last epoch if model exists',
                    action='store_true', default=False)
parser.add_argument('--transfer',
                    help='pretrain on kinetics, transfer on ucf101',
                    default=False, action='store_true')
parser.add_argument('--smooth-eps',
                    help='label smoothing, default=0.0 for none',
                    type=float, default=0.0)
parser.add_argument('--multiplier',
                    help='scaling multiplier',
                    type=float, default=8)
parser.add_argument('--pretrained',
                    help='load torchvision pretrained model',
                    action='store_true', default=False)
parser.add_argument('--log-interval',
                    help='log printing frequency',
                    type=int, default=50)
parser.add_argument('--test',
                    help='test model accuracy only',
                    action='store_true', default=False)
parser.add_argument('--test-path',
                    help='path of test model')
parser.add_argument('--gpu', default='1,2,3,4,5,6,7')
args = parser.parse_args()

# if args.gpu:
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if args.transfer:
    args.epochs = 20
    args.lr_milestones = [10, 15]
else:
    args.log_interval = 100
args.optim = args.optim.lower()
args.arch = args.arch.lower()
ckpt_name = '{}_{}{}'.format(args.dataset, args.arch, '_transfer' if args.transfer else '')
args.logdir = os.path.join('checkpoint', 'test') if args.test else os.path.join('checkpoint', ckpt_name)
if not args.resume and os.path.exists(args.logdir) and not args.test:
    i = 1
    while os.path.exists(args.logdir + '_v{}'.format(i)):
        i += 1
    os.rename(args.logdir, args.logdir + '_v{}'.format(i))
os.makedirs(args.logdir, exist_ok=True)

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(
    logging.FileHandler(os.path.join(args.logdir, '{}_{}.log'.format(ckpt_name, strftime('%m%d%Y-%H%M'))), 'w'))
global print
print = logger.info

use_cuda = torch.cuda.is_available()
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda: torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # will result in non-determinism

# reproduce results https://github.com/pytorch/pytorch/issues/7068
kwargs = {'num_workers': 0, 'worker_init_fn': np.random.seed(seed), 'pin_memory': True} if use_cuda else {}

# ROOT = '../dataset/'
# args.datadir = os.path.join(ROOT, args.dataset.replace('mini-', '') + '_frame')

print(' '.join(sys.argv))

if args.test:
    args.transfer = False
    args.resume = False

print('Training arguments: ')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))


COCOPIE_MAP={
    'epochs': XgenArgs.cocopie_train_epochs,
    'datadir': XgenArgs.cocopie_train_data_path
}

def training_main(args_ai):
    global args
    args = xgen_init(args,args_ai,COCOPIE_MAP)

    # if args.dataset == 'hmdb51':
    #     num_classes = 51
    # elif args.dataset == 'ucf101':
    #     num_classes = 101
    # elif args.dataset == 'kinetics':
    #     num_classes = 400
    # elif args.dataset == 'mini-kinetics':
    #     num_classes = 200

    if 'c3d' in args.arch or 'r2+1d' in args.arch:
        scale_range = [128, 128]
        crop_size = 112
    else:
        scale_range = [256, 320]  # from Activation-Recognition-Study
        crop_size = 224

    pre = 'mini-' if 'mini' in args.dataset else ''
    seperator = ';' if 'kinetics' in args.dataset else ' '
    max_frame = 64 if 'kinetics' in args.dataset else None
    train_augmentor = get_augmentor(is_train=True, image_size=crop_size, threed_data=True, version='v2',
                                    scale_range=scale_range)
    train_data = VideoDataSetLMDB(root_path=os.path.join(args.datadir, pre + 'train.lmdb'),
                                  list_file=os.path.join(args.datadir, pre + 'train.txt'),
                                  num_groups=16, transform=train_augmentor, is_train=False, seperator=seperator,
                                  filter_video=16, max_frame=max_frame)
    train_loader = build_dataflow(dataset=train_data, is_train=True, batch_size=args.batch_size, **kwargs)

    val_augmentor = get_augmentor(is_train=False, image_size=crop_size, threed_data=True)
    val_data = VideoDataSetLMDB(root_path=os.path.join(args.datadir, pre + 'val.lmdb'),
                                list_file=os.path.join(args.datadir, pre + 'val.txt'),
                                num_groups=16, transform=val_augmentor, is_train=False, seperator=seperator,
                                filter_video=16, max_frame=max_frame)
    val_loader = build_dataflow(dataset=val_data, is_train=False, batch_size=args.batch_size, **kwargs)

    train_size, val_size = len(train_loader.dataset), len(val_loader.dataset)
    print('Train dataset size: {}\nVal dataset size: {}'.format(train_size, val_size))
    print('')

    if args.arch == 'c3d':
        model = c3d.C3D(num_classes=args.num_classes)

    elif args.arch == 'r2+1d-pretrained' or 'r2+1d':
        model = r2plus1d_scaling(args.num_classes,args.multiplier)
    elif args.arch == 's3d':
        model = s3d.S3D(num_classes=args.num_classes, without_t_stride=True)

    if not args.resume:
        print(model)
    # dummy_input = torch.randn(1, 3, 16, crop_size, crop_size)
    # flops, params, _, _ = profile_prune(model, inputs=(dummy_input,), macs=False, prune=False)
    # print('params: {:.4g}M   flops: {:.4g}G'.format(params / 1e6, flops / 1e9))
    # print('')

    # criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps)

    # https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
    if use_cuda:
        model = nn.DataParallel(model)
        model.cuda()
        criterion.cuda()

    if args.test:
        checkpoint = torch.load(args.test_path)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint)
        test(model, val_loader, val_size, criterion)
        return

    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # scheduler = optim.lr_scheduler.MultiSteplr(optimizer, milestones=args.milestones, gamma=0.1)
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                         eta_min=4e-08)

    start_epoch = 1
    best_top1 = 0.
    best_epoch = 0

    xgen_load(model.module,args_ai)
    # epoch_top1 = test(model, val_loader, val_size, criterion)
    # print(epoch_top1)
    # if not args.resume:
    #     if not args.transfer:
    #         print('Training {} model from scratch on {} dataset'.format(args.arch, args.dataset))
    #     elif not args.pretrained and 'pretrained' not in args.arch:
    #         load_path = 'checkpoint'
    #         if args.arch == 'c3d':
    #             load_path = os.path.join(load_path, 'archive/kinetics_c3d/kinetics_c3d_epoch-32_top1-43.238.pt')
    #         print('Transfering to {} dataset'.format(args.dataset))
    #         print('Loading pretrained model from {}'.format(load_path))
    #         checkpoint = torch.load(load_path)
    #         state_dict = checkpoint['state_dict']
    #         try:
    #             model.load_state_dict(state_dict)
    #         except:
    #             last_keys = list(state_dict.keys())[-2:]
    #             for key in last_keys:
    #                 del (state_dict[key])
    #             model.load_state_dict(state_dict, strict=False)
    # else:
    #     load_path = os.path.join(args.logdir, '{}.pt'.format(ckpt_name))
    #     if os.path.exists(load_path):
    #         checkpoint = torch.load(load_path)
    #         start_epoch = checkpoint['epoch'] + 1
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print('Resuming from epoch {}'.format(checkpoint['epoch']))
    #     else:
    #         exit('Checkpoint does not exist.')
    #     try:
    #         checkpoint = torch.load(load_path.replace('.pt', '_best.pt'), map_location='cpu')
    #         best_epoch = checkpoint['epoch']
    #         best_top1 = checkpoint['top1']
    #     except:
    #         pass

    if args.lr_scheduler != None:
        for epoch in range(1, start_epoch):
            for _ in range(len(train_loader)):
                scheduler.step()

    CL.init(args=args_ai, model=model, optimizer=optimizer, data_loader=train_loader)


    for epoch in range(start_epoch, args.epochs + 1):
        torch.cuda.empty_cache()

        if args.lr_scheduler == None and (epoch - 1) in args.lr_milestones:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
            p = optimizer.param_groups[0]

        #################### train ####################
        model.train()
        start_time = time.time()

        running_loss = 0.0
        running_corrects = 0.0
        running_time = 0.0
        CL.before_each_train_epoch(epoch=epoch)
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            batch_start = time.time()

            inputs = Variable(inputs, requires_grad=True)
            labels = Variable(labels)
            if use_cuda:
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)
            loss = CL.update_loss(loss)
            loss.backward()

            optimizer.step()
            if args.lr_scheduler != None:
                scheduler.step()
                CL.after_scheduler_step(epoch=epoch)

            running_loss += loss.item() * inputs.size(0)
            batch_corrects = torch.sum(preds == labels.data).item()
            running_corrects += batch_corrects
            running_time += (time.time() - batch_start)

            if i % args.log_interval == 0:
                batch_top1 = batch_corrects / inputs.size(0) * 100
                batch_time = time.time() - batch_start
                p = optimizer.param_groups[0]
                print(
                    'Epoch [{}][{:3d}/{}]   LR {:.6f}   Loss {:.4f} ({:.4f})   Acc@1 {:7.3f}% ({:7.3f}%)   Time {:.3f} ({:.3f})'
                    .format(epoch, i, len(train_loader), p['lr'],
                            loss.item(), running_loss / args.batch_size / (i + 1),
                            batch_top1, running_corrects / args.batch_size * 100 / (i + 1),
                            batch_time, running_time / (i + 1)))

        epoch_loss = running_loss / train_size
        epoch_top1 = running_corrects / train_size * 100

        print('[Train] Loss {:.4f}   Acc@1 {:.3f}%   Time {:.0f}'.format(epoch_loss, epoch_top1,
                                                                         time.time() - start_time))

        #################### validation ####################
        epoch_top1 = test(model, val_loader, val_size, criterion)

        '''save model'''
        is_best = epoch_top1 > best_top1

        with open(os.path.join(args_ai['general']['work_place'],'tmp_result.txt'),'a+') as f:
            f.write(f"accuracy:{epoch_top1}|epoch:{epoch} \n")
        if is_best:
            best_top1 = epoch_top1
            best_epoch = epoch

            xgen_record(args_ai, model.module, epoch_top1, epoch)
        # save_checkpoint(
        #     {
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'top1': epoch_top1,
        #     },
        #     is_best, filename=os.path.join(args.logdir, '{}.pt'.format(ckpt_name)))

        print('Best Acc@1 {:.3f}%  Best epoch {}'.format(best_top1, best_epoch))
        print('')
    epoch_top1 = test(model, val_loader, val_size, criterion)
    xgen_record(args_ai, model.module, epoch_top1, -1)
    return args_ai
    # os.rename(os.path.join(args.logdir, '{}_best.pt'.format(ckpt_name)), \
    #           os.path.join(args.logdir, '{}_epoch-{}_top1-{:.3f}.pt'.format(ckpt_name, best_epoch, best_top1)))


def test(model, val_loader, val_size, criterion):
    model.eval()
    start_time = time.time()

    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in tqdm(val_loader):
        # for inputs, labels in val_loader:
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / val_size
    epoch_top1 = running_corrects / val_size * 100

    print('[Val] Loss {:.4f}   Acc@1 {:.3f}%   Time {:.0f}'.format(epoch_loss, epoch_top1, time.time() - start_time))

    return epoch_top1


if __name__ == '__main__':
    json_path = './s2+1d_config/xgen.json'
    args_ai = json.load(open(json_path, 'r'))
    args_ai['origin'][ "pretrain_model_weights_path"] = args_ai['task']['pretrained_model_path']
    work_place = args_ai['general']['work_place']
    for i in [2,4,6,12,16,24,32]:
        _work_place = os.path.join(work_place,str(i))
        args_ai['general']['work_place']=_work_place
        os.makedirs(_work_place, exist_ok=True)
        args_ai['origin']['multiplier'] = i
        training_main(args_ai)
