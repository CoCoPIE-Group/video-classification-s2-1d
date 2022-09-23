import os
import sys
import time
from time import strftime
import shutil
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import admm

import numpy as np
from network import c3d, s3d

from video_utils.utils import build_dataflow, get_augmentor
from video_utils.video_dataset import VideoDataSet, VideoDataSetLMDB
from utils import *

import argparse
from options import parser
args = parser.parse_args()

if args.sparsity_type in ['blk-vanilla', 'blk-kgs']:
    row, col = args.connectivity_block_size
    args.blk_str = '-r{}c{}'.format(row, col)
else:
    args.blk_str = ''
ckpt_name = '{}_{}_{}{}'.format(args.dataset, args.arch, args.sparsity_type, args.blk_str)
args.ckpt_dir = os.path.join('checkpoint', ckpt_name)
if args.admm and not args.resume and os.path.exists(args.ckpt_dir):
    i = 1
    while os.path.exists(args.ckpt_dir + '_v{}'.format(i)):
        i += 1
    os.rename(args.ckpt_dir, args.ckpt_dir + '_v{}'.format(i))
os.makedirs(args.ckpt_dir, exist_ok=True)

if args.logger:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    retrain_str = 'retrain_{}rhos{}'.format(args.rho_num, '_distill' if args.distill else '')
    logger.addHandler(logging.FileHandler(os.path.join(args.ckpt_dir, '{}_{}_{}.log'.format( \
        ckpt_name, retrain_str if args.masked_retrain else 'admm', strftime('%m%d%Y-%H%M'))), 'w'))
    global print
    print = logger.info

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.backends.cudnn.benchmark = True  # will result in non-determinism

kwargs = {'num_workers': args.workers, 'worker_init_fn': np.random.seed(args.seed), 'pin_memory': True} if use_cuda else {}

if args.dataset == 'hmdb51':
    num_classes = 51
elif args.dataset == 'ucf101':
    num_classes = 101
elif args.dataset == 'kinetics':
    num_classes = 400

# set up model architecture
if args.arch == 'c3d':
    model = c3d.C3D(num_classes=num_classes)
elif args.arch == 'r2+1d-pretrained':
    model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = nn.Linear(512, num_classes)
elif args.arch == 's3d':
    model = s3d.S3D(num_classes=num_classes, without_t_stride=True)


def get_load_path(args):
    if args.arch == 'c3d':
        if args.dataset == 'ucf101':
            load_path = 'checkpoint/c3d_ucf101_dense_top1-81.6.pt'
        elif args.dataset == 'hmdb51':
            load_path = 'checkpoint/hmdb51_c3d_transfer_epoch-18_top1-53.458.pt'
    elif args.arch == 'r2+1d-pretrained':
        if args.dataset == 'ucf101':
            load_path = 'checkpoint/r2+1d_ucf101_dense_top1-94.0.pt'
        elif args.dataset == 'hmdb51':
            load_path = 'checkpoint/hmdb51_r2+1d-pretrained_transfer_epoch-20_top1-71.300.pt'
    elif args.arch == 's3d':
        if args.dataset == 'ucf101':
            load_path = 'checkpoint/s3d_ucf101_dense_top1-90.6.pt'
        elif args.dataset == 'hmdb51':
            load_path = 'checkpoint/hmdb51-s3d-ts-max-f16-multisteps-bs32-e20_epoch-17_top1-67.151.pt'
    return load_path


### Set up teacher model structure
# For 3D models, there is usually not larger and similar models, 
# so use the same arch as teacher arch and baseline model as teacher model
if args.distill:
    args.teacharch = args.arch
    if args.teacharch == 'c3d':
        teacher = c3d.C3D(num_classes=num_classes)
    elif args.teacharch == 'r2+1d-pretrained':
        teacher = torchvision.models.video.r2plus1d_18(pretrained=False)
        teacher.fc = nn.Linear(512, num_classes)
    elif args.teacharch == 's3d':
        teacher = s3d.S3D(num_classes=num_classes, without_t_stride=True)
    args.teacher_path = get_load_path(args)

if 'c3d' in args.arch or 'r2+1d' in args.arch:
    scale_range = [128, 128]
    crop_size = 112
else:
    scale_range = [256, 320]
    crop_size = 224

if use_cuda:
    model.cuda()
    if args.distill:
        teacher.cuda()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
        if args.distill:
            teacher = torch.nn.DataParallel(teacher)

# teacher load model
if args.distill:
    teach_checkpoint = torch.load(args.teacher_path)
    try:
        teach_state_dict = teach_checkpoint['state_dict']
    except:
        teach_state_dict = teach_checkpoint
    teacher.load_state_dict(teach_state_dict)

''' disable all bag of tricks'''
if args.no_tricks:
    # disable all trick even if they are set to some value
    args.lr_scheduler = 'default'
    args.warmup = False
    args.mixup = False
    args.smooth = False
    args.alpha = 0.0
    args.smooth_eps = 0.0

''' working directories '''
dir_profile = 'user_profile'

ROOT = '../../ms/dataset/'
args.datadir = os.path.join(ROOT, args.dataset + '_frame')

pre = 'mini_' if 'mini' in args.dataset else ''
seperator = ';' if 'kinetics' in args.dataset else ' '
max_frame = 64 if 'kinetics' in args.dataset else None
train_augmentor = get_augmentor(is_train=True, image_size=crop_size, threed_data=True, version='v2', scale_range=scale_range)
train_data = VideoDataSetLMDB(root_path=os.path.join(args.datadir, pre+'train.lmdb'), list_file=os.path.join(args.datadir, pre+'train.txt'),
                              num_groups=16, transform=train_augmentor, is_train=False, seperator=seperator,
                              filter_video=16, max_frame=max_frame)
train_loader = build_dataflow(dataset=train_data, is_train=True, batch_size=args.batch_size, **kwargs)

val_augmentor = get_augmentor(is_train=False, image_size=crop_size, threed_data=True)
val_data = VideoDataSetLMDB(root_path=os.path.join(args.datadir, pre+'val.lmdb'), list_file=os.path.join(args.datadir, pre+'val.txt'),
                            num_groups=16, transform=val_augmentor, is_train=False, seperator=seperator,
                            filter_video=16, max_frame=max_frame)
val_loader = build_dataflow(dataset=val_data, is_train=False, batch_size=args.batch_size, **kwargs)

def load_multi_gpu(model, checkpoint, optimizer, first=False):
    # baseline model for pruning, pruned model for retrain
    try:
        state_dict = checkpoint['state_dict']
        if not first:
            optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        state_dict = checkpoint
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = {}
        for key, value in state_dict.items():
            newkey = 'module.' + key
            new_state_dict[newkey] = value
        model.load_state_dict(new_state_dict)

def main():
    print(' '.join(sys.argv))
    print('General config:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if (args.admm and args.masked_retrain):
        raise ValueError('cannot do both masked retrain and admm')

    '''bag of tricks setups'''
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
    args.smooth = args.smooth_eps > 0.0
    args.mixup = args.alpha > 0.0

    optimizer_init_lr = args.warmup_lr if args.warmup else args.learning_rate

    optimizer = None
    if args.optmzr == 'sgd':
        optimizer = optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optmzr == 'adam':
        optimizer = optim.Adam(model.parameters(), optimizer_init_lr)

    scheduler = None
    if args.lr_scheduler == 'default':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(args.epochs * 0.2)) * len(train_loader), gamma=0.5)
    elif args.lr_scheduler == 'multistep':
        epoch_milestones = [45, 90, 120]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=0.5)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)
    else:
        raise Exception('unknown lr scheduler')

    if args.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=args.learning_rate / args.warmup_lr, 
                                           total_iter=args.warmup_epochs * len(train_loader), after_scheduler=scheduler)

    '''====================='''
    ''' multi-rho admm train'''
    '''====================='''
    initial_rho = args.rho
    if args.admm:
        admm_prune(initial_rho, criterion, optimizer, scheduler)


    '''=============='''
    '''masked retrain'''
    '''=============='''
    if args.masked_retrain:                
        masked_retrain(initial_rho, criterion, optimizer, scheduler)


def admm_prune(initial_rho, criterion, optimizer, scheduler):    
    for i in range(args.rho_num):
        current_rho = initial_rho * 10 ** i

        if i == 0 and not args.resume:
            if args.load_path is not '':
                load_path = args.load_path
            else:
                load_path = get_load_path(args)            
            print('>_ Loading baseline/progressive model from {}\n'.format(load_path))

        elif i > 0 and not args.resume:
            load_path = os.path.join(args.ckpt_dir, '{}_{}.pt'.format(ckpt_name, current_rho / 10))
        else:
            load_path = os.path.join(args.ckpt_dir, '{}_{}.pt'.format(ckpt_name, current_rho))

        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
        else:
            exit('Checkpoint does not exist.')

        load_multi_gpu(model, checkpoint, optimizer, first=(i == 0 and not args.resume))

        start_epoch = 1
        best_top1 = 0.
        best_epoch = 0
        if args.resume:
            start_epoch = checkpoint['epoch'] + 1
            try:
                checkpoint = torch.load(load_path.replace('.pt', '_best.pt'), map_location='cpu')
                best_epoch = checkpoint['epoch']
                best_top1 = checkpoint['top1']
            except:
                pass
        
        ADMM = admm.ADMM(model, file_name=os.path.join(dir_profile, args.config_file + '.yaml'), rho=current_rho)
        admm.admm_initialization(args, ADMM=ADMM, model=model)  # intialize Z variable
        if i == 0:
            print('Prune config:')
            for k, v in ADMM.prune_cfg.items():
                print('\t{}: {}'.format(k, v))
            print('')

            shutil.copy(os.path.join(dir_profile, args.config_file + '.yaml'), \
                os.path.join(args.ckpt_dir, args.config_file + '.yaml'))

        # admm train

        save_path = os.path.join(args.ckpt_dir, '{}_{}.pt'.format(ckpt_name, current_rho))

        for epoch in range(start_epoch, args.epochs + 1):
            print('current rho: {}'.format(current_rho))
            train(ADMM, train_loader, criterion, optimizer, scheduler, epoch, args)
            t_loss, top1 = test(model, criterion, val_loader)

            is_best = top1 > best_top1
            if epoch < args.epochs:
                save_state_dict = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'top1': top1,
                }
            else:
                save_state_dict = model.state_dict()
            save_checkpoint(save_state_dict, False, save_path)
            if is_best:
                best_top1 = top1
                best_epoch = epoch

            print('Best Acc@1 {:.3f}%   Best epoch {}'.format(best_top1, best_epoch))
            print('')

            if ((epoch - 1) % args.admm_epochs == 0 or epoch == args.epochs):
                print('Weight < 1e-4:')
                for layer in ADMM.prune_cfg.keys():
                    weight = model.state_dict()[layer]
                    zeros = len((abs(weight)<1e-4).nonzero())
                    weight_size = torch.prod(torch.tensor(weight.shape))
                    print('   {}: {}/{} = {:.4f}'.format(layer.split('module.')[-1], zeros, weight_size, float(zeros)/float(weight_size)))
                print('')


def masked_retrain(initial_rho, criterion, optimizer, scheduler):
    # load admm trained model
    if not args.resume:
        load_path = os.path.join(args.ckpt_dir, '{}_{}.pt'.format(ckpt_name, initial_rho * 10 ** (args.rho_num - 1)))
        print('>_ Loading model from {}\n'.format(load_path))
    else:
        load_path = os.path.join(args.ckpt_dir, '{}_{}rhos.pt'.format(ckpt_name, args.rho_num))
    
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
    else:
        exit('Checkpoint does not exist.')

    load_multi_gpu(model, checkpoint, optimizer, first=True)

    start_epoch = 1
    top1_list = [0.]
    loss_list = []
    best_epoch = 0
    if args.resume:
        start_epoch = checkpoint['epoch'] + 1
        try:
            checkpoint = torch.load(load_path.replace('.pt', '_best.pt'), map_location='cpu')
            best_epoch = checkpoint['epoch']
            best_top1 = checkpoint['top1']
        except:
            pass

    # restore scheduler
    for epoch in range(1, start_epoch):
        for _ in range(len(train_loader)):
            scheduler.step()

    ADMM = admm.ADMM(model, file_name=os.path.join(args.ckpt_dir, args.config_file + '.yaml'), rho=initial_rho)
    print('Prune config:')
    for k, v in ADMM.prune_cfg.items():
        print('\t{}: {}'.format(k, v))
    print('')

    admm.hard_prune(args, ADMM, model)

    save_path = os.path.join(args.ckpt_dir, '{}_{}rhos{}.pt'.format(ckpt_name, args.rho_num, \
        '_distill' if args.distill else ''))

    for epoch in range(start_epoch, args.epochs + 1):
        idx_loss_dict = train(ADMM, train_loader, criterion, optimizer, scheduler, epoch, args)
        t_loss, top1 = test(model, criterion, val_loader)

        best_top1 = max(top1_list)
        is_best = top1 > best_top1
        save_state_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'top1': top1,
        }
        save_checkpoint(save_state_dict, is_best, save_path)
        if is_best:
            best_top1 = top1
            best_epoch = epoch

        print('Best Acc@1: {:.3f}%  Best epoch: {}\n'.format(best_top1, best_epoch))

        loss_list.append(t_loss)
        top1_list.append(top1)

        ## save loss and acc plots
        np.savez(save_path.replace('.pt', '_results.npz'), loss=np.array(loss_list), acc=np.array(top1_list[1:]))
        
        import matplotlib.pyplot as plt
        epoch_list = range(start_epoch, epoch + 1)
        
        # https://matplotlib.org/gallery/api/two_scales.html
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('test loss', color=color)
        ax1.plot(epoch_list, loss_list, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('test acc', color=color)  # we already handled the x-label with ax1
        ax2.plot(epoch_list, top1_list[1:], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(save_path.replace('.pt', '_results.pdf'))

        plt.close()

    os.rename(save_path.replace('.pt', '_best.pt'), \
        save_path.replace('.pt', '_epoch-{}_top1-{:.3f}.pt'.format(best_epoch, best_top1)))


def train(ADMM, train_loader, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill_losses = AverageMeter() if args.distill else None
    old_ce_losses = AverageMeter() if args.distill else None
    top1 = AverageMeter()
    idx_loss_dict = {}

    # switch to train mode
    model.train()

    if args.masked_retrain:
        print('full acc re-train masking')
    elif args.combine_progressive:
        print('progressive admm-train/re-train masking')
    if args.masked_retrain or args.combine_progressive:
        masks = {}
        for name, W in model.named_parameters():
            weight = W.detach()
            non_zeros = weight != 0 
            zero_mask = non_zeros.type(torch.float32)
            masks[name] = zero_mask
    
    end = time.time()
    epoch_start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, args)
        else:  # only in masked retrain
            scheduler.step()

        if use_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if args.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, args.alpha)

        # compute output
        output = model(input)

        if args.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, args.smooth)
        else:
            ce_loss = criterion(output, target, smooth=args.smooth)

        # compute teacher output
        if args.distill and epoch <= args.epochs - 5:
            with torch.no_grad():
                teacher_output = teacher(input)
            distill_loss = KL(output, teacher_output, args)
            old_ce_loss = ce_loss.clone()
            ce_loss = ce_loss * (1 - args.kd_coefficient) + distill_loss * args.kd_coefficient

        if args.admm:
            admm.z_u_update(args, ADMM, model, train_loader, optimizer, epoch, input, i)  # update Z and U variables
            ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss

        # measure accuracy and record loss
        prec1, _ = accuracy(output, target, topk=(1,5))

        losses.update(ce_loss.item(), input.size(0))
        if args.distill and epoch <= args.epochs - 5:
            distill_losses.update(distill_loss.item(), input.size(0))
            old_ce_losses.update(old_ce_loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.admm:
            mixed_loss.backward()
        else:
            ce_loss.backward()
            
        if args.masked_retrain or args.combine_progressive:
            with torch.no_grad():
                for name, W in model.named_parameters():
                    if name in masks:
                        W.grad *= masks[name]

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            distill_str = 'Distill Loss {distill_loss.avg:.4f}   CE Loss {ce_loss.avg:.4f}   ' \
                .format(distill_loss=distill_losses, ce_loss=old_ce_losses) if args.distill and epoch <= args.epochs - 5 else ''
            print('({0}) lr [{1:.6f}]   '
                  'Epoch [{2}][{3:3d}/{4}]   '
                  'Status [admm-{5}][retrain-{6}]   '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                  '{distill_str}'
                  'Acc@1 {top1.val:7.3f}% ({top1.avg:7.3f}%)'
                  .format(args.optmzr, current_lr,
                   epoch, i, len(train_loader), args.admm, args.masked_retrain, batch_time=batch_time, loss=losses, top1=top1,
                   distill_str=distill_str))
        if i % 100 == 0:
            idx_loss_dict[i] = losses.avg
    print('[Train] Loss {:.4f}   Acc@1 {:.3f}%   Time {}'.format(
        losses.avg, top1.avg, int(time.time() - epoch_start_time)))
    return idx_loss_dict


def test(model, criterion, test_loader):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = len(test_loader.dataset)
    epoch_start_time = time.time()
    with torch.no_grad():
        for input, target in test_loader:
            if use_cuda:
                input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    top1 = 100. * float(correct) / float(total)
    print('Test Loss {:.4f}   Acc@1 {}/{} ({:.3f}%)   Time {}' \
        .format(losses.avg, correct, total, top1, int(time.time() - epoch_start_time)))
    return losses.avg, top1


if __name__ == '__main__':
    main()
