from __future__ import print_function
import os
import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as LA
import yaml
from utils import *


class ADMM:
    def __init__(self, model, file_name, rho=0.001):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}

        self.init(file_name, model)

    def init(self, config, model):
        '''
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        '''
        with open(config, 'r') as stream:
            raw_dict = yaml.full_load(stream)
            if 'prune_ratios' in raw_dict:
                self.prune_cfg = raw_dict['prune_ratios']
            for k, v in self.prune_cfg.items():
                self.rhos[k] = self.rho
            for (name, W) in model.named_parameters():
                if name not in self.prune_cfg:
                    continue
                self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z


def weight_pruning(args, weight_in, prune_ratio):
    '''
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    '''
    weight = weight_in.clone().detach()  # gpu tensor on pytorch (original)
    percent = prune_ratio * 100

    if args.sparsity_type in ['blk-vanilla', 'blk-kgs']:
        global shape, ext_shape, blk_h, blk_w, num_blk_h, num_blk_w, kernel_size
        shape = weight.shape
        block_shape = args.connectivity_block_size

        ext_shape = [(shape[i] + block_shape[i] - 1) // block_shape[i] * block_shape[i] for i in range(2)] + list(shape[2:])

        blk_h, blk_w = block_shape
        num_blk_h, num_blk_w = ext_shape[0]//blk_h, ext_shape[1]//blk_w
        kernel_size = torch.prod(torch.tensor(shape[2:]))

        padding = nn.ZeroPad2d((0, ext_shape[1] - shape[1], 0, ext_shape[0] - shape[0]))
        weight_ = weight.reshape(shape[0], shape[1], -1)
        weight_ = torch.stack([padding(weight_[:, :, i]) for i in range(kernel_size)], dim=2)

    if args.sparsity_type == 'filter':
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        row_l2_norm = torch.norm(weight2d, 2, dim=1)
        percentile = np.percentile(row_l2_norm.cpu(), percent)
        under_threshold = row_l2_norm <= percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        above_threshold = above_threshold.type(torch.float32)
        return above_threshold, weight

    elif args.sparsity_type == 'blk-vanilla':
        weight_ = reshape_matrix2block(weight_, blk_h, blk_w)
        l2_norm = torch.norm(weight_, 2, dim=1)  # gpu tensor

        valid = torch.Tensor(num_blk_h * num_blk_w)
        for i in range(num_blk_h * num_blk_w):
            valid[i] = max(len((weight_[i]!=0).nonzero()), 1)
        if 'cuda' in str(l2_norm.device):
            valid = valid.cuda()
        l2_norm_mean = torch.div(l2_norm, torch.sqrt(valid))

        percentile = np.percentile(l2_norm_mean.cpu(), percent)
        under_threshold = l2_norm_mean <= percentile
        above_threshold = l2_norm_mean > percentile
        weight_[under_threshold, :] = 0
        above_threshold = above_threshold.type(torch.float32)

        weight_ = reshape_block2matrix(weight_, num_blk_h, num_blk_w, blk_h, blk_w * kernel_size)
        weight = weight_.reshape(ext_shape)[:shape[0], :shape[1]]

        return above_threshold, weight

    elif args.sparsity_type == 'blk-kgs':
        weight_ = reshape_matrix2block_kernel(weight_, blk_h, blk_w)
        l2_norm = torch.norm(weight_, 2, dim=1)  # gpu tensor

        valid = torch.Tensor(num_blk_h * num_blk_w * int(kernel_size))
        for i in range(num_blk_h * num_blk_w * kernel_size):
            valid[i] = max(len((weight_[i]!=0).nonzero()), 1)
        if 'cuda' in str(l2_norm.device):
            valid = valid.cuda()
        l2_norm_mean = torch.div(l2_norm, torch.sqrt(valid))

        percentile = np.percentile(l2_norm_mean.cpu(), percent)
        under_threshold = l2_norm_mean <= percentile
        above_threshold = l2_norm_mean > percentile
        weight_[under_threshold, :] = 0
        above_threshold = above_threshold.type(torch.float32)

        weight_ = reshape_block2matrix_kernel(weight_, num_blk_h, num_blk_w, blk_h, blk_w, int(kernel_size))
        weight = weight_.reshape(ext_shape)[:shape[0], :shape[1]]
        return above_threshold, weight


def hard_prune(args, ADMM, model):
    '''
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda

    '''
    print('hard pruning\n')

    index_npz = os.path.join(args.ckpt_dir, '{}_{}_{}_{}rhos_index.npz'.format(args.dataset, args.arch, \
        args.sparsity_type + args.blk_str, args.rho_num))
    index_dict = {}

    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_cfg:  # ignore layers that do not have rho
            continue
        retained_index, cuda_pruned_weights = weight_pruning(args, W, ADMM.prune_cfg[name])  # get sparse model in cuda
        if name.startswith('module.'):
            name = name[7:]
        index_dict[name] = retained_index.cpu()
        W.data = cuda_pruned_weights  # replace the data field in variable

    np.savez(index_npz, **index_dict)

def admm_initialization(args, ADMM, model):

    if not args.admm:
        return    
    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_cfg:
            continue
        _, updated_Z = weight_pruning(args, W, ADMM.prune_cfg[name])  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
        ADMM.ADMM_Z[name] = updated_Z


def z_u_update(args, ADMM, model, train_loader, optimizer, epoch, data, batch_idx):

    if not args.admm:
        return
    if epoch != 1 and (epoch - 1) % args.admm_epochs == 0 and batch_idx == 0:
        for (name, W) in model.named_parameters():
            if name not in ADMM.prune_cfg:
                continue
            W_detach = W.detach()
            U_detach = ADMM.ADMM_U[name].detach()
            ADMM.ADMM_Z[name] = W_detach + U_detach  # Z(k+1) = W(k+1) + U[k]
            _, updated_Z = weight_pruning(args, ADMM.ADMM_Z[name], ADMM.prune_cfg[name])  # equivalent to Euclidean Projection
            ADMM.ADMM_Z[name] = updated_Z
            Z_detach = ADMM.ADMM_Z[name].detach()
            ADMM.ADMM_U[name] = W_detach - Z_detach + U_detach  # U(k+1) = W(k+1) - Z(k+1) + U(k)


def append_admm_loss(args, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    if args.admm:
        for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in ADMM.prune_cfg:
                continue
            # admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm((W + ADMM.ADMM_U[name])[ADMM.ADMM_Z[name]==0], p=2) ** 2)
    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def admm_adjust_learning_rate(optimizer, epoch, args):
    ''' (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''
    '''
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    '''
    admm_epoch = args.admm_epochs
    lr = None
    if (epoch - 1) % admm_epoch == 0:
        lr = args.learning_rate
    else:
        admm_epoch_offset = (epoch - 1) % admm_epoch
        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.
        lr = args.learning_rate * (0.5 ** (admm_epoch_offset // admm_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
