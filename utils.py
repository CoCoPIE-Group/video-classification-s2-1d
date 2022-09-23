import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
    

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        if 'state_dict' in state:
            state = state['state_dict']
        torch.save(state, filename.replace('.pt', '_best.pt'))

        
class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    '''Computes the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''
    # only in masked retrain

    # adjust learning rate
    if args.warmup and epoch - 1 <= args.warmup_epochs:
        lr = args.warmup_lr + (args.learning_rate - args.warmup_lr) / args.warmup_epochs * (epoch - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    args.lr_decay = max(1, int(args.epochs * 0.2))
    #lr = args.learning_rate * (0.3 ** (epoch // args.lr_decay))
    lr = args.learning_rate * (0.5 ** ((epoch - 1) // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)

class GradualWarmupScheduler(_LRScheduler):
    ''' Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    '''

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def KL(outputs, targets, args):
    '''distillation'''
    log_softmax_outputs = F.log_softmax(outputs/args.temperature, dim=1)
    softmax_targets = F.softmax(targets/args.temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def reshape_matrix2block(matrix, blk_h, blk_w):
    block = torch.cat(torch.split(matrix, blk_h), dim=1)
    block = torch.split(block, blk_w, dim=1)
    block = torch.stack([i.reshape(-1) for i in block])
    return block
def reshape_block2matrix(block, num_blk_h, num_blk_w, blk_h, blk_w):
    matrix = []
    for i in range(num_blk_h):
        for j in range(blk_h):
            matrix.append(block[num_blk_w*i:num_blk_w*(i+1), blk_w*j:blk_w*(j+1)].reshape(-1))
    matrix = torch.stack(matrix)
    return matrix


def reshape_matrix2block_kernel(matrix, blk_h, blk_w):
    block = torch.cat(torch.split(matrix, blk_h), dim=1)
    block = torch.split(block, blk_w, dim=1)
    block = torch.cat([i.permute(2, 0, 1).reshape(-1, blk_h*blk_w) for i in block])
    return block
def reshape_block2matrix_kernel(block, num_blk_h, num_blk_w, blk_h, blk_w, kernel_size):
    matrix = []
    blocks = torch.stack(torch.split(block, kernel_size), dim=1).permute(1, 2, 0)
    for i in range(num_blk_h):
        for j in range(blk_h):
            matrix.append(blocks[num_blk_w*i:num_blk_w*(i+1), blk_w*j:blk_w*(j+1)].reshape(-1))
    matrix = torch.stack(matrix)
    return matrix
