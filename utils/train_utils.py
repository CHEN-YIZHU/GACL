import torch_optimizer
from torch.nn import Module
from torch import optim
from torch.optim import lr_scheduler
import torch
import os

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

def select_optimizer(opt_name: str, lr: float, model: Module) -> optim.Optimizer:
    if opt_name == "adam":
        # print("opt_name: adam")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt

def select_scheduler(sched_name: str, opt: optim.Optimizer, hparam=None) -> lr_scheduler._LRScheduler:
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80, 90], gamma=0.1)
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler


def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if not output_has_class_ids:
        output = torch.Tensor(output)
    else:
        output = torch.LongTensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = 1
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        
        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        res=correct_k.mul_(1.0/batch_size)
        return res

def load_pretrain(target_model):
    """
        target_model: the model we want to replace the parameters (most likely un-trained)
    """
    if os.path.isfile('models/best_checkpoint.pth'):
        checkpoint = torch.load('models/best_checkpoint.pth', map_location='cpu')
        print("Load Deit Checkpoints")
      
    target = target_model.state_dict()
    pretrain = checkpoint['model']
    transfer, missing = {}, []
    for k, _ in target.items():
        if k in pretrain and 'head' not in k:
            transfer[k] = pretrain[k]
        else:
            missing.append(k)
    target.update(transfer)
    target_model.load_state_dict(target)
    return target_model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count
    

    
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'