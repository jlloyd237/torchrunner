import torch


def accuracy(output, target):
    _, label = torch.max(output, dim=1)
    return (label == target).float().mean()

def mean_abs_err(output, target, dim=None, keepdim=False):
    return (target - output).abs().mean(dim=dim, keepdim=keepdim)

def mean_sqr_err(output, target, dim=None, keepdim=False):
    return (target - output).pow(2).mean(dim=dim, keepdim=keepdim)

def rmse(output, target, dim=None, keepdim=False):
    return (target - output).pow(2).mean(dim=dim, keepdim=keepdim).sqrt()
