import torch
import torch.nn as nn
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class GeneralRelu(nn.Module):
    def __init__(self, negative_slope=None, output_bias=None, max_output=None):
        super().__init__()
        self.negative_slope = negative_slope
        self.output_bias = output_bias
        self.max_output = max_output

    def forward(self, x):
        x = F.leaky_relu(x, self.negative_slope) if self.negative_slope is not None else F.relu(x)
        if self.output_bias is not None:
            x.add_(self.output_bias)
        if self.max_output is not None:
            x.clamp_max_(self.max_output)
        return x


class ResBlock(nn.Module):
    def __init__(self, nf, ks=3, p=0):
        super().__init__()
        self.conv1 = conv_layer(nf, nf, ks=ks, stride=1, p=p)
        self.conv2 = conv_layer(nf, nf, ks=ks, stride=1, p=p)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class DenseBlock(nn.Module):
    def __init__(self, nf, ks=3, p=0):
        super().__init__()
        self.conv1 = conv_layer(nf, nf, ks=ks, stride=1, p=p)
        self.conv2 = conv_layer(nf, nf, ks=ks, stride=1, p=p)

    def forward(self, x):
        return torch.cat([x, self.conv2(self.conv1(x))], dim=1)


def fc_layer(ni, nf, p=0):
    return nn.Sequential(
        nn.Linear(ni, nf, bias=False),  # bias redundant with batch norm),
        nn.BatchNorm1d(nf),
        nn.ReLU(),
        nn.Dropout(p=p),
    )

def conv_layer(ni, nf, ks=3, stride=2, p=0, **kwargs):
    return nn.Sequential(
        nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=False, **kwargs),  # bias redundant with batch norm
        nn.BatchNorm2d(nf),
        nn.ReLU(),
        nn.Dropout2d(p=p)
    )

def conv_net(ni, no, nf=[], nh =[], ks=3, stride=2, p_conv=0, p_fc=0):
    nf = [ni] + nf
    nh = [nf[-1]] + nh
    return nn.Sequential(
        *[conv_layer(nf[i], nf[i+1], ks=ks, stride=stride, p=p_conv) for i in range(len(nf) - 1)],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        *[fc_layer(nh[i], nh[i+1], p=p_fc) for i in range(len(nh) - 1)],
        nn.Linear(nh[-1], no),
    )

def res_block(nf, ks=3, p=0):
    return ResBlock(nf, ks, p)

def res_conv_block(ni, nf, ks=3, stride=2, p=0):
    return nn.Sequential(
        res_block(ni, ks=ks, p=p),
        conv_layer(ni, nf, ks=ks, stride=stride, p=p),
    )

def res_net(ni, no, nf=[], nh =[], ks=3, stride=2, p_conv=0, p_fc=0):
    nf = [ni] + nf
    nh = [nf[-1]] + nh
    return nn.Sequential(
        conv_layer(nf[0], nf[1], ks=ks, stride=stride, p=p_conv),
        *[res_conv_block(nf[i], nf[i+1], ks=ks, stride=stride, p=p_conv) for i in range(1, len(nf) - 1)],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        *[fc_layer(nh[i], nh[i+1], p=p_fc) for i in range(len(nh) - 1)],
        nn.Linear(nh[-1], no),
    )

def dense_block(nf, ks=3, p=0):
    return DenseBlock(nf, ks, p)

def dense_conv_block(ni, nf, ks=3, stride=2, p=0):
    return nn.Sequential(
        dense_block(ni, ks=ks, p=p),
        conv_layer(2 * ni, nf, ks=ks, stride=stride, p=p),
    )

def dense_net(ni, no, nf=[], nh =[], ks=3, stride=2, p_conv=0, p_fc=0):
    nf = [ni] + nf
    nh = [nf[-1]] + nh
    return nn.Sequential(
        conv_layer(nf[0], nf[1], ks=ks, stride=stride, p=p_conv),
        *[dense_conv_block(nf[i], nf[i+1], ks=ks, stride=stride, p=p_conv) for i in range(1, len(nf) - 1)],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        *[fc_layer(nh[i], nh[i+1], p=p_fc) for i in range(len(nh) - 1)],
        nn.Linear(nh[-1], no),
    )
