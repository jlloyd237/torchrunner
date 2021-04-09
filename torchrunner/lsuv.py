import torch.nn as nn

from torchrunner.module import GeneralRelu, fc_layer
from torchrunner.hook import OutputStatsHook


class LSUVConvLayer(nn.Module):
    def __init__(self, ni, nf, ks=3, stride=2, output_bias=0, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride, bias=True)
        self.relu = GeneralRelu(output_bias=output_bias, **kwargs)

    def forward(self, x):
        return self.relu(self.conv(x))

    @property
    def output_bias(self):
        return self.relu.output_bias

    @output_bias.setter
    def output_bias(self, value):
        self.relu.output_bias = value

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias


def lsuv_conv_layer(ni, nf, ks=3, stride=2, output_bias=0, **kwargs):
    return LSUVConvLayer(ni, nf, ks, stride, output_bias, **kwargs)

def lsuv_conv_net(ni, no, nf=[], nh =[], ks=3, stride=2, output_bias=0, p_fc=0):
    nf = [ni] + nf
    nh = [nf[-1]] + nh
    return nn.Sequential(
        *[lsuv_conv_layer(nf[i], nf[i+1], ks=ks, stride=stride, output_bias=output_bias)
          for i in range(len(nf) - 1)],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        *[fc_layer(nh[i], nh[i+1], p=p_fc) for i in range(len(nh) - 1)],
        nn.Linear(nh[-1], no),
    )

def init_lsuv_conv_net(model, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    for l in model:
        if isinstance(l, LSUVConvLayer):
            f(l.weight, a=0.1)
            l.bias.data.zero_()
        elif isinstance(l, nn.Linear):
            f(l.weight, a=0.1)
            l.bias.data.zero_()

def lsuv_module(module, model, xb):
    with OutputStatsHook(module, mode='all') as h:
        while model(xb) is not None and abs(h.stds[-1] - 1) > 1e-3:
            module.weight.data /= h.stds[-1]
        while model(xb) is not None and abs(h.means[-1]) > 1e-3:
            module.output_bias -= h.means[-1]
    return h.means[-1].item(), h.stds[-1].item()
