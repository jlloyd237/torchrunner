import torch.nn as nn

from torchrunner.hook import Hook


def find_modules(m, cond):
    if cond(m):
        return [m]
    return sum([find_modules(o, cond) for o in m.children()], [])

def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ReLU, nn.BatchNorm2d)
    return isinstance(l, lin_layers)

def get_module_device(module):
    return next(module.parameters()).device

def get_module_params_size(module):
    params = sum([p.numel() for p in module.parameters()])
    trains = [p.requires_grad for p in module.parameters()]
    return params, (False if len(trains) == 0 else trains[0])


class ModelSummaryHook(Hook):
    def __init__(self, module, is_backward=True, print_mod=False):
        super().__init__(module, is_backward)
        self.print_mod = print_mod

    def hook_func(self, module, input, output):
        print((f"====\n{module}\n" if self.print_mod else "") + f"output shape: {list(output.shape)}")
        params, trainable = get_module_params_size(module)
        print(f"total params: {params}, trainable: {trainable}")


def model_summary(model, dl, find_all=False, print_mod=False):
    device = get_module_device(model)
    xb, yb = next(iter(dl))
    xb, yb = xb.to(device), yb.to(device)
    mods = find_modules(model, is_lin_layer) if find_all else model.children()
    with ModelSummaryHook.apply_to_modules(mods, print_mod=print_mod) as hooks:
        model(xb)
    params, trainable = get_module_params_size(model)
    print(f"====\ntotal params: {params}, trainable: {trainable}")