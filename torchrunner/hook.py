from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torchrunner.callback import Callback


class HookList:
    def __init__(self, hooks):
        self.hooks = hooks

    def __enter__(self, *args):
        return self

    def __exit__ (self, *args):
        self.remove()

    def __getitem__(self, i):
        return self.hooks[i]

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.hooks[:10]}'
        if len(self) > 10:
            res = res[:-1] + '...]'
        return res

    def remove(self):
        for h in self.hooks:
            h.remove()


class Hook(ABC):
    def __init__(self, module, is_backward=False):
        self.module = module
        self.is_backward = is_backward

        if is_backward:
            self.hook = module.register_backward_hook(self.hook_func)
        else:
            self.hook = module.register_forward_hook(self.hook_func)

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __call__(self, module, input, output):
        return self.hook_func(module, input, output)

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    @abstractmethod
    def hook_func(self, module, input, output):
        pass

    @classmethod
    def apply_to_modules(cls, modules, is_backward=False, **hook_kwargs):
        return HookList([cls(module, is_backward, **hook_kwargs) for module in modules])


class OutputStatsHook(Hook):
    def __init__(self, module, is_backward=False, mode='train'):
        super().__init__(module, is_backward)
        self.means = []
        self.stds = []
        self.hists = []
        assert mode in ('train', 'eval', 'all')
        self.mode = mode

    def hook_func(self, module, input, output):
        if self.mode == 'train' and not module.training:
            return
        if self.mode == 'eval' and module.training:
            return
        self.means.append(output.data.mean().cpu())
        self.stds.append(output.data.std().cpu())
        self.hists.append(output.data.cpu().histc(40, 0, 10))  # histc isn't implemented on the GPU


def get_train_hist(h):
    return torch.stack(h.hists).t().float().log1p()

def get_train_min(h):
    h1 = torch.stack(h.hists).t().float()
    return h1[:2].sum(0) / h1.sum(0)

def children(module):
    return list(module.children())

def has_children(module):
    try:
        next(module.children())
    except StopIteration:
        return False
    return True

class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p):
        self.val = p

    def forward(self, x):
        return x

def children_and_parameters(module):
    "Return the children of `module` and its direct parameters not registered in modules."
    children = list(module.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in module.children()],[])
    for p in module.parameters():
        if id(p) not in children_p:
            children.append(ParameterModule(p))
    return children

def flatten_model(model):
    "Return the list of all submodules and parameters of `model`"
    return sum(map(flatten_model, children_and_parameters(model)),[]) if has_children(model) else [model]

def has_params(module):
    "Check if `module` has at least one parameter"
    return len(list(module.parameters())) > 0


class HookManager(Callback):
    def __init__(self, hook_factory, modules=None, every=None, remove_end=True, is_backward=False, **hook_kwargs):
        super().__init__()
        self.hook_factory = hook_factory
        self.modules = modules
        self.every = every
        self.remove_end = remove_end
        self.is_backward = is_backward
        self.hook_kwargs = hook_kwargs
        self.hooks = None

    def register_hooks(self):
        self.hooks = self.hook_factory(self.modules, self.is_backward, **self.hook_kwargs)

    def remove(self):
        if self.hooks is not None:
            self.hooks.remove()

    def __del__(self):
        self.remove()

    def before_train_all_epochs(self, ns):
        if self.modules is None:
            self.modules = [module for module in flatten_model(self.model) if has_params(module)]
        if self.every is None:
            self.register_hooks()
        self.step_idx = 0

    def before_train_batch(self, ns):
        if self.every is not None and self.step_idx % self.every == 0:
            self.register_hooks()

    def after_train_batch(self, ns):
        if self.every is not None and self.step_idx % self.every == 0:
            self.remove()
        self.step_idx += 1

    def after_train_all_epochs(self, ns):
        if self.remove_end:
            self.remove()
