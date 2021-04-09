import math
from collections import defaultdict
from functools import partial

import numpy as np

import torch

from torchrunner.callback import Callback


class OptimLRScheduler(Callback):
    def __init__(self, sched_cls, **sched_kwargs):
        super().__init__()
        self.sched_cls = sched_cls
        self.sched_kwargs = sched_kwargs
        self.step_idx = 0

    def before_train_all_epochs(self, ns):
        self.scheduler = self.sched_cls(self.optimizer, **self.sched_kwargs)
        self.history = defaultdict(list)
        print("Warning: LR scheduler callback overriding runner/optimizer learning rates")

    def before_train_batch(self, ns):
        self.history['step'].append(self.step_idx + 1)
        self.history['lr'].append(self.scheduler.get_last_lr())

    def after_train_batch(self, ns):
        self.scheduler.step()
        self.step_idx += 1


def annealer(f):
    def _inner(start, end):
        return partial(f, start, end)
    return _inner

@annealer
def lin_schedule(start, end, pos):
    return start + pos * (end - start)

@annealer
def cos_schedule(start, end, pos):
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2

@annealer
def null_schedule(start, end, pos):
    return start

@annealer
def exp_schedule(start, end, pos):
    return start * (end / start) ** pos

def cos_1_cycle_schedule(start, high, end):
    return [cos_schedule(start, high), cos_schedule(high, end)]

def combine_schedules(pcts, scheds):
    pcts = torch.tensor(pcts)
    assert sum(pcts) == 1.
    pcts = torch.cat((torch.tensor([0.]), pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero(as_tuple=False).max()
        if idx == 2:
            idx = 1
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)
    return _inner


class OptimParamScheduler(Callback):
    def __init__(self, sched_funcs):
        super().__init__()
        self.sched_fns = sched_funcs

    def before_train_all_epochs(self, ns):
        self.n_steps = ns.n_epochs * len(self.train_dl.dataset) // self.train_dl.batch_size
        self.step_idx = 0
        self.history = defaultdict(list)
        print("Warning: LR scheduler callback overriding runner/optimizer learning rates")

    def before_train_batch(self, ns):
        self.history['step'].append(self.step_idx + 1)
        for param_name, sched_fn in self.sched_fns.items():
            for pg_idx, pg in enumerate(self.optimizer.param_groups):
                if isinstance(sched_fn, (tuple, list, np.ndarray, torch.Tensor)):
                    pg[param_name] = sched_fn[pg_idx](self.step_idx / self.n_steps)
                else:
                    pg[param_name] = sched_fn(self.step_idx / self.n_steps)
            self.history[param_name].append([pg[param_name] for pg in self.optimizer.param_groups])
        self.step_idx += 1
