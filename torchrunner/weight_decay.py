import torch

from torchrunner.callback import Callback
from torchrunner.utils import format_opt_param


class WeightDecay(Callback):
    # Note: this overrides the default optimizer weight/bias decay
    def __init__(self, wd=0):
        super().__init__()
        self.this_wd = wd

    def before_train_init(self, ns):
        # temporarily nullify runner weight decay before optimizer initializes
        self.runner_wd = self.context.wd
        self.context.wd = None
        print("Warning: Weight decay callback overriding optimizer weight decay")

    def after_train_init(self, ns):
        # reinstate runner weight decay after optimizer initializes
        self.context.wd = self.runner_wd

    def before_update_params(self, ns):
        wd = format_opt_param(self.optimizer, 'wd', self.this_wd)
        with torch.no_grad():
            for pg, pg_wd in zip(self.optimizer.param_groups, wd):
                for p in pg['params']:
                    if p.grad is None:
                        continue
                    p.mul_(1 - pg['lr'] * pg_wd)