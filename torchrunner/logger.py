from collections import defaultdict

import numpy as np
import torch

from torchrunner.callback import Callback


class Logger(Callback):
    def __init__(self, print_every=100):
        super().__init__()
        self.print_every = print_every

    def before_train_all_epochs(self, ns):
        self.history = defaultdict(list)

    def after_train_epoch(self, ns):
        self.history['epoch'].append(ns.epoch_idx + 1)

        if self.val_dl:
            train_loss, train_metrics, val_loss, val_metrics = ns.ret
        else:
            train_loss, train_metrics = ns.ret

        self.history['loss'].append(train_loss)
        for i, metric_fn in enumerate(self.metric_fns):
            self.history[metric_fn.__name__].append(train_metrics[i])

        if self.val_dl:
            self.history['val_loss'].append(val_loss)
            for i, metric_fn in enumerate(self.metric_fns):
                self.history['val_' + metric_fn.__name__].append(val_metrics[i])

        if ns.epoch_idx % self.print_every == 0:
            with np.printoptions(precision=4):
                for i, (name, series) in enumerate(self.history.items(), 1):
                    val = series[-1].numpy().reshape(-1) if isinstance(series[-1], torch.Tensor) else series[-1]
                    print(f"{name}: {val}", end=', ' if i < len(self.history) else '\n')
