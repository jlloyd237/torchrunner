import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import torch

from torchrunner.callback import Callback, CancelOpException
from torchrunner.scheduler import exp_schedule


class LearningRateFinder(Callback):

    def __init__(self, max_steps=100, min_lr=1e-7, max_lr=10, filter_window=11):
        super().__init__()
        self.max_steps, self.min_lr, self.max_lr, self.filter_window = max_steps, min_lr, max_lr, filter_window
        if isinstance(min_lr, (tuple, list, np.ndarray, torch.Tensor)) :
            self.lr_sched = [exp_schedule(min_lr_i, max_lr_i) \
                             for (min_lr_i, max_lr_i) in zip(min_lr, max_lr)]
        else:
            self.lr_sched = exp_schedule(min_lr, max_lr)
        self.best_loss = float('inf')
        self.avg_loss = 0.0
        self.tmp_model_file_path = Path.cwd()/"_tmp.pth"

    def before_train_init(self, ns):
        self.save(self.tmp_model_file_path)
        ns.reset_opt = True

    def before_train_all_epochs(self, ns):
        ns.n_epochs = self.max_steps
        self.step_idx = 0
        self.history = defaultdict(list)
        print("Warning: LR finder callback overriding runner/optimizer learning rates")
        print("Scanning learning rates ...")

    def before_train_batch(self, ns):
        self.step_idx += 1
        pos = self.step_idx / self.max_steps
        lr = self.lr_sched(pos)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.history['lr'].append([pg['lr'] for pg in self.optimizer.param_groups])

    def after_train_batch(self, ns):
        loss, metrics = ns.ret
        self.history['loss'].append(loss)

        if (self.step_idx >= self.max_steps) or (loss > 4 * self.best_loss):
            raise CancelOpException('train_all_epochs')

        if loss < self.best_loss:
            self.best_loss = loss

    def cancel_train_all_epochs(self, ns):
        # Record Savitzky-Golay filtered batch loss
        self.history['sg_loss'] = savgol_filter(self.history['loss'], window_length=self.filter_window, polyorder=2)
        self.optimizer.zero_grad()
        if self.tmp_model_file_path.exists():
            self.load(self.tmp_model_file_path)
            os.remove(self.tmp_model_file_path)
        print("Finished!")

def lr_find(runner, max_steps=100, min_lr=1e-7, max_lr=10, filter_window=9):
    assert runner.train_dl is not None
    assert runner.loss_fn is not None
    assert runner.optimizer is not None

    lr_finder = LearningRateFinder(max_steps, min_lr, max_lr, filter_window)
    runner.add_callbacks([lr_finder])

    runner.train_init(reset_opt=False)
    runner.train_all_epochs(n_epochs=1)

    runner.remove_callbacks([lr_finder])

    # plot lr
    lr = lr_finder.history['lr']
    lr = list(zip(*lr))
    plt.plot(lr[-1])
    plt.xlabel("step")
    plt.ylabel("lr")

    # plot loss
    loss = lr_finder.history['loss']
    plt.figure()
    plt.plot(lr[-1], loss)
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("loss")

    # plot Savitzky-Golay filtered loss
    sg_loss = lr_finder.history['sg_loss']
    plt.figure()
    plt.plot(lr[-1], sg_loss)
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("loss (sg)")

    plt.show()