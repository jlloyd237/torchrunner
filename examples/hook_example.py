from pathlib import Path

import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

from torchrunner.loss_metric import accuracy
from torchrunner.runner import Runner
from torchrunner.logger import Logger
from torchrunner.module import conv_net
from torchrunner.weight_decay import WeightDecay
from torchrunner.one_cycle import OneCycleScheduler
from torchrunner.hook import OutputStatsHook, HookManager, get_train_hist, get_train_min


def main():
    # specify data transforms
    train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # load data
    path = Path.cwd()
    print(path)
    train_ds = MNIST(path, train=True, download=True, transform=train_tfms)
    test_ds = MNIST(path, train=False, download=True, transform=test_tfms)

    # specify training/validation split
    val_pct = 0.2
    val_size = int(val_pct * len(train_ds))
    train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_size, val_size])
    val_ds.transform = test_tfms
    print(f"Training set size: {len(train_ds)}")
    print(f"Validation set size: {len(val_ds)}")
    print(f"Test set size: {len(test_ds)}")

    # set up data loaders
    batch_size = 64
    print(f"Batch size: {batch_size}")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    for label, dl in zip(['Training', 'Validation', 'Test'], [train_dl, val_dl, test_dl]):
        x_b, y_b = next(iter(dl))
        print(f"{label} set: Input shape: {list(x_b.shape)}, Output shape: {list(y_b.shape)}")

    # specify model
    model = conv_net(ni=1, no=10, nf=[16, 32, 64], nh=[128, 64])
    print(model)

    # specify loss function
    def loss_fn(logits, labels):
        return F.cross_entropy(logits, labels)

    with OutputStatsHook.apply_to_modules(model[:4]) as hooks:
        print(hooks)

    model = conv_net(ni=1, no=10, nf=[16, 32, 64], nh=[128, 64])
    optimizer = optim.Adam([{'params': model[:3].parameters()}, {'params': model[3:].parameters()}])
    run = Runner(model, train_dl=train_dl, val_dl=val_dl, loss_fn=loss_fn,
                 metric_fns=[accuracy], optimizer=optimizer,
                 callbacks=[Logger(print_every=1), WeightDecay(wd=1e-2), OneCycleScheduler()])

    with OutputStatsHook.apply_to_modules(model[:3]) as hooks:
        run.train(n_epochs=2, lr=(1e-3, 1e-2), device='cuda')

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        for h in hooks:
            means, stds, hists = h.means, h.stds, h.hists
            ax0.plot(means[:10])
            ax1.plot(stds[:10])
        plt.legend(range(6));

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        for h in hooks:
            means, stds, hists = h.means, h.stds, h.hists
            ax0.plot(means)
            ax1.plot(stds)
        plt.legend(range(6));

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for ax, h in zip(axes.flatten(), hooks[:3]):
            ax.imshow(get_train_hist(h)[:, :100], origin='lower')
            ax.axis('off')
        plt.tight_layout()

        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for ax, h in zip(axes.flatten(), hooks):
            ax.plot(get_train_min(h))
            ax.set_ylim(0, 1)
        plt.tight_layout()

        plt.show()

    # test hook manager callback
    model = conv_net(ni=1, no=10, nf=[16, 32, 64], nh=[128, 64])
    optimizer = optim.Adam([{'params': model[:3].parameters()}, {'params': model[3:].parameters()}])
    run = Runner(model, train_dl=train_dl, val_dl=val_dl, loss_fn=loss_fn,
                 metric_fns=[accuracy], optimizer=optimizer,
                 callbacks=[Logger(print_every=1), WeightDecay(wd=1e-2), OneCycleScheduler(),
                 HookManager(hook_factory=OutputStatsHook.apply_to_modules, modules=model[:3])])
    run.train(n_epochs=2, lr=(1e-3, 1e-2), device='cuda')

    hooks = run.callbacks['HookManager'].hooks
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    for h in hooks:
        means, stds, hists = h.means, h.stds, h.hists
        ax0.plot(means[:10])
        ax1.plot(stds[:10])
    plt.legend(range(6));

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    for h in hooks:
        means, stds, hists = h.means, h.stds, h.hists
        ax0.plot(means)
        ax1.plot(stds)
    plt.legend(range(6));

    fig, axes = plt.subplots(3, 1, figsize=(6, 6))
    for ax, h in zip(axes.flatten(), hooks[:3]):
        ax.imshow(get_train_hist(h)[:, :100], origin='lower')
        ax.axis('off')
    plt.tight_layout()

    fig, axes = plt.subplots(3, 1, figsize=(6, 6))
    for ax, h in zip(axes.flatten(), hooks):
        ax.plot(get_train_min(h))
        ax.set_ylim(0, 1)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
