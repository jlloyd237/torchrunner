from pathlib import Path

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

from torchrunner.loss_metric import accuracy
from torchrunner.runner import Runner
from torchrunner.logger import Logger
from torchrunner.weight_decay import WeightDecay
from torchrunner.module import conv_net
from torchrunner.scheduler import null_schedule, lin_schedule, cos_schedule, exp_schedule, \
    combine_schedules, OptimParamScheduler


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

    # specify optimizer
    optimizer = optim.Adam([{'params': model[0:3].parameters()}, {'params': model[3:8].parameters()}])

    # plot schedules
    torch.Tensor.ndim = property(lambda x: len(x.shape))    # monkey patch for plotting tensors

    annealings = "null linear cos exp".split()
    a = torch.arange(0, 100)
    p = torch.linspace(0.01, 1, 100)
    fns = [null_schedule, lin_schedule, cos_schedule, exp_schedule]
    plt.figure()
    for fn, t in zip(fns, annealings):
        f = fn(2, 1e-2)
        plt.plot(a, [f(o) for o in p], label=t)
    plt.legend()

    # specify schedules for learning rates
    sched_1 = combine_schedules([0.3, 0.7], [cos_schedule(1e-4, 1e-3), cos_schedule(1e-3, 1e-5)])
    plt.figure()
    plt.plot(a, [sched_1(o) for o in p])

    sched_2 = combine_schedules([0.3, 0.7], [cos_schedule(1e-3, 1e-2), cos_schedule(1e-2, 1e-4)])
    plt.figure()
    plt.plot(a, [sched_2(o) for o in p])

    plt.show()

    # execute training loop
    run = Runner(model, train_dl=train_dl, val_dl=val_dl, loss_fn=loss_fn,
                 metric_fns=[accuracy], optimizer=optimizer,
                 callbacks=[Logger(print_every=1), WeightDecay(wd=1e-2),
                            OptimParamScheduler({'lr': [sched_1, sched_2]})])
    run.train(n_epochs=3, device='cuda')

    # plot learning rates
    lr = run.callbacks['OptimParamScheduler'].history['lr']
    lr = list(zip(*lr))
    plt.figure()
    plt.plot(lr[0])
    plt.figure()
    plt.plot(lr[1])
    plt.show()


if __name__ == '__main__':
    main()
