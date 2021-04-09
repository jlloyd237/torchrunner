from pathlib import Path

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

from torchrunner.module import conv_net
from torchrunner.summary import model_summary


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

    # execute training loop
    model_summary(model, train_dl, find_all=True, print_mod=True)


if __name__ == '__main__':
    main()
