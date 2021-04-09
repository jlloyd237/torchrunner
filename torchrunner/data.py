from pathlib import Path
import urllib
import tarfile

import PIL
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


def untar_data(url, dest):
    print(f"Downloading and extracting data from {url} to {dest} ...")
    ftp_stream = urllib.request.urlopen(url)
    tar = tarfile.open(fileobj=ftp_stream, mode='r|gz')
    tar.extractall(path=dest)
    print("Finished downloading and extracting data")

def df_names_to_idx(df, cols):
    if not isinstance(cols, (tuple, list)):
        cols = [cols]
    if isinstance(cols[0], int):
        return cols
    return [df.columns.get_loc(c) for c in cols]

def load_image(filepath, convert_mode, transform=None):
    image = PIL.Image.open(filepath).convert(convert_mode)
    if transform:
        image = transform(image)
    return image

def calc_image_stats(dataset):
    n_channels = dataset[0][0].shape[0]
    mean, std = torch.zeros(n_channels), torch.zeros(n_channels)
    data_loader = DataLoader(dataset)
    for images, _ in data_loader:  # alternatively, could just select a single batch at random
        mean += images.mean(axis=(2, 3)).sum(axis=0)
        std += images.std(axis=(2, 3)).sum(axis=0)
    mean /= len(data_loader.dataset)
    std /= len(data_loader.dataset)
    return mean.data, std.data

def denormalize(image, std, mean):
    image = image.numpy().transpose(1, 2, 0)  # PIL images have channel last
    image = (image * std.numpy() + mean.numpy()).clip(0, 1)
    return image

def display_batch(images, labels, mean, std, rows=5, cols=3, cmap='gray'):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten()
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(denormalize(image, mean, std).squeeze(), cmap=cmap)
        ax.set_axis_off()
        ax.set_title(str(label.numpy()), fontsize=10)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.suptitle("Augmented training set images", fontsize=12)
    plt.show()

class DataframeImageList(Dataset):
    def __init__(self, df, path, fname_col=0, label_cols=1, suffix='', convert_mode='RGB',
                 transform=None, label_transform=None):
        self.df = df
        self.path = Path(path)
        self.fname_col = fname_col
        self.label_cols = label_cols
        self.suffix = suffix
        self.convert_mode = convert_mode
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.iloc[idx, df_names_to_idx(self.df, self.fname_col)].item() + self.suffix
        image = load_image(self.path/fname, self.convert_mode, self.transform)
        labels = self.df.iloc[idx, df_names_to_idx(self.df, self.label_cols)].values.tolist()
        if self.label_transform:
            labels = self.label_transform(labels)
        return image, labels

