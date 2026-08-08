"""
dataset.py

Downloads and preprocesses the MNIST handwritten digit image dataset for
CNN training. Applies normalization and data augmentation (small random
rotation + translation) to the training set, and only normalization to the
validation/test sets.

Note: MNIST's original host (yann.lecun.com) is frequently unreachable /
rate-limited, so this pipeline downloads from the widely-used GitHub mirror
https://github.com/fgnt/mnist instead. If the raw files already exist under
DATA_DIR/MNIST/raw, no download is attempted.
"""

import gzip
import os
import shutil
import urllib.request

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# MNIST global mean/std (standard, precomputed values), 1 channel (grayscale)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CLASS_NAMES = [str(i) for i in range(10)]  # digits 0-9

DATA_DIR = "./data"
MIRROR_BASE = "https://raw.githubusercontent.com/fgnt/mnist/master"
RAW_FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def _ensure_mnist_downloaded(root: str = DATA_DIR):
    """Downloads MNIST raw files from the GitHub mirror if not already present."""
    raw_dir = os.path.join(root, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    for name in RAW_FILES:
        extracted_path = os.path.join(raw_dir, name)
        if os.path.exists(extracted_path):
            continue

        gz_path = extracted_path + ".gz"
        url = f"{MIRROR_BASE}/{name}.gz"
        urllib.request.urlretrieve(url, gz_path)

        with gzip.open(gz_path, "rb") as f_in, open(extracted_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def get_transforms():
    """Returns (train_transform, eval_transform)."""
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),                         # data augmentation
        transforms.RandomAffine(0, translate=(0.1, 0.1)),       # data augmentation
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),            # normalization
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(batch_size: int = 64, train_subset: int = None, test_subset: int = None,
                     val_fraction: float = 0.1, seed: int = 42, num_workers: int = 0):
    """
    Downloads MNIST (if not already present in DATA_DIR) and returns
    train/val/test DataLoaders.

    Args:
        batch_size: batch size for all loaders.
        train_subset: if set, only use this many samples from the training
            pool (before the train/val split) — useful for faster iteration
            on limited compute. None uses the full 60,000 training images.
        test_subset: if set, only use this many samples from the test set.
        val_fraction: fraction of the training pool held out for validation.
        seed: random seed for the train/val split.
        num_workers: DataLoader worker processes.
    """
    _ensure_mnist_downloaded(DATA_DIR)

    train_transform, eval_transform = get_transforms()

    full_train = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=train_transform)
    full_train_eval = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=eval_transform)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=eval_transform)

    if train_subset is not None:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(full_train), generator=g)[:train_subset]
        full_train = torch.utils.data.Subset(full_train, idx)
        full_train_eval = torch.utils.data.Subset(full_train_eval, idx)

    if test_subset is not None:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(test_set), generator=g)[:test_subset]
        test_set = torch.utils.data.Subset(test_set, idx)

    n_total = len(full_train)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=g)

    train_set = torch.utils.data.Subset(full_train, train_idx.indices)
    # Validation set uses the eval (non-augmented) transform pipeline
    val_set = torch.utils.data.Subset(full_train_eval, val_idx.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
