"""Datasets, transforms and the validation split.

Five image-classification datasets. The validation split is carved out of the training
split in a class-balanced way with a fixed seed that is independent of the training seed,
so every cell sees the same validation set. The test split is never touched during
training or selection.
"""
import copy, os
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

DATA_ROOT = os.environ.get('PAFT_DATA', './data')

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
FLIP = {'cifar100': True, 'svhn': False, 'stl10': True, 'tinyimagenet': True, 'cub200': True}
N_CLASSES = {'cifar100': 100, 'svhn': 10, 'stl10': 10, 'tinyimagenet': 200, 'cub200': 200}


def make_transforms(dataset):
    mean, std = IMNET
    tr = [transforms.Resize(256), transforms.RandomCrop(224)]
    if FLIP[dataset]:
        tr.append(transforms.RandomHorizontalFlip())
    tr += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    te = [transforms.Resize(256), transforms.CenterCrop(224),
          transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(tr), transforms.Compose(te)


class CUB200(torch.utils.data.Dataset):
    """Caltech-UCSD Birds 200-2011 with the official split and zero-based labels."""
    def __init__(self, root, train=True, transform=None):
        base = os.path.join(root, 'CUB_200_2011')
        self.img_dir = os.path.join(base, 'images'); self.transform = transform
        with open(os.path.join(base, 'images.txt')) as f:
            id2path = dict(l.split() for l in f)
        with open(os.path.join(base, 'image_class_labels.txt')) as f:
            id2lab = {i: int(c) - 1 for i, c in (l.split() for l in f)}
        with open(os.path.join(base, 'train_test_split.txt')) as f:
            id2tr = {i: int(t) for i, t in (l.split() for l in f)}
        self.samples = [(id2path[i], id2lab[i]) for i in id2path if (id2tr[i] == 1) == train]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        rel, y = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, rel)).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, y


def _labels_of(ds):
    """Read the labels without loading the images, for stratification."""
    for attr in ('targets', 'labels'):
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))
    if hasattr(ds, 'samples'):
        return np.asarray([y for _, y in ds.samples])
    raise TypeError(f'could not read labels: {type(ds)}')


def stratified_val_split(train_ds, fraction, seed=0):
    """Carve a class-balanced validation set out of the training split under a fixed seed that is
    independent of the training seed.
    """
    y = _labels_of(train_ds)
    rng = np.random.RandomState(seed)
    val_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = max(1, int(round(fraction * len(idx))))
        val_idx.extend(rng.permutation(idx)[:k])
    val_idx = np.sort(np.asarray(val_idx))
    mask = np.ones(len(y), bool); mask[val_idx] = False
    return np.where(mask)[0], val_idx


def make_datasets(dataset, root=None):
    root = root or DATA_ROOT
    ttr, tte = make_transforms(dataset)
    if dataset == 'cifar100':
        tr = datasets.CIFAR100(root, True, download=False, transform=ttr)
        te = datasets.CIFAR100(root, False, download=False, transform=tte)
    elif dataset == 'svhn':
        tr = datasets.SVHN(root, split='train', download=False, transform=ttr)
        te = datasets.SVHN(root, split='test', download=False, transform=tte)
    elif dataset == 'stl10':
        tr = datasets.STL10(root, split='train', download=False, transform=ttr)
        te = datasets.STL10(root, split='test', download=False, transform=tte)
    elif dataset == 'tinyimagenet':
        tr = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200/train'), transform=ttr)
        te = datasets.ImageFolder(os.path.join(root, 'tiny-imagenet-200/val'), transform=tte)
    elif dataset == 'cub200':
        tr = CUB200(root, True, ttr)
        te = CUB200(root, False, tte)
    else:
        raise ValueError(f'unknown dataset: {dataset}')
    return tr, te, N_CLASSES[dataset]


def make_val_subset(full_train, dataset, fraction, split_seed):
    """The validation subset under the test transform, that is without augmentation."""
    tr_idx, va_idx = stratified_val_split(full_train, fraction, split_seed)
    val_base = copy.copy(full_train)
    val_base.transform = make_transforms(dataset)[1]
    from torch.utils.data import Subset
    return tr_idx, va_idx, Subset(val_base, va_idx.tolist())
