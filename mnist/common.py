import os

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from mnist.datasets.mnist_dataset import MNISTwithIndices, FashionMNISTwithIndices


def load_mnist_data(args):
    mnist_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    fashion_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # load one dataset as train and test, and the other as an out-of-domain test set
    if args.fashion:
        train_dataset = FashionMNISTwithIndices(
            os.path.join(args.root_dir, 'fashion'), train=True,
            download=True, transform=fashion_transform,
        )
        test_dataset = FashionMNISTwithIndices(
            os.path.join(args.root_dir, 'fashion'), train=False,
            download=True, transform=fashion_transform,
        )
        ood_dataset = MNISTwithIndices(
            os.path.join(args.root_dir, 'mnist'), train=False,
            download=True, transform=mnist_transform,
        )
    else:
        train_dataset = MNISTwithIndices(
            os.path.join(args.root_dir, 'mnist'), train=True,
            download=True, transform=mnist_transform,
        )
        test_dataset = MNISTwithIndices(
            os.path.join(args.root_dir, 'mnist'), train=False,
            download=True, transform=mnist_transform,
        )
        ood_dataset = FashionMNISTwithIndices(
            os.path.join(args.root_dir, 'fashion'), train=False,
            download=True, transform=fashion_transform,
        )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.dev_prop * num_train))

    np.random.shuffle(indices)

    train_idx, dev_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    dev_sampler = SubsetRandomSampler(dev_idx)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device != 'cpu' else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        shuffle=False,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)
    ref_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        sampler=train_sampler,
        shuffle=False,
        **kwargs)
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)

    return train_loader, dev_loader, test_loader, ref_loader, ood_loader


def to_numpy(var, device):
    if device != 'cpu':
        return var.cpu().data.numpy()
    else:
        return var.data.numpy()
