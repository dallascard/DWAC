import os

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from cifar.datasets.cifar_dataset import CIFAR10withIndices
from cifar.datasets.tiny_imagenet_dataset import TinyImageNet


def load_data(args):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_ood = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10withIndices(root=os.path.join(args.root_dir, 'cifar', 'processed'),
                    train=True, download=True, transform=transform_train)
    dev_dataset = CIFAR10withIndices(root=os.path.join(args.root_dir, 'cifar', 'processed'),
                    train=True, download=True, transform=transform_test)
    test_dataset = CIFAR10withIndices(root=os.path.join(args.root_dir, 'cifar', 'processed'),
                    train=False, download=True, transform=transform_test)
    ood_dataset = TinyImageNet(root=os.path.join(args.root_dir, 'tiny'),
                    train=True, download=True, transform=transform_ood)

    n_train = len(train_dataset)
    indices = list(range(n_train))
    split = int(np.floor(args.dev_prop * n_train))
    np.random.shuffle(indices)
    train_idx, dev_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    dev_sampler = SubsetRandomSampler(dev_idx)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.device != 'cpu' else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
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
        dev_dataset,
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

