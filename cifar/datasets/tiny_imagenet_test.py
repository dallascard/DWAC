import torch

from cifar.datasets.tiny_imagenet_dataset import TinyImageNet

train_dataset = TinyImageNet('./data/tiny', train=True, download=True)

cuda = False
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    **kwargs)
