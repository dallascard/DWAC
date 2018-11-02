from torchvision.datasets import CIFAR10

class CIFAR10withIndices(CIFAR10):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index
