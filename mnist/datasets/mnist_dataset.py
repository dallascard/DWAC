from torchvision.datasets import MNIST, FashionMNIST


# Wrappers for torchvision Datasets, but return (data, label, index) tuples, rather than just (data, label)

class MNISTwithIndices(MNIST):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class FashionMNISTwithIndices(FashionMNIST):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

