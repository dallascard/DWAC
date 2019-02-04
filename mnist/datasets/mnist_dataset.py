from torchvision.datasets import MNIST, FashionMNIST


# Wrappers for torchvision Datasets, but return (data, label, index) tuples, rather than just (data, label)

class MNISTwithIndices(MNIST):

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class FashionMNISTwithIndices(FashionMNIST):

    classes = ['T-shirt',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Boot']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

