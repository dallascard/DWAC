import os
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from mnist.mnist_dataset import MNISTwithIndices, FashionMNISTwithIndices


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--root-dir', default='./data',
                      help='Root dir: default=%default')
    parser.add_option('--train', action="store_true", dest="train", default=False,
                      help='Index into the training set (instead of test): default=%default')
    parser.add_option('--fashion', action="store_true", dest="fashion", default=False,
                      help='Use FashionMNIST: default=%default')
    parser.add_option('-i', type=int, dest='index', default=0,
                      help='Index to print: default=%default')

    (options, args) = parser.parse_args()
    fashion = options.fashion
    train = options.train
    index = options.index

    if fashion:
        train_dataset = FashionMNISTwithIndices(
            './data/fashion', train=True,
            download=True
        )
        test_dataset = FashionMNISTwithIndices(
            './data/fashion', train=False,
            download=True
        )
        ood_dataset = MNISTwithIndices(
            './data/mnist', train=False,
            download=True
        )
    else:
        train_dataset = MNISTwithIndices(
            './data/mnist', train=True,
            download=True
        )
        test_dataset = MNISTwithIndices(
            './data/mnist', train=False,
            download=True
        )
        ood_dataset = FashionMNISTwithIndices(
            './data/fashion', train=False,
            download=True
        )

    options.cuda = False
    if train:
        image = train_dataset.train_data[index]
    else:
        image = test_dataset.test_data[index]

    two_d = np.reshape(image, (28, 28))
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
