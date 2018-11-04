import matplotlib as mpl
mpl.use('Agg')

import os
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from torchvision import transforms

from mnist.datasets.mnist_dataset import MNISTwithIndices, FashionMNISTwithIndices
from eval.find_closest import find_closest


def main():
    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--root-dir', default='./data',
                      help='Root dir: default=%default')
    parser.add_option('--fashion', action="store_true", dest="fashion", default=False,
                      help='Use FashionMNIST: default=%default')
    parser.add_option('-i', dest='index', default=0,
                      help='Index of test point to lookup: default=%default')
    parser.add_option('--distf', dest='dist_function', default='Gauss',
                      help='Distance function [Gauss|Laplace|InverseQuad]: default=%default')
    parser.add_option('--gamma', dest='gamma', default=1.0,
                      help='Parameter of distance function: default=%default')
    #parser.add_option('-n', action="store_true", dest="adaptive", default=False,
    #                  help='Use adaptive binning: default=%default')

    (options, args) = parser.parse_args()
    indir = args[0]

    root_dir = options.root_dir
    dist_function = options.dist_function
    gamma = float(options.gamma)
    index = int(options.index)
    fashion = options.fashion

    test_index, closest_train, dists = find_closest(indir, dist_function, gamma, index)

    if fashion:
        train_dataset = FashionMNISTwithIndices(
            os.path.join(root_dir, 'fashion'), train=True,
            download=True
        )
        test_dataset = FashionMNISTwithIndices(
            os.path.join(root_dir, 'fashion'), train=False,
            download=True
        )
        ood_dataset = MNISTwithIndices(
            os.path.join(root_dir, 'mnist'), train=False,
            download=True
        )
    else:
        train_dataset = MNISTwithIndices(
            os.path.join(root_dir, 'mnist'), train=True,
            download=True
        )
        test_dataset = MNISTwithIndices(
            os.path.join(root_dir, 'mnist'), train=False,
            download=True
        )
        ood_dataset = FashionMNISTwithIndices(
            os.path.join(root_dir, 'mnist'), train=False,
            download=True
        )

    fig = plt.figure(figsize=(6, 3))
    gs = GridSpec(nrows=2, ncols=8)

    test = np.load(os.path.join(indir, 'test.npz'))
    test_labels = test['labels']
    true_label = test_dataset.classes[test_dataset.test_labels[test_index]]

    ax1 = fig.add_subplot(gs[0, 3:6])
    #index = test_index
    image = test_dataset.test_data[test_index]
    two_d = np.reshape(image, (28, 28))
    ax1.imshow(two_d, interpolation='nearest', cmap='gray')
    ax1.set_xlabel("{:s")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("{:s}".format(true_label))

    train = np.load(os.path.join(indir, 'train.npz'))
    train_labels = train['labels']
    train_indices = list(train['indices'])

    for i in range(4):
        ax = fig.add_subplot(gs[1, i*2:i*2+2])
        index = closest_train[i]
        image = train_dataset.train_data[index]
        true_label = train_dataset.classes[train_dataset.train_labels[index]]
        two_d = np.reshape(image, (28, 28))
        ax.imshow(two_d, interpolation='nearest', cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        weight = dists[i]
        ax.set_xlabel("{:s}\n{:.4f}".format(true_label, weight))

    print(dists[:10])

    plt.savefig('mnist_closest.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
