import matplotlib as mpl
mpl.use('Agg')

import os
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt


def main():
    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--split', action="store_true", dest="split", default=False,
                      help='Split into multiple plots: default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()
    indir = args[0]
    train_file = os.path.join(indir, 'train.npz')

    seed = int(options.seed)
    split = options.split
    np.random.seed(seed)

    train_data = np.load(train_file)

    train_z = train_data['z']
    train_labels = train_data['labels']
    n_train, dz = train_z.shape
    print(train_z.shape)

    n_classes = np.max(train_labels+1)

    # scatter the labels
    train_labels = scatter(train_labels, n_classes)

    if split:
        fig, axes = plt.subplots(nrows=1, ncols=n_classes, figsize=(n_classes*2, 2), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots()
    for k in range(n_classes):
        indices = np.array(train_labels[:, k], dtype=bool)
        if split:
            axes[k].scatter(train_z[indices, 0], train_z[indices, 1], c='k', alpha=0.5)
        else:
            ax.scatter(train_z[indices, 0], train_z[indices, 1], s=1, alpha=0.5)

    plt.savefig('test.pdf')


def scatter(labels, n_classes):
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        n_items = len(labels)
        temp = np.zeros((n_items, n_classes), dtype=int)
        temp[np.arange(n_items), labels] = 1
        labels = temp
    return labels


if __name__ == '__main__':
    main()
