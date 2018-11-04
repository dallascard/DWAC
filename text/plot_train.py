import matplotlib as mpl
mpl.use('Agg')

import os
import itertools

from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from text.common import load_dataset


def main():
    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--root-dir', default='./data',
                      help='Root dir: default=%default')
    parser.add_option('--split', action="store_true", dest="split", default=False,
                      help='Split into multiple plots: default=%default')
    parser.add_option('--dataset', default='stackoverflow',
                      help='Dataset: default=%default')
    parser.add_option('--subset', default=None,
                      help='Subset (for amazon and framing): default=%default')
    parser.add_option('--batch-size', default=1000,
                      help='Batch size for plotting: default=%default')
    parser.add_option('--max-points', default=200000,
                      help='Max number of points to plot: default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()
    indir = args[0]
    train_file = os.path.join(indir, 'train.npz')

    root_dir = options.root_dir
    seed = int(options.seed)
    dataset = options.dataset
    subset = options.subset
    batch_size = int(options.batch_size)
    max_points = int(options.max_points)
    split = options.split
    np.random.seed(seed)

    train_data = np.load(train_file)

    train_z = train_data['z']
    train_labels = train_data['labels']
    n_train, dz = train_z.shape
    print(train_z.shape)

    train_dataset, test_dataset, ood_dataset = load_dataset(root_dir, dataset, subset, lower=False)

    data = train_dataset
    labels = data.classes

    n_classes = np.max(train_labels+1)

    # scatter the labels
    train_labels = scatter(train_labels, n_classes)

    if split:
        fig, axes = plt.subplots(nrows=1, ncols=n_classes, figsize=(n_classes*2, 2), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots()

    palette = itertools.cycle(seaborn.color_palette("hls", 5))

    order = np.arange(n_train)
    np.random.shuffle(order)

    n_batches = int(max_points // batch_size)
    for b in range(n_batches):
        indices = order[b * batch_size: (b+1) * batch_size]
        batch_z = train_z[indices, :]
        batch_labels = train_labels[indices, :]
        for k in range(n_classes):
            indices = np.array(batch_labels[:, k], dtype=bool)
            if b == 0:
                label = labels[k]
            else:
                label = None
            if split:
                axes[k].scatter(batch_z[indices, 0], batch_z[indices, 1], s=1, color=next(palette), alpha=0.5, label=label)
            else:
                ax.scatter(batch_z[indices, 0], batch_z[indices, 1], s=1, color=next(palette), alpha=0.5, label=label)

    if split:
        for k in range(n_classes):
            axes[k].legend()
    else:
        ax.legend(markerscale=6)
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
