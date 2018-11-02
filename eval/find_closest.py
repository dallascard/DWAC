import os
from optparse import OptionParser

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def main():
    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--distf', dest='dist_function', default='Gauss',
                      help='Distance function [Gauss|Laplace|InverseQuad]: default=%default')
    parser.add_option('--gamma', dest='gamma', default=1.0,
                      help='Parameter of distance function: default=%default')
    parser.add_option('-i', dest='index', default=0,
                      help='Index of point to lookup: default=%default')

    (options, args) = parser.parse_args()
    indir = args[0]

    dist_function = options.dist_function
    gamma = float(options.gamma)
    index = int(options.index)

    test_index, closest_train, dists = find_closest(indir, dist_function, gamma, index)

    print("Test index:", test_index)
    for i in range(10):
        print(i, closest_train[i], dists[i])


def find_closest(indir, dist_function, gamma, index):

    train_file = os.path.join(indir, 'train.npz')
    test_file = os.path.join(indir, 'test.npz')

    train_data = np.load(train_file)
    test_data = np.load(test_file)

    train_z = train_data['z']
    train_labels = train_data['labels']
    n_train, dz = train_z.shape
    train_indices = train_data['indices']
    print(train_z.shape)

    test_z = test_data['z']
    test_labels = test_data['labels']
    test_pred_probs = test_data['pred_probs']
    test_confs = test_data['confs']
    test_indices = test_data['indices']
    print(test_confs.shape)

    n_test, n_classes = test_pred_probs.shape

    # scatter the labels
    if len(train_labels.shape) == 1 or train_labels.shape[1] == 1:
        temp = np.zeros((n_train, n_classes), dtype=int)
        temp[np.arange(n_train), train_labels] = 1
        train_labels = temp

    if len(test_labels.shape) == 1 or test_labels.shape[1] == 1:
        temp = np.zeros((n_test, n_classes), dtype=int)
        temp[np.arange(n_test), test_labels] = 1
        test_labels = temp

    test_index = list(test_indices).index(index)
    test_point = test_z[test_index, :].reshape([1, dz])

    print("True label", test_labels[test_index])
    print("Pred probs", test_pred_probs[test_index, :])
    print("Dists:", test_confs[test_index, :])

    if dist_function == 'Gauss':
        dists = pairwise_distances(test_point, train_z, metric='sqeuclidean', n_jobs=16)
        dists = np.exp(-gamma * dists)
    elif dist_function == 'Laplace':
        dists = pairwise_distances(test_point, train_z, metric='l1', n_jobs=16)
        dists = 0.5 * np.exp(-1.0 * dists)
    elif dist_function == 'InverseQuad':
        dists = pairwise_distances(test_point, train_z, metric='sqeuclidean', n_jobs=16)
        dists = 1.0/(dists + gamma)
    else:
        raise ValueError("Distance function not recognized.")

    dists = dists.reshape((n_train, ))
    order = list(np.argsort(dists))
    order.reverse()

    return test_indices[test_index], [train_indices[i] for i in order], [dists[i] for i in order]


if __name__ == '__main__':
    main()
