import os
from optparse import OptionParser

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def main():
    usage = "%prog exp_dir [exp_dir2 ...]"
    parser = OptionParser(usage=usage)
    parser.add_option('-k', dest='k', default=10,
                      help='Number of neighbours to use: default=%default')
    parser.add_option('--batch_size', dest='batch_size', default=100,
                      help='Size of batches of training points: default=%default')
    parser.add_option('--distf', dest='dist_function', default='Gauss',
                      help='Distance function [Gauss|Laplace|InverseQuad]: default=%default')
    parser.add_option('--gamma', dest='gamma', default=1.0,
                      help='Parameter of distance function: default=%default')
    parser.add_option('--threads', dest='threads', default=8,
                      help='Number of threads to use in computing distances: default=%default')
    parser.add_option('--fast', action="store_true", dest="fast", default=False,
                      help='Make fast (but potentially include greater than k neighbours): default=%default')

    (options, args) = parser.parse_args()
    indirs = args

    k = int(options.k)
    batch_size = int(options.batch_size)
    dist_function = options.dist_function
    gamma = float(options.gamma)
    threads = int(options.threads)

    acc_vals = []
    agreements = []
    for indir in indirs:
        train_file = os.path.join(indir, 'train.npz')
        test_file = os.path.join(indir, 'test.npz')

        acc, agreement = eval_topk(train_file, test_file, k, batch_size, dist_function, gamma, threads)
        acc_vals.append(acc)
        agreements.append(agreement)

    print("Mean ACC = {:.4f}".format(np.mean(acc_vals)))
    print("Mean agreement = {:.4f}".format(np.mean(agreements)))


def eval_topk(train_file, test_file, k, batch_size, dist_function, gamma, threads):


    train_data = np.load(train_file)
    test_data = np.load(test_file)

    train_z = train_data['z']
    train_labels = train_data['labels']
    n_train, dz = train_z.shape
    print(train_z.shape)

    test_z = test_data['z']
    test_labels = test_data['labels']
    test_pred_probs = test_data['pred_probs']
    test_confs = test_data['confs']
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

    sparsity = []
    correct = 0
    agreed = 0
    total_points = 0
    mae = 0.0
    n_batches = int(np.ceil(n_test / batch_size))
    for b in range(n_batches):
        if b < n_batches - 1:
            indices = np.arange(b * batch_size, (b+1) * batch_size)
        else:
            indices = np.arange(b * batch_size, n_test)
        batch_size_b = len(indices)
        test_points = test_z[indices, :]

        if dist_function == 'Gauss':
            dists = pairwise_distances(test_points, train_z, metric='sqeuclidean', n_jobs=threads)
            dists = np.exp(-gamma * dists)
        elif dist_function == 'Laplace':
            dists = pairwise_distances(test_points, train_z, metric='l1', n_jobs=threads)
            dists = 0.5 * np.exp(-1.0 * dists)
        elif dist_function == 'InverseQuad':
            dists = pairwise_distances(test_points, train_z, metric='sqeuclidean', n_jobs=threads)
            dists = 1.0/(dists + gamma)
        else:
            raise ValueError("Distance function not recognized.")

        # sort each row by weight (smallest first)
        order = np.argsort(dists, axis=1)

        # trying alternate masking
        mask = np.zeros_like(dists)
        for j in range(len(indices)):
            mask[np.ones(k, dtype=int) * j, order[j, -k:]] = 1

        # hopefully, we get exactly k points per row
        print("{:d}/{:d}".format(b, n_batches), np.min(mask.sum(1)), np.max(mask.sum(1)))

        # compute the weighted sums per class using only these points
        class_dists = np.dot(mask * dists, train_labels)
        class_dist_sums = class_dists.sum(1)
        topk_probs = class_dists / class_dist_sums.reshape((batch_size_b, 1))

        # measure accuracy
        correct += np.sum(class_dists.argmax(1) == test_labels[indices, :].argmax(1))
        # also measure agreement with predicted labels
        agreed += np.sum(class_dists.argmax(1) == test_pred_probs[indices, :].argmax(1))
        total_points += batch_size_b
        test_preds = test_pred_probs[indices, :].argmax(1)
        mae += np.sum(np.abs(topk_probs[np.arange(batch_size_b), test_preds] - test_pred_probs[indices, test_preds])) / float(batch_size_b)

    print(total_points, n_test)
    acc = correct / float(total_points)
    print(acc)
    agreement = agreed / float(total_points)
    print(agreement)
    return acc, agreement


if __name__ == '__main__':
    main()
