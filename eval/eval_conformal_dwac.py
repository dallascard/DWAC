import matplotlib as mpl
mpl.use('Agg')

import os
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from eval.eval_calibration import eval_calibration

def main():
    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--eps', dest='epsilon', default=0.05,
                      help='Confidence threshold (1 - desired accuracy): default=%default')
    parser.add_option('--normalize', action="store_true", dest="normalize", default=False,
                      help='Normalize scores: default=%default')
    parser.add_option('--test', dest='test_prefix', default='test',
                      help='Name of test file: default=%default')
    parser.add_option('--xmax', dest='xmax', default=0.2,
                      help='x max: default=%default')
    parser.add_option('--ymax', dest='ymax', default=3000,
                      help='y max: default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()
    indir = args[0]
    test_prefix = options.test_prefix
    test_file = os.path.join(indir, test_prefix + '.npz')
    dev_file = os.path.join(indir, 'dev.npz')

    epsilon = float(options.epsilon)
    normalize = options.normalize
    seed = int(options.seed)
    np.random.seed(seed)
    xmax = float(options.xmax)
    ymax = float(options.ymax)

    eval_conformal_dwac(test_file, dev_file, epsilon, xmax, ymax, normalize)


def eval_conformal_dwac(test_file, dev_file, epsilon, xmax, ymax, normalize, plot=True):


    test_data = np.load(test_file)
    dev_data = np.load(dev_file)

    dev_labels = dev_data['labels']
    dev_pred_probs = dev_data['pred_probs']
    dev_confs = dev_data['confs']
    n_dev, _ = dev_pred_probs.shape
    print("Dev shape:", dev_pred_probs.shape)

    test_labels = test_data['labels']
    test_pred_probs = test_data['pred_probs']
    test_confs = test_data['confs']
    n_test, n_classes = test_pred_probs.shape
    print("Test shape:", test_pred_probs.shape)

    print("Accuracy = ", np.sum(test_labels == test_pred_probs.argmax(axis=1))/float(n_test))

    assert np.all(dev_confs.argmax(1) == dev_pred_probs.argmax(1))
    assert np.all(test_confs.argmax(1) == test_pred_probs.argmax(1))

    # scatter the labels
    test_labels = scatter(test_labels, n_classes)
    dev_labels = scatter(dev_labels, n_classes)

    # get the nonconformity score for each dev item, based on the true label
    if normalize:
        dev_scores = -dev_pred_probs[np.arange(n_dev), dev_labels.argmax(axis=1)]
        test_scores = -test_pred_probs
    else:
        dev_scores = -dev_confs[np.arange(n_dev), dev_labels.argmax(axis=1)]
        test_scores = -test_confs

    test_pred = test_pred_probs.argmax(axis=1)

    assert np.all(test_pred == test_pred_probs.argmax(axis=1))

    # compute conformal p-values for all test points/labels = proportion of dev points with greater nonconformity
    test_quantiles = np.sum(test_scores.reshape((n_test, n_classes, 1)) < dev_scores.reshape((1, 1, n_dev)), axis=2) / float(n_dev)

    credibility = test_quantiles.max(axis=1)
    temp = test_quantiles.copy()
    temp[np.arange(n_test), test_quantiles.argmax(axis=1)] = 0.0
    confidence = 1.0 - temp.max(axis=1)

    print("Mean credibility:", np.mean(credibility))
    print("Credibility histogram:", np.histogram(credibility, bins=np.linspace(0, 1.0, 11))[0])
    print("Confidences histogram:", np.histogram(confidence, bins=np.linspace(0, 1.0, 11))[0])

    test_preds = np.array(test_quantiles > epsilon, dtype=int)
    acc = np.sum(test_preds[np.arange(n_test), test_labels.argmax(axis=1)]) / float(n_test)
    print("Accuracy @ {:.2f} = {:.4f}".format(epsilon, acc))

    label_set_counts = np.histogram(test_preds.sum(axis=1), bins=np.arange(n_classes+2))[0]
    print("Empty predictions: {:d} {:.2f}%".format(label_set_counts[0], label_set_counts[0]/float(n_test)*100))
    print("Singly-labeled   : {:d} {:.2f}%".format(label_set_counts[1], label_set_counts[1]/float(n_test)*100))
    print("Multiply-labeled : {:d} {:.2f}%".format(label_set_counts[2:].sum(), label_set_counts[2:].sum()/float(n_test)*100))
    print("Histogram:", label_set_counts)

    print("By credibility")
    print("Bin Min_p   Acc   Cal   MAE MaxAE")
    n_bins = 10
    bin_size = n_test // n_bins
    order = np.argsort(credibility)
    for i in range(10):
        if i < 9:
            indices = order[i * bin_size:(i+1) * bin_size]
        else:
            indices = order[i * bin_size:]
        label_mean = test_labels[indices, :].mean(0)
        probs_mean = test_pred_probs[indices, :].mean(0)
        mae = np.mean(np.abs(label_mean - probs_mean))
        max_ae = np.max(np.abs(label_mean - probs_mean))
        calibration = eval_calibration(test_labels[indices, :], test_pred_probs[indices, :])
        acc = np.mean(np.array(test_labels[indices, :].argmax(1) == test_pred_probs[indices, :].argmax(1), dtype=float))
        print("{:d}   {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(i, test_pred_probs[indices, :].max(1).mean(), acc, calibration, mae, max_ae))

    prob_accs = []
    cred_accs = []
    conf_accs = []
    x = np.arange(0, 100)
    prob_thresholds = np.percentile(test_pred_probs.max(1), np.arange(0, 101))
    cred_thresholds = np.percentile(credibility, np.arange(0, 101))
    conf_thresholds = np.percentile(confidence, np.arange(0, 101))
    for t in x:
        prob_instances = test_pred_probs.max(1) >= prob_thresholds[t]
        acc = np.sum(np.array(test_labels[prob_instances, :].argmax(1) == test_pred_probs[prob_instances, :].argmax(1), dtype=float)) / prob_instances.sum()
        prob_accs.append(acc)

        cred_instances = credibility >= cred_thresholds[t]
        acc = np.sum(np.array(test_labels[cred_instances, :].argmax(1) == test_pred_probs[cred_instances, :].argmax(1), dtype=float)) / cred_instances.sum()
        cred_accs.append(acc)

        conf_instances = confidence >= conf_thresholds[t]
        acc = np.sum(np.array(test_labels[conf_instances, :].argmax(1) == test_pred_probs[conf_instances, :].argmax(1), dtype=float)) / conf_instances.sum()
        conf_accs.append(acc)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, prob_accs, label='prob')
        ax.plot(x, cred_accs, label='cred')
        ax.plot(x, conf_accs, label='conf')
        ax.set_ylim(0.9, 1.0)
        ax.legend()
        plt.savefig('conformal_dwac.pdf')

        fig, ax = plt.subplots(figsize=(6, 2))
        ax.hist(test_pred_probs.max(1), bins=np.linspace(0, 1, 101))
        ax.set_ylim(0, ymax)
        plt.savefig('hist_dwac_probs.pdf')

        fig, ax = plt.subplots(figsize=(6, 2))
        ax.hist(credibility, bins=np.linspace(0, 1, 101))
        ax.set_ylim(0, ymax)
        ax.set_ylabel('Test instances')
        ax.set_xlabel('Credibility')
        plt.savefig('hist_dwac_creds.pdf', bbox_inches='tight')

        fig, ax = plt.subplots()
        ax.hist(confidence, bins=np.linspace(0, 1, 101))
        plt.savefig('hist_dwac_confs.pdf')

    acc_vals = []
    empty_vals = []
    multi_vals = []
    multi_means = []
    x = np.linspace(0, xmax, 101)
    for epsilon in x:
        #epsilon = 1.0 - eps
        test_preds = np.array(test_quantiles > epsilon, dtype=int)
        acc = np.sum(test_preds[np.arange(n_test), test_labels.argmax(axis=1)]) / float(n_test)
        empty = np.sum(test_preds.sum(1) == 0) / float(n_test)
        multi = np.sum(test_preds.sum(1) > 1) / float(n_test)
        multi_mean = np.mean(test_preds[test_preds.sum(1) > 0, :].sum(1))
        multi_means.append(multi_mean)
        acc_vals.append(acc)
        empty_vals.append(empty)
        multi_vals.append(multi)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        #ax.plot([0, 0.2], [0, 0.2], 'k--', linewidth=1, label=None)
        ax.plot([0, xmax], [1, 1-xmax], 'k--', linewidth=1, label=None)
        ax.plot(x, acc_vals, label='Correct')
        ax.plot(x, empty_vals, label='Empty')
        ax.plot(x, multi_vals, label='Multi')
        ax.set_xlabel('epsilon')
        ax.set_ylabel('Proportion of test data')
        ax.legend()
        if normalize:
            plt.savefig('test_dwac_normalize.pdf', bbox_inches='tight')
        else:
            plt.savefig('test_dwac.pdf', bbox_inches='tight')

    return acc_vals, empty_vals, multi_vals, multi_means


def scatter(labels, n_classes):
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        n_items = len(labels)
        temp = np.zeros((n_items, n_classes), dtype=int)
        temp[np.arange(n_items), labels] = 1
        labels = temp
    return labels


if __name__ == '__main__':
    main()
