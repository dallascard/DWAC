from optparse import OptionParser

import numpy as np


# Evaluate the overall calibration of the output of a model

def main():
    usage = "%prog output.npz [output2.npz ...]"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='n_bins', default=10,
                      help='Number of bins: default=%default')
    parser.add_option('--exp', action="store_true", dest="exp", default=False,
                      help='Exponentiate the log-probs: default=%default')
    parser.add_option('-v', action="store_true", dest="verbose", default=False,
                      help='Print details: default=%default')

    (options, args) = parser.parse_args()
    infiles = args

    n_bins = int(options.n_bins)
    verbose = options.verbose
    adaptive = True
    exp = options.exp

    mae_vals = []
    acc_vals = []
    for infile in infiles:

        acc, mae = eval_calibration_file(infile, n_bins=n_bins, adaptive=adaptive, verbose=verbose, exp=exp)
        print(infile, "ACC = {:.5f}".format(acc))
        print(infile, "MAE = {:.5f}".format(mae))

        mae_vals.append(mae)
        acc_vals.append(acc)

    print("Mean ACC = {:.5f} ({:.5f})".format(np.mean(acc_vals), np.std(acc_vals)))
    print("Mean MAE = {:.5f} ({:.5f})".format(np.mean(mae_vals), np.std(mae_vals)))


def eval_calibration_file(infile, n_bins=10, adaptive=True, verbose=False, exp=False):

    data = np.load(infile)

    labels = data['labels']
    pred_probs = data['pred_probs']
    if exp:
        pred_probs = np.exp(pred_probs)
    n_items, n_classes = pred_probs.shape
    # scatter the labels
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        temp = np.zeros((n_items, n_classes), dtype=int)
        temp[np.arange(n_items), labels] = 1
        labels = temp

    mae = eval_calibration(labels, pred_probs, n_bins, adaptive, verbose)
    acc = np.sum(labels.argmax(axis=1) == pred_probs.argmax(axis=1)) / float(n_items)

    return acc, mae


def eval_calibration(label_matrix, pred_probs, n_bins=10, adaptive=True, verbose=False):

    n_items, n_classes = label_matrix.shape
    if n_classes > 2:
        mae = 0.0
        for c in range(n_classes):
            if verbose:
                print("Class {:d}".format(c))
            mae += eval_calibration_by_class(label_matrix, pred_probs, col=c, n_bins=n_bins, adaptive=adaptive, verbose=verbose) / n_bins
    else:
        mae = eval_calibration_by_class(label_matrix, pred_probs, col=0, n_bins=n_bins, adaptive=adaptive, verbose=verbose)

    return mae


def eval_calibration_by_class(labels, pred_probs, col=0, n_bins=10, adaptive=True, verbose=False):
    n_items, n_classes = pred_probs.shape
    order = np.argsort(pred_probs[:, col])
    bin_size = n_items // n_bins
    counts = []
    lower_vals = []
    label_means = []
    probs_means = []
    ae_vals = []
    mae = 0.0
    for i in range(n_bins):
        if adaptive:
            if i < n_bins-1:
                indices = order[i * bin_size:(i+1) * bin_size]
            else:
                indices = order[i * bin_size:]
            lower = np.min(pred_probs[indices, col])
            counts.append(len(indices))
        else:
            lower = 1.0 / n_bins * i
            upper = 1.0 / n_bins * (i+1)
            if i < n_bins - 1:
                indices = (pred_probs[:, col] >= lower) * (pred_probs[:, col] < upper)
            else:
                indices = (pred_probs[:, col] >= lower)
            counts.append(indices.sum())

        mean_probs = pred_probs[indices, col].mean()
        mean_label = labels[indices, col].mean()

        ae = np.abs(mean_probs - mean_label)
        mae += ae
        lower_vals.append(lower)
        label_means.append(mean_label)
        probs_means.append(mean_probs)
        ae_vals.append(ae)

    if verbose:
        print('Bins:\t' + '\t'.join(['{:.3f}'.format(low) for low in lower_vals]))
        print('Count:\t' + '\t'.join(['{:d}'.format(val) for val in counts]))
        print('True:\t' + '\t'.join(['{:.3f}'.format(val) for val in label_means]))
        print('Pred:\t' + '\t'.join(['{:.3f}'.format(val) for val in probs_means]))
        print('AE:\t' + '\t'.join(['{:.3f}'.format(val) for val in ae_vals]))

    return mae / n_bins


if __name__ == '__main__':
    main()
