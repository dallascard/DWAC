import matplotlib as mpl
mpl.use('Agg')

import os
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from eval.eval_conformal_baseline import eval_conformal_baseline
from eval.eval_conformal_dwac import eval_conformal_dwac


def main():
    usage = "%prog baseline_dir dwac_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--xmax', dest='xmax', default=0.2,
                      help='x max for plots: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    base_dir = args[0]
    dwac_dir = args[1]
    xmax = float(options.xmax)

    test_file = os.path.join(base_dir, 'test.npz')
    dev_file = os.path.join(base_dir, 'dev.npz')
    base_acc, base_empty, base_multi, base_multi_mean = eval_conformal_baseline(test_file, dev_file, 0.05, xmax, ymax=3000, plot=False)

    test_file = os.path.join(dwac_dir, 'test.npz')
    dev_file = os.path.join(dwac_dir, 'dev.npz')
    dwac_acc_norm, dwac_empty_norm, dwac_multi_norm, dwac_multi_mean_norm = eval_conformal_dwac(test_file, dev_file, 0.05, xmax, ymax=3000, normalize=True, plot=False)

    dwac_acc, dwac_empty, dwac_multi, dwac_multi_mean = eval_conformal_dwac(test_file, dev_file, 0.05, xmax, ymax=3000, normalize=False, plot=False)

    x = np.linspace(0, xmax, 101)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot([0, xmax], [1, 1-xmax], 'k--', linewidth=1, label=None)
    ax.plot(x, base_acc, label='Softmax (probs)', alpha=0.8)
    ax.plot(x, dwac_acc_norm, label='DWAC (probs)', alpha=0.8)
    ax.plot(x, dwac_acc, label='DWAC (weights)', alpha=0.8)
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Proportion of test instances')
    ax.legend()
    ax.set_title('(a) Correct predictions')
    ax.set_ylim(-0.05, 1.05)
    plt.savefig('fashion_acc.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot([0, xmax], [0, xmax], 'k--', linewidth=1, label=None)
    ax.plot(x, base_empty, label='Softmax (probs)', alpha=0.8)
    ax.plot(x, dwac_empty_norm, label='DWAC (probs)', alpha=0.8)
    ax.plot(x, dwac_empty, label='DWAC (weights)', alpha=0.8)
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Proportion of test instances')
    ax.legend()
    ax.set_title('(b) Empty predictions')
    ax.set_ylim(-0.05, 1.05)
    plt.savefig('fashion_empty.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot([0, xmax], [0, 0], 'k--', linewidth=1, label=None)
    ax.plot(x, base_multi, label='Softmax (probs)', alpha=0.8)
    ax.plot(x, dwac_multi_norm, label='DWAC (probs)', alpha=0.8)
    ax.plot(x, dwac_multi, label='DWAC (weights)', alpha=0.8)
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Proportion of test instances')
    ax.legend()
    ax.set_title('(c) Multiple predictions')
    ax.set_ylim(-0.05, 1.05)
    plt.savefig('fashion_multi.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot([0, xmax], [1, 1], 'k--', linewidth=1, label=None)
    ax.plot(x[:len(base_multi_mean)], base_multi_mean, label='Softmax (probs)', alpha=0.8)
    ax.plot(x[:len(dwac_multi_mean_norm)], dwac_multi_mean_norm, label='DWAC (probs)', alpha=0.8)
    ax.plot(x[:len(dwac_multi_mean)], dwac_multi_mean, label='DWAC (weights)', alpha=0.8)
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Number of predicted labels')
    ax.legend()
    ax.set_title('(d) Mean size of non-empty prediction sets')
    ax.set_ylim(-0.5, 10.5)
    plt.savefig('fashion_multi_size.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
