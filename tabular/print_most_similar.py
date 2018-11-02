import os
from optparse import OptionParser

from eval.find_closest import find_closest

from tabular.datasets.lending_dataset import LendingData
from tabular.datasets.income_dataset import AdultIncomeData
from tabular.datasets.covertype_dataset import CovertypeData


def main():
    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--root-dir', default='./data',
                      help='Root dir: default=%default')
    parser.add_option('--dataset', default='income',
                      help='Dataset: default=%default')
    parser.add_option('-i', dest='index', default=0,
                      help='Index of test point to lookup: default=%default')
    parser.add_option('-n', dest='n', default=5,
                      help='Number of columns: default=%default')
    parser.add_option('--distf', dest='dist_function', default='Gauss',
                      help='Distance function [Gauss|Laplace|InverseQuad]: default=%default')
    parser.add_option('--gamma', dest='gamma', default=1.0,
                      help='Parameter of distance function: default=%default')
    #parser.add_option('-n', action="store_true", dest="adaptive", default=False,
    #                  help='Use adaptive binning: default=%default')

    (options, args) = parser.parse_args()
    indir = args[0]

    root_dir = options.root_dir
    dataset = options.dataset
    dist_function = options.dist_function
    gamma = float(options.gamma)
    index = int(options.index)
    n = int(options.n)

    test_index, closest_train, dists = find_closest(indir, dist_function, gamma, index)

    index = options.index


    ood_dataset = None
    ood_loader = None
    if dataset == 'lending':
        train_dataset = LendingData(os.path.join(root_dir, 'lending'), subset='train')
        test_dataset = LendingData(os.path.join(root_dir, 'lending'), subset='test')
    elif dataset == 'income':
        train_dataset = AdultIncomeData(os.path.join(root_dir, 'income'), subset='train')
        test_dataset = AdultIncomeData(os.path.join(root_dir, 'income'), subset='test')
    elif dataset == 'covertype':
        train_dataset = CovertypeData(os.path.join(root_dir, 'covertype'), subset='train')
        test_dataset = CovertypeData(os.path.join(root_dir, 'covertype'), subset='test')
    else:
        raise ValueError("Dataset not recognized.")

    test_instance, test_label, test_index = test_dataset.get_raw(test_index)
    columns = train_dataset.columns

    train_instances = []
    train_labels = []
    for i in range(n):
        train_instance, train_label, train_index = train_dataset.get_raw(closest_train[i])
        train_instances.append(train_instance)
        train_labels.append(train_label)

    print("Index\tTest\t", '\t'.join([str(closest_train[i]) for i in range(n)]))
    print("Label\t" + str(train_dataset.classes[test_label]) + "\t", '\t'.join([str(train_dataset.classes[train_labels[i]]) for i in range(n)]))
    print("Weight\tTest\t", '\t'.join([str(dists[i]) for i in range(n)]))
    for c_i, c in enumerate(columns):
        print("{:s}\t".format(c) + str(test_instance[c_i]) + '\t' + '\t'.join([str(train_instances[i][c_i]) for i in range(n)]))



if __name__ == '__main__':
    main()
