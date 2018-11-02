import matplotlib as mpl
mpl.use('Agg')

import os
from optparse import OptionParser

import numpy as np
from text.common import load_dataset


def main():
    usage = "%prog exp_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--root-dir', default='./data',
                      help='Root dir: default=%default')
    parser.add_option('--dataset', dest='dataset', default='imdb',
                      help='Dataset: default=%default')
    parser.add_option('--subset', dest='subset', default=None,
                      help='Subset (for amazon): default=%default')
    parser.add_option('-i', type=int, dest='index', default=0,
                      help='Index to print: default=%default')
    parser.add_option('--train', action="store_true", dest="train", default=False,
                      help='Load from training data (instead of test): default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()
    indir = args[0]
    test_file = os.path.join(indir, 'test.npz')
    train_file = os.path.join(indir, 'train.npz')

    train = options.train
    index = int(options.index)
    seed = int(options.seed)
    np.random.seed(seed)

    if train:
        data = np.load(train_file)
    else:
        data = np.load(test_file)
    indices = list(data['indices'])
    atts = data['atts']
    output_index = indices.index(index)
    print("Orig index = {:d}; output index = {:d}".format(index, output_index))

    train_dataset, test_dataset, ood_dataset = load_dataset(options.root_dir, options.dataset, options.subset)

    if train:
        dataset = train_dataset
    else:
        dataset = test_dataset
    document = dataset.all_docs[index]
    tokens = document[dataset.text_field_name]

    att = atts[output_index]

    print(len(att), len(tokens))
    print(document[dataset.label_field_name])
    scaled = np.array([np.sqrt(a) for a in att])
    scaled = scaled / scaled.sum()
    for i in range(len(tokens)):
        decile = int(np.round(scaled[i]*10))
        viz = '#' * decile + ' ' * (10 - decile)
        print("{:.3f} {:s} {:s}".format(att[i], viz, tokens[i]))


if __name__ == '__main__':
    main()
