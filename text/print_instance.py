import os
from optparse import OptionParser

from text.datasets.imdb_dataset import IMDB
from text.datasets.amazon_dataset import AmazonReviews
from text.datasets.subjectivity_dataset import SubjectivityDataset
from text.datasets.stackoverflow_dataset import StackOverflowDataset


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--dataset', dest='dataset', default='imdb',
                      help='Dataset [imdb|amazon|stackoverflow|subjectivity|framing]: default=%default')
    parser.add_option('--subset', dest='subset', default=None,
                      help='Subset (for amazon): default=%default')
    parser.add_option('--root-dir', default='./data',
                      help='Root dir: default=%default')
    parser.add_option('--train', action="store_true", dest="train", default=False,
                      help='Index into the training set (instead of test): default=%default')
    parser.add_option('-i', type=int, dest='index', default=0,
                      help='Index to print: default=%default')

    (options, args) = parser.parse_args()
    dataset = options.dataset
    subset = options.subset
    train = options.train
    index = options.index

    if dataset == 'imdb':
        if train:
            data = IMDB(os.path.join(options.root_dir, 'imdb'), train=True, download=True, strip_html=True, lower=False)
        else:
            data = IMDB(os.path.join(options.root_dir, 'imdb'), train=False, download=True, strip_html=True, lower=False)
    elif dataset == 'amazon':
        if subset is None:
            raise ValueError("Please provide a subset for the Amazon dataset.")
        if train:
            data = AmazonReviews(os.path.join(options.root_dir, 'amazon'), subset=subset, train=True, download=True, lower=False)
        else:
            data = AmazonReviews(os.path.join(options.root_dir, 'amazon'), subset=subset, train=False, download=True, lower=False)
    elif dataset == 'stackoverflow':
        if train:
            data = StackOverflowDataset(os.path.join(options.root_dir, 'stackoverflow'), train=True, download=True, lower=False)
        else:
            data = StackOverflowDataset(os.path.join(options.root_dir, 'stackoverflow'), train=False, download=True, lower=False)
    elif dataset == 'subjectivity':
        if train:
            data = SubjectivityDataset(os.path.join(options.root_dir, 'subjectivity'), train=True, download=True, lower=False)
        else:
            data = SubjectivityDataset(os.path.join(options.root_dir, 'subjectivity'), train=False, download=True, lower=False)
    else:
        raise ValueError("Dataset not recognized.")

    document = data.all_docs[index]
    print(' '.join(document[data.text_field_name]), 'Label:{:s}'.format(document[data.label_field_name]))


if __name__ == '__main__':
    main()
