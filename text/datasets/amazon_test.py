import os
from optparse import OptionParser

import torch

from text.datasets.amazon_dataset import AmazonReviews
from text.datasets.text_dataset import collate_fn


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--root-dir', default='./data',
                      help='Root dir: default=%default')
    parser.add_option('--subset', dest='subset', default=None,
                      help='Subset (for amazon): default=%default')

    (options, args) = parser.parse_args()

    train_dataset = AmazonReviews('./data/amazon', subset=options.subset, train=True, download=True)

    cuda = False
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn,
        **kwargs)



if __name__ == '__main__':
    main()
