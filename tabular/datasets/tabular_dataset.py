import os

from torch.utils import data


class TabularDataset(data.Dataset):
    """
    Abstract tabular dataset (i.e. input features occur in named columns).

    Assumes that subclasses will compute mean and standard deviation vectors, and that these will be
    used to normalize data in __getitem__()
    """

    raw_folder = 'raw'
    training_file = 'train.npz'
    test_file = 'test.npz'
    transform_file = 'transform.npz'
    columns_file = 'columns.json'

    def __init__(self, root, subset='train'):
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.columns = None
        self.mean = None
        self.std = None
        self.data = None
        self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row, target = self.data[index], self.labels[index]
        row = (row - self.mean) / self.std
        return row, target, index

    def get_raw(self, index):
        row, target = self.data[index], self.labels[index]
        return row, target, index

    def shape(self):
        return self.data.shape

