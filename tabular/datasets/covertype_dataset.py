import os
import errno

from torchvision.datasets.utils import download_url

import numpy as np
import pandas as pd

from tabular.datasets.tabular_dataset import TabularDataset


class CovertypeData(TabularDataset):
    """
    UCI Covertype dataset: https://archive.ics.uci.edu/ml/datasets/Covertype

    Args:
    - root (string): Root directory of dataset where ``raw`` and  ``processed`` will exist.
    - subset (string): Subset to load; possible values are ``train``, ``test``, and ``ood``.
    - download (bool, optional): If true, download the dataset and place it in the raw dir.
    - ood_class (int, optional): Which 1-based label to use as an out-of-domain dataset
    (to be excluded from train and test); if None, don't make a separate ood split.
    - test_prop (float, optional): the proportion of the dataset to use as a test set
    - seed (int, optional): random seed to use for splitting into train and test
    - force (bool, optional): if True, redo the preprocessing
    """

    urls = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    ]

    raw_file = 'covtype.data.gz'
    ood_file = 'ood.npz'
    columns = ['Elevation',
               'Aspect',
               'Slope',
               'Horizontal_Distance_To_Hydrology',
               'Vertical_Distance_To_Hydrology',
               'Horizontal_Distance_To_Roadways',
               'Hillshade_9am',
               'Hillshade_Noon',
               'Hillshade_3pm',
               'Horizontal_Distance_To_Fire_Points'] + \
              ['Wilderness_Area' + str(i) for i in range(4)] + \
              ['Soil_type' + str(i) for i in range(40)]

    classes = ['1 - Spruce/Fir',
               '2 - Lodgepole-Pine',
               '3 - Ponderosa-Pine',
               '4 - Cottonwood/Willow',
               '5 - Aspen',
               '6 - Douglas-fir',
               '7 - Krummholz']

    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, root, subset='train', download=True, ood_class=None, test_prop=0.1, seed=42, force=False):
        super().__init__(root, subset)
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.ood_class = ood_class
        self.test_prop = test_prop
        self.seed = seed
        self.force = force
        self.processed_folder = 'processed'
        if self.ood_class is not None:
            self.processed_folder = self.processed_folder + str(self.ood_class)

        if download and not self._check_raw_exists():
            self.download()

        if force or not self._check_exists():
            if self.subset == 'train':
                self.preprocess_covertype_data()

        if not self._check_exists():
            raise RuntimeError('Processed data not found.')

        if self.subset == 'ood':
            data = np.load(os.path.join(self.root, self.processed_folder, self.ood_file))
            self.data = data['X']
            self.labels = data['Y']
        elif self.subset == 'train':
            data = np.load(os.path.join(self.root, self.processed_folder, self.training_file))
            self.data = data['X']
            self.labels = data['Y']
        elif self.subset == 'test':
            data = np.load(os.path.join(self.root, self.processed_folder, self.test_file))
            self.data = data['X']
            self.labels = data['Y']
        else:
            raise RuntimeError("Subset must be one of [train|test|ood].")

        transform = np.load(os.path.join(self.root, self.processed_folder, self.transform_file))
        self.mean = transform['mean']
        self.std = transform['std']

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_folder, self.raw_file))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.transform_file))

    def download(self):
        # download files
        try:
            print("Making", os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, root=os.path.join(self.root, self.raw_folder),
                         filename=filename, md5=None)

    def preprocess_covertype_data(self):

        try:
            print("Making", os.path.join(self.root, self.processed_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        np.random.seed(self.seed)
        infile = os.path.join(self.root, self.raw_folder, self.raw_file)

        df = pd.read_csv(infile, header=None, index_col=None)

        np.random.seed(self.seed)

        data = np.array(df.values)
        n_rows, n_cols = data.shape

        X = data[:, :-1]
        Y = data[:, -1]

        # consider label 7 to be out-of-domain, and put it into a separate partition
        if self.ood_class is not None:
            ood_indices = [i for i in np.arange(n_rows) if Y[i] == self.ood_class]
            if len(ood_indices) == 0:
                raise RuntimeError("No instances found for ood class")
            ood_X = X[ood_indices, :]
            ood_Y = np.array(Y[ood_indices] - 1, dtype=int)
            if self.ood_class is not None:
                np.savez(os.path.join(self.root, self.processed_folder, self.ood_file), X=ood_X, Y=ood_Y)
        else:
            ood_indices = []

        ind_indices = set(np.arange(n_rows)) - set(ood_indices)
        test_indices = list(np.random.choice(list(ind_indices), size=int(len(ind_indices) * self.test_prop), replace=False))
        train_indices = list(ind_indices - set(test_indices))

        train_X = X[train_indices, :]
        train_Y = np.array(Y[train_indices] - 1, dtype=int)
        test_X = X[test_indices, :]
        test_Y = np.array(Y[test_indices] - 1, dtype=int)

        print("Features shape:", train_X.shape)
        print("Labels shape:", train_Y.shape)

        # compute means and standard deviations for non-binary columns
        means = train_X.mean(axis=0)
        means[10:] = 0
        std = train_X.std(axis=0)
        std[10:] = 1.0

        np.savez(os.path.join(self.root, self.processed_folder, self.training_file), X=train_X, Y=train_Y)
        np.savez(os.path.join(self.root, self.processed_folder, self.test_file), X=test_X, Y=test_Y)
        np.savez(os.path.join(self.root, self.processed_folder, self.transform_file), mean=means, std=std)

