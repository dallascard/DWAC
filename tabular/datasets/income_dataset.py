import os
import json
import errno
from collections import Counter

import numpy as np
import pandas as pd

from torchvision.datasets.utils import download_url

from tabular.datasets.tabular_dataset import TabularDataset


class AdultIncomeData(TabularDataset):
    """
    UCI Adult Income dataset: https://archive.ics.uci.edu/ml/datasets/adult

    Args:
    - root (string, optional): Root directory of dataset where ``raw`` and ``processed`` will exist
    - subset (string, optional): Name of subset. Should be ``train`` or ``test``.
    """

    urls = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
    ]

    processed_folder = 'processed'
    raw_train_file = 'adult.data'
    raw_test_file = 'adult.test'
    classes = ['<=50K', '>50K']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, root, subset='train'):
        super().__init__(root, subset)
        self.root = os.path.expanduser(root)

        if not self._check_raw_exists():
            self.download()

        if not self._check_exists():
            self.preprocess_income_data()

        if self.subset == 'train':
            data = np.load(os.path.join(self.root, self.processed_folder, self.training_file))
            self.data = data['X']
            self.labels = data['Y']
        elif self.subset == 'test':
            data = np.load(os.path.join(self.root, self.processed_folder, self.test_file))
            self.data = data['X']
            self.labels = data['Y']
        else:
            raise RuntimeError("subset must be one of ``train`` or ``test``")

        transform = np.load(os.path.join(self.root, self.processed_folder, self.transform_file))
        self.mean = transform['mean']
        self.std = transform['std']

        with open(os.path.join(self.root, self.processed_folder, self.columns_file), 'r') as f:
            self.columns = json.load(f)

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_folder, self.raw_train_file)) and \
               os.path.exists(os.path.join(self.root, self.raw_folder, self.raw_test_file))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.columns_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.transform_file))

    def download(self):
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

    def preprocess_income_data(self):

        if not os.path.exists(os.path.join(self.root, self.processed_folder)):
            os.makedirs(os.path.join(self.root, self.processed_folder))

        train_file = os.path.join(self.root, self.raw_folder, 'adult.data')
        test_file = os.path.join(self.root, self.raw_folder, 'adult.test')

        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        continuous_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        label_columns = ['income']

        if self.subset == 'train':
            df = pd.read_csv(train_file, header=-1)
            output_file = os.path.join(self.root, self.processed_folder, self.training_file)
        elif self.subset == 'test':
            df = pd.read_csv(test_file, header=-1, skiprows=1)
            output_file = os.path.join(self.root, self.processed_folder, self.test_file)
        else:
            raise RuntimeError("subset must be ``train`` or ``test``.")

        df.columns = columns

        # Drop fnlwgt column
        df.drop(['fnlwgt'], axis=1, inplace=True)
        columns.remove('fnlwgt')
        continuous_columns.remove('fnlwgt')

        print("Removing missing values")
        not_missing = np.min(df!= ' ?', axis=1)
        df = df.loc[not_missing, :]

        n_items, _ = df.shape
        print('Processing', self.subset)

        features = []
        labels = []

        transforms = {}

        # Load the training data to determine the proper indexing and normalization
        if self.subset == 'train':
            for c_i, c in enumerate(columns):
                if c in continuous_columns:
                    features.append(c)
                    values = df[c].values
                    values = np.array([float(v) for v in values])
                    transforms[c] = (values.mean(), values.std())

                elif c in categorical_columns:
                    counter = Counter()
                    counter.update(df[c].values)
                    keys = [k.strip() for k in counter.keys()]
                    keys = [c + '_' + k for k in keys]
                    features.extend(keys)
                    for k in keys:
                        transforms[k] = (0.0, 1.0)

                else:
                    counter = Counter()
                    counter.update(df[c].values)
                    keys = [k.strip() for k in counter.keys()]
                    labels.extend(keys)

            mean = [transforms[f][0] for f in features]
            std  = [transforms[f][1] for f in features]

            print("Saving transform and column information")
            np.savez(os.path.join(self.root, self.processed_folder, self.transform_file), mean=mean, std=std)
            with open(os.path.join(self.root, self.processed_folder, self.columns_file), 'w') as f:
                json.dump(features, f, sort_keys=False)

        else:
            with open(os.path.join(self.root, self.processed_folder, self.columns_file), 'r') as f:
                features = json.load(f)

        n_features = len(features)
        feature_index = dict(zip(features, range(n_features)))

        # deal with period at the end of test data labels
        label_index = {k: v for k, v in self.class_to_idx.items()}
        label_index['<=50K.'] = 0
        label_index['>50K.'] = 1

        print("Converting data")
        X = np.zeros([n_items, n_features])
        y = np.zeros(n_items, dtype=int)
        for c in columns:
            if c in continuous_columns:
                col_index = feature_index[c]
                values = df[c]
                X[:, col_index] = values

            elif c in categorical_columns:
                values = df[c]
                col_indices = [feature_index[c + '_' + v.strip()] for v in values]
                X[np.arange(n_items), col_indices] = 1

            elif c in label_columns:

                y = np.array([label_index[v.strip()] for v in df[c]], dtype=int)

        print("Features shape:", X.shape)
        print("Labels shape:", y.shape)

        print("Saving processed data")
        np.savez(output_file, X=X, Y=y)
