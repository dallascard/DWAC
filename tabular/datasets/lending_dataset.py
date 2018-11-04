import os
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

from tabular.datasets.tabular_dataset import TabularDataset


class LendingData(TabularDataset):
    """
    Lending Club dataset: https://www.kaggle.com/wendykan/lending-club-loan-data

    Note that this dataset must be downloaded manually from Kaggle and placed in data/lending/raw/

    Processing adapted from https://www.kaggle.com/wsogata/good-or-bad-loan-draft/notebook

    Args:
    - root (string, optional): Root directory of dataset where ``raw`` and ``processed`` will exist
    - subset (string, optional): Name of subset. Should be ``train`` or ``test``.
    - test_prop (float, optional): the proportion of the dataset to use as a test set
    - seed (int, optional): random seed to use for splitting into train and test

    Based on the torchvision.MNIST Dataset
    """

    url = "https://www.kaggle.com/wendykan/lending-club-loan-data"
    processed_folder = 'processed'
    raw_filename = 'loan.csv'
    classes = [0, 1]
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, root, subset=True, test_prop=0.1, seed=42):
        super().__init__(root, subset)
        self.test_prop = test_prop
        self.seed = seed

        if not self._check_exists():
            self.preprocess_data()

        if self.subset == 'train':
            data = np.load(os.path.join(self.root, self.processed_folder, self.training_file))
            self.data = data['X']
            self.labels = data['Y']
        elif self.subset == 'test':
            data = np.load(os.path.join(self.root, self.processed_folder, self.test_file))
            self.data = data['X']
            self.labels = data['Y']
        else:
            raise RuntimeError("Subset must be ``train`` or ``test``.")

        transform = np.load(os.path.join(self.root, self.processed_folder, self.transform_file))
        self.mean = transform['mean']
        self.std = transform['std']

        with open(os.path.join(self.root, self.processed_folder, self.columns_file), 'r') as f:
            self.columns = json.load(f)

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_folder, self.raw_filename))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.columns_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.transform_file))

    def preprocess_data(self):

        data_file = os.path.join(self.root, self.raw_folder, self.raw_filename)

        if not self._check_raw_exists():
            raise FileNotFoundError("Please download the data from {:s} and place it in directory {:s}".
                                    format(self.url, os.path.join(self.root, self.raw_folder)))

        if not os.path.exists(os.path.join(self.root, self.processed_folder)):
            os.makedirs(os.path.join(self.root, self.processed_folder))

        print("Reading csv file")
        df = pd.read_csv(data_file, low_memory=False)
        print(df.shape)

        print("Dropping features")
        # Drop these features for now
        df.drop(['id',
                 'member_id',
                 'emp_title',
                 'title',
                 'url',
                 'zip_code',
                 'verification_status',
                 'home_ownership',
                 'issue_d',
                 'earliest_cr_line',
                 'last_pymnt_d',
                 'next_pymnt_d',
                 'desc',
                 'last_credit_pull_d'],
                axis=1,
                inplace=True)

        # dropping columns with too many nans
        lack_of_data_idx = [x for x in df.count() < 887379*0.25]
        df.drop(df.columns[lack_of_data_idx], 1, inplace=True)

        counts = Counter()
        counts.update(df['loan_status'].values)

        # drop fully paid
        df = df[df.loan_status != 'Fully Paid']
        df = df[df.loan_status != 'Does not meet the credit policy. Status:Fully Paid']
        df = df[df.loan_status != 'Does not meet the credit policy. Status:Charged Off']
        print("Dropping rows")

        counts = Counter()
        counts.update(df['loan_status'].values)

        df.mths_since_last_delinq = df.mths_since_last_delinq.fillna(df.mths_since_last_delinq.median())
        df.dropna(inplace=True)

        # create an bad/good loan indicator feature
        df['good_loan'] = np.where((df.loan_status == 'Fully Paid') |
                                   (df.loan_status == 'Current') |
                                   (df.loan_status == 'Does not meet the credit policy. Status:Fully Paid'), 1, 0)

        # Hot encode some categorical features
        columns = ['term', 'grade', 'sub_grade', 'emp_length', 'purpose', 'application_type','addr_state',
                   'pymnt_plan', 'initial_list_status']

        for col in columns:
            tmp_df = pd.get_dummies(df[col], prefix=col)
            df = pd.concat((df, tmp_df), axis=1)

        df.drop(['loan_status',
                 'term',
                 'grade',
                 'sub_grade',
                 'emp_length',
                 'addr_state',
                 'initial_list_status',
                 'pymnt_plan',
                 'purpose',
                 'application_type'], axis=1, inplace=True)

        # Rename some features to concur w/ some algorithms
        df = df.rename(columns= {'emp_length_< 1 year':'emp_length_lt_1 year',
                                 'emp_length_n/a':'emp_length_na'})

        # split into data and labels
        y = df['good_loan']
        X = df.ix[:, df.columns != 'good_loan']

        columns = list(X.columns)

        # split into train and test
        print("Splitting data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_prop, random_state=self.seed)
        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)

        # convert to the right data types
        X_train = np.array(X_train.values, dtype=float)
        y_train = np.array(y_train.values, dtype=int)
        X_test = np.array(X_test.values, dtype=float)
        y_test = np.array(y_test.values, dtype=int)

        # compute the mean and standard deviation per column from the training data
        mean = X_train.mean(0)
        std = X_train.std(0)
        std[std == 0] = 1.

        # save the train, test, mean, standard deviation, and column names
        print("Saving data")
        np.savez(os.path.join(self.root, self.processed_folder, self.training_file), X=X_train, Y=y_train)
        np.savez(os.path.join(self.root, self.processed_folder, self.test_file), X=X_test, Y=y_test)
        np.savez(os.path.join(self.root, self.processed_folder, self.transform_file), mean=mean, std=std)
        with open(os.path.join(self.root, self.processed_folder, self.columns_file), 'w') as f:
            json.dump(columns, f, sort_keys=False)

