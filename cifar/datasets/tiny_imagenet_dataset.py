import os
import glob
from zipfile import ZipFile

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from utils import file_handling as fh


class TinyImageNet(Dataset):
    """`TinyImageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
    """
    base_folder = 'tiny-imagenet-200'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'

    raw_folder = 'raw'
    raw_filename = 'tiny-imagenet-200.zip'
    data_folder = 'tiny-imagenet-200'

    processed_folder = 'processed'
    train_file = 'train.npz'
    # test_file = 'test.npz'


    def __init__(self, root, train=True,
                 transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = True  # training set or test set

        if download:
            self.download()
        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.preprocess()

        processed_data = np.load(os.path.join(self.root, self.processed_folder, self.train_file))
        self.data = processed_data['data']
        self.targets = processed_data['targets']

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        # target = self.targets[index]
        target = 0  # done because the different number of classes from cifar
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.data)

    def download(self):
        if self._check_raw_exists():
            print("Download found")
            return

        # download files
        print("Downloading files")
        fh.makedirs(os.path.join(self.root, self.raw_folder))
        download_url(self.url, root=os.path.join(self.root, self.raw_folder),
                     filename=self.raw_filename, md5=None)

        print("Unzipping raw data")
        with ZipFile(os.path.join(self.root, self.raw_folder, self.raw_filename)) as raw_zip:
            raw_zip.extractall(os.path.join(self.root, self.raw_folder))

        if not self._check_raw_exists():
            raise RuntimeError("Unable to find downloaded file. Please try again.")
        else:
            print("Download finished")

    def preprocess(self):
        if self._check_processed_exists():
            print("Processed data found")
            return

        print("Processing data")
        data = []
        targets = []
        labels = []
        for label_dir in glob.glob(os.path.join(self.root, self.raw_folder, self.data_folder, 'train', '*')):
            label = os.path.basename(label_dir)
            labels.append(label)
            for image in glob.glob(os.path.join(self.root, self.raw_folder,
                                   self.data_folder, 'train', label, 'images', '*')):
                im = np.array(Image.open(image).convert('RGB'), dtype=np.uint8)

                data.append(im)
                targets.append(label)

        labels = {l:i for i, l in enumerate(labels)}
        targets = np.array([labels[l] for l in targets])
        data = np.asarray(data, dtype=np.uint8)

        fh.makedirs(os.path.join(self.root, self.processed_folder))
        np.savez(os.path.join(self.root, self.processed_folder, self.train_file), data=data, targets=targets)

        if not self._check_processed_exists():
            raise RuntimeError("Unable to find processed file. Please try again.")
        else:
            print("Processing finished")

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_folder, self.raw_filename)) and \
               os.path.exists(os.path.join(self.root, self.raw_folder, self.data_folder))

    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_file))
