import os
import errno
import zipfile
from collections import Counter

from spacy.lang.en import English
from torchvision.datasets.utils import download_url

from utils import file_handling as fh
from text.datasets.text_dataset import TextDataset, Vocab, tokenize


class StackOverflowDataset(TextDataset):
    """`Dataset of Stack Overflow titles from 20 categories <https://github.com/jacoxu/StackOverflow>`.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, load the training data, otherwise test
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'https://github.com/jacoxu/StackOverflow/archive/master.zip'

    raw_folder = 'raw'
    filename = 'master.zip'
    processed_folder = 'processed'
    train_file = 'train.jsonlist'
    test_file = 'test.jsonlist'
    ood_file = 'ood.jsonlist'
    vocab_file = 'vocab.json'
    classes = ['wordpress',
               'oracle',
               'svn',
               'apache',
               'excel',
               'matlab',
               'visual-studio',
               'cocoa',
               'osx',
               'bash',
               'spring',
               'hibernate',
               'scala',
               'sharepoint',
               'ajax',
               'qt',
               'drupal',
               'linq',
               'haskell',
               'magento']

    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, root, partition='train', download=False, lower=True, ood_class=None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.partition = partition
        if ood_class is not None:
            self.ood_classes = [ood_class]
        else:
            self.ood_classes = []
        self.text_field_name = 'tokens'
        self.label_field_name = 'label'

        if download:
            self.download()

        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.preprocess()

        if partition == 'train':
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.train_file))
        elif partition == 'ood':
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.ood_file))
        elif partition == 'test':
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.test_file))
        else:
            raise RuntimeError("Partition {:s} not recognized".format(partition))

        # Do lower-casing on demand, to avoid redoing slow tokenization
        if lower:
            for doc in self.all_docs:
                doc['tokens'] = [token.lower() for token in doc['tokens']]

        # load and build a vocabulary, also lower-casing if necessary
        vocab = fh.read_json(os.path.join(self.root, self.processed_folder, self.vocab_file))
        if lower:
            vocab = list(set([token.lower() for token in vocab]))
        self.vocab = Vocab(vocab, add_pad_unk=True)

        self.label_vocab = Vocab(self.classes)

    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.vocab_file))

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_folder, self.filename))

    def download(self):
        """Download the IMDB data if it doesn't exist in processed_folder already."""

        if self._check_raw_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        download_url(self.url,
                     root=os.path.join(self.root, self.raw_folder),
                     filename=self.filename,
                     md5=None)
        if not self._check_raw_exists():
            raise RuntimeError("Unable to find downloaded file. Please try again.")
        else:
            print("Download finished.")

    def preprocess(self):
        """Preprocess the raw data file"""
        if self._check_processed_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print("Preprocessing raw data")
        print("Loading spacy")
        # load a spacy parser
        tokenizer = English()

        train_lines = []
        test_lines = []
        ood_lines = []
        unsup_lines = []
        vocab = set()


        ratings = set()
        train_labels = Counter()
        test_labels = Counter()

        print("Processing documents")
        # read in the raw data
        zf = zipfile.ZipFile(os.path.join(self.root, self.raw_folder, self.filename), 'r')
        titles = zf.read('StackOverflow-master/rawText/title_StackOverflow.txt')
        titles = self.bytes_to_list(titles)[:-1]

        labels = zf.read('StackOverflow-master/rawText/label_StackOverflow.txt')
        labels = self.bytes_to_list(labels)[:-1]

        for line_i, line in enumerate(titles):

            if line_i % 1000 == 0:
                print("Processing line {:d} / 20000".format(line_i))

            text = tokenize(tokenizer, line)
            label = self.classes[int(labels[line_i]) - 1]

            # save the text, label, and original file name
            doc_out = {'id': line_i, 'tokens': text.split(), 'label': label}

            # take every tenth review as the training set
            if line_i % 10 == 0:
                if label in self.ood_classes:
                    ood_lines.append(doc_out)
                else:
                    test_lines.append(doc_out)
                    test_labels.update([label])
            else:
                if label in self.ood_classes:
                    ood_lines.append(doc_out)
                    vocab.update(doc_out['tokens'])
                else:
                    train_lines.append(doc_out)
                    vocab.update(doc_out['tokens'])
                    train_labels.update([label])

        print("Train counts:", train_labels.most_common())
        print("Test counts:", test_labels.most_common())
        vocab = list(vocab)
        vocab.sort()
        print("Vocab size = {:d}".format(len(vocab)))

        print("Saving processed data")
        fh.write_jsonlist(train_lines, os.path.join(self.root, self.processed_folder, self.train_file))
        fh.write_jsonlist(test_lines, os.path.join(self.root, self.processed_folder, self.test_file))
        fh.write_jsonlist(ood_lines, os.path.join(self.root, self.processed_folder, self.ood_file))
        fh.write_json(vocab, os.path.join(self.root, self.processed_folder, self.vocab_file), sort_keys=False)

    def bytes_to_list(self, bytes):
        string = bytes.decode('utf-8')
        return string.split('\n')
