import os
import errno
import tarfile
from collections import Counter

import numpy as np
from spacy.lang.en import English
from torchvision.datasets.utils import download_url

from utils import file_handling as fh
from text.datasets.text_dataset import TextDataset, Vocab, tokenize


class SubjectivityDataset(TextDataset):
    """`Subjectivity dataset from Pang and Lee <http://www.cs.cornell.edu/people/pabo/movie-review-data/>`.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, load the training data, otherwise test
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        lower (bool, optional): If true, lowercase text
    """
    url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz'

    raw_folder = 'raw'
    filename = 'rotten_imdb.tar.gz'
    processed_folder = 'processed'
    train_file = 'train.jsonlist'
    test_file = 'test.jsonlist'
    ood_file = 'ood.jsonlist'
    vocab_file = 'vocab.json'
    classes = ['subjective', 'objective']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, root, train=True, download=False, lower=True, ood=False, vocab=None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.ood = ood
        self.text_field_name = 'tokens'
        self.label_field_name = 'label'

        if download:
            self.download()

        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.preprocess()

        if train:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.train_file))
        elif ood:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.ood_file))
        else:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.test_file))

        # Do lower-casing on demand, to avoid redoing slow tokenization
        if lower:
            for doc in self.all_docs:
                doc['tokens'] = [token.lower() for token in doc['tokens']]

        # load and build a vocabulary, also lower-casing if necessary
        if vocab is None:
            vocab_file = os.path.join(self.root, self.processed_folder, self.vocab_file)
            print("Reading vocab from {:s}".format(vocab_file))
            vocab = fh.read_json(vocab_file)
            if lower:
                vocab = list(set([token.lower() for token in vocab]))
            self.vocab = Vocab(vocab, add_pad_unk=True)
        else:
            print("Using vocab as given")
            self.vocab = vocab

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
        vocab_counter = Counter()

        doc_lengths = []

        count = 0
        print("Processing documents")
        tar = tarfile.open(os.path.join(self.root, self.raw_folder, self.filename), "r:gz")
        # process all the data files in the archive
        for m_i, member in enumerate(tar.getmembers()):

            print(member, type(member))
            if member.name == 'plot.tok.gt9.5000':
                label = 'objective'
                f = tar.extractfile(member)
                bytes = f.read()
                lines = self.bytes_to_list(bytes)
                print(len(lines))

                for line_i, line in enumerate(lines[:-1]):
                    line = tokenize(tokenizer, line)
                    # save the text, label, and original file name
                    doc_lengths.append(len(line.split()))
                    doc = {'id': count, 'tokens': line.split(), 'label': label, 'orig': member.name + '.' + str(line_i)}
                    count += 1

                    if line_i % 10 == 0:
                        test_lines.append(doc)
                    else:
                        train_lines.append(doc)
                        vocab_counter.update(doc['tokens'])

            elif member.name == 'quote.tok.gt9.5000':
                label = 'subjective'
                f = tar.extractfile(member)
                bytes = f.read()
                lines = self.bytes_to_list(bytes)

                for line_i, line in enumerate(lines[:-1]):
                    line = tokenize(tokenizer, line)
                    # save the text, label, and original file name
                    doc_lengths.append(len(line.split()))
                    doc = {'id': count, 'tokens': line.split(), 'label': label, 'orig': member.name + '.' + str(line_i)}
                    count += 1

                    if line_i % 10 == 0:
                        test_lines.append(doc)
                    else:
                        train_lines.append(doc)
                        vocab_counter.update(doc['tokens'])

        print(len(train_lines))
        print(len(test_lines))
        vocab = list(vocab_counter.keys())
        vocab.sort()
        print("Vocab size = ", len(vocab))

        words, freqs = zip(*vocab_counter.items())
        words = list(words)
        freqs = np.array(freqs, dtype=float)
        freqs = freqs / freqs.sum()

        for i in range(1000):
            length = np.random.choice(doc_lengths, size=1)[0]
            tokens = list(np.random.choice(words, p=freqs, size=length, replace=True))
            ood_line = {'id': count, 'tokens': tokens, 'label': 'subjective', 'orig': 'random' + str(i)}
            ood_lines.append(ood_line)
            count += 1

        print("Saving processed data")
        fh.write_jsonlist(train_lines, os.path.join(self.root, self.processed_folder, self.train_file))
        fh.write_jsonlist(test_lines, os.path.join(self.root, self.processed_folder, self.test_file))
        fh.write_jsonlist(ood_lines, os.path.join(self.root, self.processed_folder, self.ood_file))
        fh.write_json(vocab, os.path.join(self.root, self.processed_folder, self.vocab_file), sort_keys=False)

    def bytes_to_list(self, bytes):
        string = bytes.decode('Windows-1252')
        return string.split('\n')
