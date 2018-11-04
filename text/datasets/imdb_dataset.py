import os
import errno
import tarfile

from spacy.lang.en import English
from torchvision.datasets.utils import download_url

from utils import file_handling as fh
from text.datasets.text_dataset import TextDataset, Vocab, tokenize


class IMDB(TextDataset):
    """`IMDB <http://ai.stanford.edu/~amaas/data/sentiment/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, load the training data, otherwise test
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        strip_html (bool, optional): If True, remove html tags during preprocessing; default=True
        lower (bool, optional): If true, lowercase text
    """
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    raw_folder = 'raw'
    raw_filename = 'aclImdb_v1.tar.gz'
    processed_folder = 'processed'
    train_file = 'train.jsonlist'
    test_file = 'test.jsonlist'
    vocab_file = 'vocab.json'
    classes = ['neg', 'pos']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, root, train=True, download=False, strip_html=True, lower=True):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.strip_html = strip_html

        if download:
            self.download()

        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.preprocess()

        if train:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.train_file))
        else:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.test_file))

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
        return os.path.exists(os.path.join(self.root, self.raw_folder, self.raw_filename))

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

        download_url(self.url, root=os.path.join(self.root, self.raw_folder),
                     filename=self.raw_filename, md5=None)
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
        unsup_lines = []
        vocab = set()

        print("Processing documents")
        # read in the raw data
        tar = tarfile.open(os.path.join(self.root, self.raw_folder, self.raw_filename), "r:gz")
        # process all the data files in the archive
        for m_i, member in enumerate(tar.getmembers()):
            # Display occassional progress
            if (m_i + 1) % 500 == 0:
                print("Processed {:d} / 50000".format(m_i+1))
            # get the internal file name
            parts = member.name.split(os.sep)

            if len(parts) > 3:
                split = parts[1]  # train or test
                label = parts[2]  # pos, neg, or unsup
                name = parts[3].split('.')[0]
                doc_id, rating = name.split('_')
                doc_id = int(doc_id)
                rating = int(rating)

                # read the text from the archive
                f = tar.extractfile(member)
                bytes = f.read()
                text = bytes.decode("utf-8")
                # tokenize it using spacy
                if label != 'unsup':
                    text = tokenize(tokenizer, text, strip_html=self.strip_html)
                # save the text, label, and original file name
                doc = {'id': doc_id, 'tokens': text.split(), 'label': label, 'orig': member.name, 'rating': rating}
                if label != 'unsup':
                    if split == 'train':
                        train_lines.append(doc)
                        vocab.update(doc['tokens'])
                    elif split == 'test':
                        test_lines.append(doc)
                    else:
                        raise ValueError("Unexpected split:", split)
                else:
                    doc['label'] = None
                    unsup_lines.append(doc)

        vocab = list(vocab)
        vocab.sort()

        print("Saving processed data")
        fh.write_jsonlist(train_lines, os.path.join(self.root, self.processed_folder, self.train_file))
        fh.write_jsonlist(test_lines, os.path.join(self.root, self.processed_folder, self.test_file))
        fh.write_json(vocab, os.path.join(self.root, self.processed_folder, self.vocab_file), sort_keys=False)
