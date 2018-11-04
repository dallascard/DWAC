import os
import gzip
import json
import errno
from collections import Counter

from spacy.lang.en import English
from torchvision.datasets.utils import download_url

from utils import file_handling as fh
from text.datasets.text_dataset import TextDataset, Vocab, tokenize


class AmazonReviews(TextDataset):
    """`Amazon reviews <http://jmcauley.ucsd.edu/data/amazon/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        subset (str, optional): Subset of amazon review data to download and process
        train (bool, optional): If True, load the training data, otherwise test
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        lower (bool, optional): If true, lowercase text
    """
    url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'

    subsets = {'books': 'reviews_Books_5.json.gz',
               'electronics': 'reviews_Electronics_5.json.gz',
               'movies': 'reviews_Movies_and_TV_5.json.gz',
               'cds': 'reviews_CDs_and_Vinyl_5.json.gz',
               'clothing': 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
               'home': 'reviews_Home_and_Kitchen_5.json.gz',
               'kindle': 'reviews_Kindle_Store_5.json.gz',
               'sports': 'reviews_Sports_and_Outdoors_5.json.gz',
               'phones': 'reviews_Cell_Phones_and_Accessories_5.json.gz',
               'health': 'eviews_Health_and_Personal_Care_5.json.gz',
               'toys': 'reviews_Toys_and_Games_5.json.gz',
               'videogames': 'reviews_Video_Games_5.json.gz',
               'tools': 'reviews_Tools_and_Home_Improvement_5.json.gz',
               'beauty': 'reviews_Beauty_5.json.gz',
               'apps': 'reviews_Apps_for_Android_5.json.gz',
               'office': 'reviews_Office_Products_5.json.gz',
               'pets': 'reviews_Pet_Supplies_5.json.gz',
               'cars': 'reviews_Automotive_5.json.gz',
               'grocery': 'reviews_Grocery_and_Gourmet_Food_5.json.gz',
               'garden': 'reviews_Patio_Lawn_and_Garden_5.json.gz',
               'baby': 'reviews_Baby_5.json.gz',
               'music': 'reviews_Digital_Music_5.json.gz',
               'musical': 'reviews_Musical_Instruments_5.json.gz',
               'video': 'reviews_Amazon_Instant_Video_5.json.gz'}

    raw_folder = 'raw'
    processed_folder = 'processed'
    train_file = 'train.jsonlist'
    test_file = 'test.jsonlist'
    vocab_file = 'vocab.json'
    classes = ['1', '2', '3', '4', '5']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, root, subset='beauty', train=True, download=False, lower=True):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.filename = self.subsets[subset]
        self.train = train
        self.text_field_name = 'tokens'
        self.label_field_name = 'rating'

        if download:
            self.download()

        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.preprocess()

        if train:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.subset, self.train_file))
        else:
            self.all_docs = fh.read_jsonlist(os.path.join(self.root, self.processed_folder, self.subset, self.test_file))

        # Do lower-casing on demand, to avoid redoing slow tokenization
        if lower:
            for doc in self.all_docs:
                doc['tokens'] = [token.lower() for token in doc['tokens']]

        # load and build a vocabulary, also lower-casing if necessary
        vocab = fh.read_json(os.path.join(self.root, self.processed_folder, self.subset, self.vocab_file))
        if lower:
            vocab = list(set([token.lower() for token in vocab]))
        self.vocab = Vocab(vocab, add_pad_unk=True)

        self.label_vocab = Vocab(self.classes)

    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.subset, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.subset, self.test_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.subset, self.vocab_file))

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

        download_url(os.path.join(self.url, self.filename),
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
            os.makedirs(os.path.join(self.root, self.processed_folder, self.subset))
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
        vocab = set()

        ratings = set()
        train_labels = Counter()
        test_labels = Counter()

        print("Processing documents")
        # read in the raw data
        with gzip.open(os.path.join(self.root, self.raw_folder, self.filename), 'r') as f:
            for line_i, line in enumerate(f):
                doc = json.loads(line, encoding='utf-8')
                # Display occassional progress
                if (line_i + 1) % 1000 == 0:
                    print("Processed {:d}".format(line_i+1))

                asin = doc['asin']
                rating = float(doc['overall'])
                ratings.add(rating)
                text = doc['summary'] + '\n\n' + '<SEP>' + '\n\n' + doc['reviewText']

                text = tokenize(tokenizer, text)

                # save the text, label, and original file name
                doc_out = {'id': line_i, 'asin': asin, 'tokens': text.split(), 'rating': str(int(rating))}

                # take every tenth review as the training set
                if line_i % 10 == 0:
                    test_lines.append(doc_out)
                    test_labels.update(str(int(rating)))
                else:
                    train_lines.append(doc_out)
                    vocab.update(doc_out['tokens'])
                    train_labels.update(str(int(rating)))

        print("Ratings:", ratings)
        print("Train:", len(train_lines))
        print("Test:", len(test_lines))
        print(train_labels.most_common())
        print(test_labels.most_common())
        vocab = list(vocab)
        vocab.sort()
        print("Vocab size = {:d}".format(len(vocab)))

        print("Saving processed data")
        fh.write_jsonlist(train_lines, os.path.join(self.root, self.processed_folder, self.subset, self.train_file))
        fh.write_jsonlist(test_lines, os.path.join(self.root, self.processed_folder, self.subset, self.test_file))
        fh.write_json(vocab, os.path.join(self.root, self.processed_folder, self.subset, self.vocab_file), sort_keys=False)
