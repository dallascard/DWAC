import re
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Abstract Dataset class for text datasets
    All classes inheriting from this should create the three objects in __init__:
    - all_docs: list of dicts, each with a text field and a label field, at least
    - vocab: Vocab object, made from a list of words
    - label_vocab: Vocab object made from a list of classes
    __len__() and __getitem__() should be useable without modification
    """

    def __init__(self, text_field_name='tokens', label_field_name='label'):

        self.all_docs = None
        self.vocab = None
        self.label_vocab = None
        self.text_field_name = text_field_name
        self.label_field_name = label_field_name

    def __len__(self):
        return len(self.all_docs)

    def __getitem__(self, idx):
        doc = self.all_docs[idx]
        x = torch.LongTensor([self.vocab.word2idx[w] if w in self.vocab.word2idx else self.vocab.unk_idx for w in doc[self.text_field_name]])
        y = torch.LongTensor([self.label_vocab.word2idx[doc[self.label_field_name]]])
        index = torch.LongTensor([idx])
        return x, y, index


class Vocab(object):
    def __init__(self, words, add_pad_unk=False):
        if add_pad_unk:
            words = ['<PAD>', '<UNK>'] + words
        self.word2idx = dict(zip(words, range(len(words))))
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        assert '<PAD>' not in self.word2idx or self.word2idx['<PAD>'] == 0
        assert '<UNK>' not in self.word2idx or self.word2idx['<UNK>'] == 1
        self.pad_idx = 0
        self.unk_idx = 1

    def __len__(self):
        return len(self.idx2word)


def tokenize(spacy_tokenizer, text, strip_html=True, lemmatize=False):
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)

    tokenized = spacy_tokenizer(text)
    output = ' '.join([token.text for token in tokenized])

    return output


def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    batch_inputs, batch_labels, batch_indices = zip(*batch)
    batch_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=0)
    batch_labels = torch.cat(batch_labels, dim=0)
    batch_indices = torch.cat(batch_indices, dim=0)
    return batch_inputs, batch_labels, batch_indices