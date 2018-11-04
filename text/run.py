import os
import gzip
import argparse

import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from text.datasets.text_dataset import collate_fn
from text.baseline_model import TextBaseline
from text.dwac_model import AttentionCnnDwac
from text.common import load_dataset
from utils.common import to_numpy


def main():
    parser = argparse.ArgumentParser(description='Text Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Fundamental options
    parser.add_argument('--model', type=str, default='baseline', metavar='N',
                        help='Model to use [baseline|dwac]')
    parser.add_argument('--dataset', type=str, default='imdb', metavar='N',
                        help='Dataset to run [imdb|amazon|stackoverflow|subjectivity]')
    parser.add_argument('--subset', type=str, default=None, metavar='N',
                        help='Subset for amazon or framing dataset [beauty|...]')

    # Text Options
    parser.add_argument('--lower', action='store_true', default=False,
                        help='Convert text to lower case')

    # Model Options
    parser.add_argument('--glove-file', type=str, default='data/vectors/glove.6B.300d.txt.gz', metavar='N',
                        help='Glove vectors')
    parser.add_argument('--embedding-dim', type=int, default=300, metavar='N',
                        help='word vector dimensions')
    parser.add_argument('--hidden-dim', type=int, default=100, metavar='N',
                        help='Size of penultimate layer')
    parser.add_argument('--fix-embeddings', action='store_true', default=False,
                        help='fix word embeddings in training')
    parser.add_argument('--kernel-size', type=int, default=5, metavar='N',
                        help='convolution filter kernel size')

    # File Options
    parser.add_argument('--root-dir', type=str, default='./data', metavar='PATH',
                        help='path to data directory')
    parser.add_argument('--output-dir', type=str, default='./data/temp', metavar='PATH',
                        help='Output directory')

    # Training Options
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='test batch size')
    parser.add_argument('--dev-prop', type=float, default=0.1, metavar='N',
                        help='Proportion of training data to use as a dev set')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training')
    parser.add_argument('--max-epochs', type=int, default=50, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='number of training epochs')

    # DWAC Architecture Options
    parser.add_argument('--z-dim', type=int, default=32, metavar='N',
                        help='dimensions of latent representation')
    parser.add_argument('--kernel', type=str, default='gaussian', metavar='k',
                        help='type of distance function [gaussian|laplace|invquad')
    parser.add_argument('--gamma', type=float, default=1, metavar='k',
                        help='hyperparameter for kernel')
    parser.add_argument('--eps', type=float, default=1e-12, metavar='k',
                        help='label smoothing factor for learning')
    parser.add_argument('--topk', type=int, default=10, metavar='N',
                        help='top k nearest neighbors to compare to at test time')

    # Running Options
    parser.add_argument('--device', type=int, default=None,
                        help='GPU to use (if any)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='random seed')

    args = parser.parse_args()

    if args.device is None:
        args.device = 'cpu'
    else:
        args.device = 'cuda:' + str(args.device)
    print("Using device:", args.device)
    args.update_embeddings = not args.fix_embeddings

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device != 'cpu':
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(args.seed)

    # load data and create vocab and label vocab objects
    vocab, label_vocab, train_loader, dev_loader, test_loader, ref_loader, ood_loader = load_data(args)
    args.n_classes = len(label_vocab)

    # load an initialize the embeddings
    embeddings_matrix = load_embeddings(args, vocab)

    # create the model
    if args.model == 'baseline':
        print("Creating baseline model")
        model = TextBaseline(args, vocab, embeddings_matrix)
    elif args.model == 'dwac':
        print("Creating DWAC model")
        model = AttentionCnnDwac(args, vocab, embeddings_matrix)
    else:
        raise ValueError("Model type not recognized.")
    print("Update embeddings = ", args.update_embeddings)

    train(args, model, train_loader, dev_loader, test_loader, ref_loader, ood_loader)


def load_data(args):

    ood_loader = None
    train_dataset, test_dataset, ood_dataset = load_dataset(args.root_dir, args.dataset, args.subset, args.lower)

    print(len(train_dataset))
    print(len(test_dataset))
    n_train = len(train_dataset)
    indices = list(range(n_train))
    split = int(np.floor(args.dev_prop * n_train))
    np.random.shuffle(indices)
    train_idx, dev_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    dev_sampler = SubsetRandomSampler(dev_idx)
    ref_sampler = SubsetRandomSampler(train_idx)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device != 'cpu' else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **kwargs)
    ref_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        sampler=ref_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        **kwargs)
    if ood_dataset is not None:
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=args.test_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **kwargs)

    vocab = train_dataset.vocab
    label_vocab = train_dataset.label_vocab

    return vocab, label_vocab, train_loader, dev_loader, test_loader, ref_loader, ood_loader


def load_embeddings(args, vocab):
    embeddings = {}

    print("Reading embeddings")
    # get the set of words which exist in glove
    if args.glove_file is not None:
        if args.glove_file[-3:] == '.gz':
            with gzip.open(args.glove_file) as embeddings_file:
                for line in embeddings_file:
                    fields = line.decode('utf-8').strip().split(" ")
                    word = fields[0]
                    if word in vocab.word2idx:
                        vector = np.asarray(fields[1:], dtype="float32")
                        embeddings[word] = vector
        else:
            with open(args.glove_file) as embeddings_file:
                for line in embeddings_file:
                    fields = line.strip().split(" ")
                    word = fields[0]
                    if word in vocab.word2idx:
                        vector = np.asarray(fields[1:], dtype="float32")
                        embeddings[word] = vector

    # create an embedding matrix for the full vocabulary and randomly initialize
    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    embedding_matrix = torch.FloatTensor(len(vocab), args.embedding_dim)
    embedding_matrix.normal_(embeddings_mean, embeddings_std)

    # overwrite embeddings for known words with pre-trained values from glove
    embedding_matrix[vocab.pad_idx].fill_(0)
    for word in embeddings:
        idx = vocab.word2idx[word]
        embedding_matrix[idx] = torch.FloatTensor(embeddings[word])
    return embedding_matrix


def train(args, model, train_loader, dev_loader, test_loader, ref_loader, ood_loader=None):
    best_dev_acc = 0.0
    done = False
    epoch = 0
    epochs_without_improvement = 0
    best_epoch = 0

    print("Creating output directory {:s}".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    while not done:
        for batch_idx, (data, target, indices) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = model.fit(data, target)
            loss = output['loss'].item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                           100. * batch_idx / len(train_loader), loss))

        if args.model == 'dwac':
            dev_output = test_fast(args, model, dev_loader, ref_loader, name='Dev', return_acc=True)
            dev_acc = dev_output['accuracy']
        else:
            dev_acc, dev_indices = test(args, model, dev_loader, ref_loader, name='Dev', return_acc=True)

        if dev_acc > best_dev_acc:
            print("New best dev accuracy: {:.5f}\n".format(dev_acc))
            best_dev_acc = dev_acc
            epochs_without_improvement = 0
            best_epoch = epoch
            model_file = os.path.join(args.output_dir, 'model.best.tar')
            model.save(model_file)
        else:
            epochs_without_improvement += 1
        print("Epochs without improvement = {:d}\n".format(epochs_without_improvement))
        epoch += 1

        if epochs_without_improvement > args.patience:
            print("Patience exceeded; exiting training\n")
            done = True
        elif epoch >= args.max_epochs:
            print("Max epochs exceeded; exiting training\n")
            done = True

    model_file = os.path.join(args.output_dir, 'model.best.tar')
    print("Reloading best model")
    model.load(model_file)

    print("Embedding training data")
    train_indices, train_z, train_labels, atts = embed(args, model, ref_loader)
    print("Saving")
    np.savez(os.path.join(args.output_dir, 'train.npz'),
             labels=train_labels,
             z=train_z,
             indices=train_indices,
             atts=atts)

    print("Doing dev eval")
    if args.model == 'dwac':
        save_output(os.path.join(args.output_dir, 'dev.npz'),
                    test_fast(args, model, dev_loader, ref_loader, name='Dev'))
    else:
        dev_labels, dev_indices, dev_pred_probs, dev_z, dev_confs, dev_atts = test(args, model, dev_loader, ref_loader, name='Dev')
        print("Saving")
        np.savez(os.path.join(args.output_dir, 'dev.npz'),
                 labels=dev_labels,
                 z=dev_z,
                 pred_probs=dev_pred_probs,
                 indices=dev_indices,
                 confs=dev_confs)

    print("Doing test eval")
    if args.model == 'dwac':
        save_output(os.path.join(args.output_dir, 'test.npz'),
                    test_fast(args, model, test_loader, ref_loader, name='Test'))
    else:
        test_labels, test_indices, test_pred_probs, test_z, test_confs, test_atts = test(args, model, test_loader, ref_loader, name='Test')
        print("Saving")
        np.savez(os.path.join(args.output_dir, 'test.npz'),
                 labels=test_labels,
                 z=test_z,
                 pred_probs=test_pred_probs,
                 indices=test_indices,
                 confs=test_confs,
                 atts=test_atts)

    if ood_loader is not None:
        if args.model == 'dwac':
            save_output(os.path.join(args.output_dir, 'ood.npz'),
                        test_fast(args, model, ood_loader, ref_loader, name='OOD'))
        else:
            print("Doing OOD eval")
            ood_labels, ood_indices, ood_pred_probs, ood_z, ood_confs, ood_atts = test(args, model, ood_loader, ref_loader, name='OOD')
            print("Saving")
            np.savez(os.path.join(args.output_dir, 'ood.npz'),
                     labels=ood_labels,
                     z=ood_z,
                     pred_probs=ood_pred_probs,
                     indices=ood_indices,
                     confs=ood_confs)


def test(args, model, test_loader, ref_loader, name='Test', return_acc=False):
    test_loss = 0
    correct = 0
    true_labels = []
    all_indices = []
    pred_probs = []
    confs = []
    zs = []
    atts = []
    n_items = 0

    for batch_idx, (data, target, indices) in enumerate(test_loader):
        data, target = data.to(args.device), target.to(args.device)
        output = model.evaluate(data, target, ref_loader)
        test_loss += output['total_loss'].item()
        batch_size = len(target)
        n_items += batch_size
        pred = output['probs'].max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        all_indices.extend(list(to_numpy(indices, args.device)))
        if not return_acc:
            true_labels.extend(list(to_numpy(target, args.device)))
            pred_probs.append(to_numpy(output['probs'].exp(), args.device))
            #all_indices.extend(list(to_numpy(indices, args.device)))
            zs.append(to_numpy(output['z'], args.device))
            atts.append(to_numpy(output['att'], args.device))
            if args.model == 'dwac':
                confs.append(to_numpy(output['confs'], args))

    test_loss /= len(test_loader.sampler)
    print('{:s} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        name, test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))
    print()

    acc = correct / len(test_loader.sampler)
    if args.model == 'dwac' and not return_acc:
        confs = np.vstack(confs)

    if return_acc:
        return acc, all_indices
    else:
        max_att_len = np.max([m.shape[1] for m in atts])
        att_matrix = np.zeros([n_items, max_att_len])
        index = 0
        for m in atts:
            batch_size, width = m.shape
            att_matrix[index:index+batch_size, :width] = m.copy()
            index += batch_size

        return true_labels, all_indices, np.vstack(pred_probs), np.vstack(zs), confs, att_matrix


def test_fast(args, model, test_loader, ref_loader, name='Test', return_acc=False):
    with torch.no_grad():
        output = model.evaluate(test_loader, ref_loader)
    test_loss = output['loss']
    correct = output['correct']

    print()
    print('{:s} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        name, test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))
    print()

    return output


def embed(args, model, loader):
    zs = []
    labels = []
    all_indices = []
    atts = []
    with torch.no_grad():
        for batch_idx, (data, target, indices) in enumerate(loader):
            data = data.to(args.device)
            output = model.embed(data)
            zs.append(to_numpy(output['z'], args.device))
            labels.extend(list(to_numpy(target, args.device)))
            all_indices.extend(list(to_numpy(indices, args.device)))
            att_matrix = to_numpy(output['att'], args.device)
            atts.extend([att_matrix[i, :] for i in range(len(att_matrix))])
    return all_indices, np.vstack(zs), labels, atts


def predict(args, model, test_loader, ref_loader):
    all_indices = []
    pred_probs = []
    confs = []
    with torch.no_grad():
        for batch_idx, (data, target, indices) in enumerate(test_loader):
            data = data.to(args.device)
            output = model.predict(data, ref_loader)
            pred_probs.append(to_numpy(output['probs'].exp(), args.device))
            all_indices.extend(list(to_numpy(indices, args.device)))
            if args.model == 'dwac':
                confs.append(to_numpy(output['confs'], args))

    return all_indices, np.vstack(pred_probs), np.vstack(confs)


def save_output(path, output):
    att_matrices = [m.cpu().data.numpy() for m in output['att']]
    att_vectors = []
    lengths = []
    for m in att_matrices:
        att_vectors.extend([m[i, :] for i in range(len(m))])
        lengths.extend([len(m[i, :]) for i in range(len(m))])
    np.savez(path,
             z          = output['zs'].cpu().data.numpy(),
             labels     = output['ys'].cpu().data.numpy(),
             indices    = output['is'].cpu().data.numpy(),
             pred_probs = output['probs'].exp().cpu().data.numpy(),
             confs      = output['confs'].cpu().data.numpy(),
             atts       = att_vectors)


if __name__ == '__main__':
    main()
