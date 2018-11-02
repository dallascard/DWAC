import os
import argparse

import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from tabular.baseline_model import TabularBaseline
from tabular.dwac_model import TabularDWAC

from tabular.datasets.lending_dataset import LendingData
from tabular.datasets.income_dataset import AdultIncomeData
from tabular.datasets.covertype_dataset import CovertypeData

from utils.common import to_numpy


def main():
    parser = argparse.ArgumentParser(description='Tabular data classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, default='baseline', metavar='N',
                        help='Model to use [baseline|dwac]')
    parser.add_argument('--dataset', type=str, default='lending', metavar='N',
                        help='Dataset to run [lending|income|covertype]')
    parser.add_argument('--ood', default=None, metavar='N',
                        help='class to use as out-of-domain (for covertype dataset)')

    # model options
    parser.add_argument('--dh1', type=int, default=32, metavar='N',
                        help='Size of first hidden layer')
    parser.add_argument('--dh2', type=int, default=8, metavar='N',
                        help='Size of second hidden layer')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='Dropout prob')

    # File Options
    parser.add_argument('--root-dir', type=str, default='./data', metavar='PATH',
                        help='path to data directory')
    parser.add_argument('--output-dir', type=str, default='./data/temp', metavar='PATH',
                        help='Output directory')

    # Training Options
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='N',
                        help='test batch size')
    parser.add_argument('--dev-prop', type=float, default=0.1, metavar='N',
                        help='Proportion of training data to use as a dev set')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training')
    parser.add_argument('--l2', type=float, default=0.0, metavar='N',
                        help='L2 regularization')
    parser.add_argument('--max-epochs', type=int, default=10, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=2, metavar='N',
                        help='number of training epochs')

    # DWAC Architecture Options
    parser.add_argument('--z-dim', type=int, default=2, metavar='N',
                        help='dimensions of latent representation')
    parser.add_argument('--kernel', type=str, default='gaussian', metavar='k',
                        help='type of distance function [gaussian|laplace|invquad')
    parser.add_argument('--gamma', type=float, default=1, metavar='k',
                        help='hyperparameter for kernel')
    parser.add_argument('--eps', type=float, default=1e-12, metavar='k',
                        help='label smoothing factor for learning')

    # Running Options
    parser.add_argument('--device', type=int, default=None,
                        help='Which GPU to use (if any)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='random seed')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(int(args.seed))

    if not os.path.exists(args.output_dir):
        print("Creating output directory {:s}".format(args.output_dir))
        os.makedirs(args.output_dir)

    if args.device is None:
        args.device = 'cpu'
    else:
        args.device = 'cuda:' + str(args.device)
    print("Using device:", args.device)

    train_loader, dev_loader, test_loader, ref_loader, ood_loader, n_classes, xdim = load_data(args)
    args.n_classes = n_classes
    args.xdim = xdim

    if args.model == 'baseline':
        print("Creating baseline model")
        model = TabularBaseline(args)
    elif args.model == 'dwac':
        print("Creating DWAC model")
        model = TabularDWAC(args)
    else:
        raise ValueError("Model type not recognized.")

    model = train(args, model, train_loader, dev_loader, ref_loader)

    print("Embedding training data")
    train_indices, train_z, train_labels = embed(args, model, ref_loader)
    print("Saving")
    np.savez(os.path.join(args.output_dir, 'train.npz'),
             labels=train_labels,
             z=train_z,
             indices=train_indices)

    print("Doing dev eval")
    save_output(os.path.join(args.output_dir, 'dev.npz'),
                test(args, model, dev_loader, ref_loader, name='Dev'))

    print("Doing test eval")
    save_output(os.path.join(args.output_dir, 'test.npz'),
                test(args, model, test_loader, ref_loader, name='Test'))

    if ood_loader is not None:
        print("Doing OOD eval")
        save_output(os.path.join(args.output_dir, 'ood.npz'),
                    test(args, model, ood_loader, ref_loader, name='OOD'))


def load_data(args):

    ood_dataset = None
    ood_loader = None
    if args.dataset == 'lending':
        train_dataset = LendingData(os.path.join(args.root_dir, 'lending'), subset='train')
        test_dataset = LendingData(os.path.join(args.root_dir, 'lending'), subset='test')
    elif args.dataset == 'income':
        train_dataset = AdultIncomeData(os.path.join(args.root_dir, 'income'), subset='train')
        test_dataset = AdultIncomeData(os.path.join(args.root_dir, 'income'), subset='test')
    elif args.dataset == 'covertype':
        if args.ood is None:
            train_dataset = CovertypeData(os.path.join(args.root_dir, 'covertype'), subset='train')
            test_dataset = CovertypeData(os.path.join(args.root_dir, 'covertype'), subset='test')
        else:
            train_dataset = CovertypeData(os.path.join(args.root_dir, 'covertype'), subset='train', ood_class=int(args.ood))
            test_dataset = CovertypeData(os.path.join(args.root_dir, 'covertype'), subset='test', ood_class=int(args.ood))
            ood_dataset = CovertypeData(os.path.join(args.root_dir, 'covertype'), subset='ood', ood_class=int(args.ood))
    else:
        raise ValueError("Dataset not recognized.")

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
        **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        shuffle=False,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)
    ref_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        sampler=ref_sampler,
        shuffle=False,
        **kwargs)
    if ood_dataset is not None:
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs)

    n_classes = len(train_dataset.classes)
    _, xdim = train_dataset.shape()

    return train_loader, dev_loader, test_loader, ref_loader, ood_loader, n_classes, xdim


def train(args, model, train_loader, dev_loader, ref_loader):
    best_dev_acc = 0.0
    done = False
    epoch = 0
    epochs_without_improvement = 0
    best_epoch = 0

    while not done:
        for batch_idx, (data, target, indices) in enumerate(train_loader):
            data, target = data.float().to(args.device), target.to(args.device)
            output = model.fit(data, target)
            loss = output['loss'].item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                           100. * batch_idx / len(train_loader), loss))

        dev_output = test(args, model, dev_loader, ref_loader, name='Dev', return_acc=True)
        dev_acc = dev_output['accuracy']

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
    return model


def test(args, model, test_loader, ref_loader, name='Test', return_acc=False):
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
    with torch.no_grad():
        for batch_idx, (data, target, indices) in enumerate(loader):
            data = data.float().to(args.device)
            output = model.embed(data)
            zs.append(to_numpy(output['z'], args.device))
            labels.extend(list(to_numpy(target, args.device)))
            all_indices.extend(list(to_numpy(indices, args.device)))
    return all_indices, np.vstack(zs), labels


def save_output(path, output):
    np.savez(path,
             z          = output['zs'].cpu().data.numpy(),
             labels     = output['ys'].cpu().data.numpy(),
             indices    = output['is'].cpu().data.numpy(),
             pred_probs = output['probs'].exp().cpu().data.numpy(),
             confs      = output['confs'].cpu().data.numpy())




if __name__ == '__main__':
    main()
