import os
import random
import argparse

import numpy as np

import torch

from mnist.models.dwac import MNISTDwac
from mnist.models.baseline import MNISTBaseline
from mnist.common import load_mnist_data, to_numpy


def main():
    parser = argparse.ArgumentParser(description='MNIST Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # File Options
    parser.add_argument('--root-dir', type=str, default='./data', metavar='PATH',
                        help='path to data directory')
    parser.add_argument('--output-dir', type=str, default='./data/temp', metavar='PATH',
                        help='path ot output data to')

    # Training Options
    parser.add_argument('--model', type=str, default='baseline',
                        help='Model type [baseline|dwac]')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='test batch size')
    parser.add_argument('--dev-prop', type=float, default=0.1, metavar='N',
                        help='Proportion of training data to use as a dev set')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='learning rate for training')
    parser.add_argument('--max-epochs', type=int, default=100, metavar='N',
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='number of training epochs')

    # DWAC Architecture Options
    parser.add_argument('--z-dim', type=int, default=10, metavar='N',
                        help='dimensions of latent representation')
    parser.add_argument('--kernel', type=str, default='gaussian', metavar='k',
                        help='hyperparameter for kernel [guassian|laplace|invquad]')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='k',
                        help='hyperparameter for kernel')
    parser.add_argument('--eps', type=float, default=1e-12, metavar='k',
                        help='label smoothing factor for learning')

    # Running Options
    parser.add_argument('--device', type=int, default=None,
                        help='GPU device to use')
    parser.add_argument('--fashion', action='store_true', default=False,
                        help='Use fashion MNIST')
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

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

    train_loader, dev_loader, test_loader, ref_loader, ood_loader = load_mnist_data(args)
    args.n_classes = 10

    if args.model == 'dwac':
        print("Creating DWAC model")
        model = MNISTDwac(args)
    else:
        print("Creating baseline model")
        model = MNISTBaseline(args)

    train(args, model, train_loader, dev_loader, ref_loader)

    model_file = os.path.join(args.output_dir, 'model.best.tar')
    model.load(model_file)

    train_z, train_labels, train_indices = embed(args, model, ref_loader)
    np.savez(os.path.join(output_dir, 'train.npz'), z=train_z, labels=train_labels, indices=train_indices)

    save_output(os.path.join(output_dir, 'dev.npz'), test(args, model, dev_loader, ref_loader, name='Dev'))
    save_output(os.path.join(output_dir, 'test.npz'), test(args, model, test_loader, ref_loader, name='Test'))
    save_output(os.path.join(output_dir, 'ood.npz'), test(args, model, ood_loader, ref_loader, name='OOD'))


def save_output(path, output):
    np.savez(path,
             z          = output['zs'].cpu().data.numpy(),
             labels     = output['ys'].cpu().data.numpy(),
             indices    = output['is'].cpu().data.numpy(),
             pred_probs = output['probs'].cpu().data.numpy() if 'probs' in output else None,
             confs      = output['confs'].cpu().data.numpy() if 'confs' in output else None)


def train(args, model, train_loader, dev_loader, ref_loader):
    best_dev_acc = 0.0
    done = False
    epoch = 0
    epochs_without_improvement = 0
    best_epoch = 0

    while not done:
        for batch_idx, (data, target, indices) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
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
            print("Saving model to ", model_file)
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


def test(args, model, test_loader, ref_loader, name='Test', return_acc=False):
    with torch.no_grad():
        output = model.evaluate(test_loader, ref_loader)

    print('\n{:s} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        name, output['loss'], output['correct'], len(test_loader.sampler), output['accuracy']))
    return output


def embed(args, model, loader):
    zs = []
    labels = []
    all_indices = []
    with torch.no_grad():
        for batch_idx, (data, target, indices) in enumerate(loader):
            data = data.to(args.device)
            output = model.get_representation(data)
            zs.append(to_numpy(output['z'], args.device))
            labels.extend(list(to_numpy(target, args.device)))
            all_indices.extend(indices)
    return np.vstack(zs), labels, all_indices


if __name__ == '__main__':
    main()
