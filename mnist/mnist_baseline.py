import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mnist.common import load_mnist_data, to_numpy


class mnist_baseline(nn.Module):
    def __init__(self):
        super(mnist_baseline, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv2_dropout(x)
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        x_probs = F.log_softmax(x, dim=1)
        return (x_probs)


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, test_loader, name='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    true_labels = []
    pred_probs = []
    all_indices = []

    with torch.no_grad():
        for data, target, indices in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            true_labels.extend(list(to_numpy(target, args)))
            pred_probs.append(to_numpy(output.exp(), args))
            all_indices.extend(indices)

    test_loss /= len(test_loader.sampler)
    print('\n{:s} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        name, test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))

    acc = correct / len(test_loader.sampler)
    return true_labels, np.vstack(pred_probs), all_indices, acc


def main():
    parser = argparse.ArgumentParser(description='MNIST Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root-dir', type=str, default='./data', metavar='PATH',
                        help='path to data directory')
    parser.add_argument('--output-dir', type=str, default='./data/mnist/baseline', metavar='PATH',
                        help='path ot output data to')

    # Training Options
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='test batch size')
    parser.add_argument('--dev-prop', type=float, default=0.1, metavar='N',
                        help='Proportion of training data to use as a dev set')
    parser.add_argument('--lr', type=float, default=10e-4, metavar='N',
                        help='learning rate for training')
    parser.add_argument('--max-epochs', type=int, default=50, metavar='N',
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='number of training epochs')

    # Running Options
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda if availible')
    parser.add_argument('--fashion', action='store_true', default=False,
                        help='Use fashion MNIST')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='random seed')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        raise FileNotFoundError("Output directory does not exist")

    train_loader, dev_loader, test_loader, _, ood_loader = load_mnist_data(args)

    model = mnist_baseline().to(args.device)
    optimizer = optim.Adam(model.parameters(), args.lr)

    best_dev_acc = 0.0
    done = False
    epoch = 0
    epochs_without_improvement = 0
    best_epoch = 0

    while not done:
        train(args, model, train_loader, optimizer, epoch)
        dev_labels, dev_pred_probs, dev_indices, dev_acc = test(args, model, dev_loader, name='Dev')
        if dev_acc > best_dev_acc:
            print("New best dev accuracy: {:.5f}\n".format(dev_acc))
            best_dev_acc = dev_acc
            epochs_without_improvement = 0
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
        print("Epochs without improvement = {:d}".format(epochs_without_improvement))
        epoch += 1

        if epochs_without_improvement > args.patience:
            print("Patience exceeded; exiting training\n")
            done = True
        elif epoch > args.max_epochs:
            print("Max epochs exceeded; exiting training\n")
            done = True

        # TODO: restore best model

    print("Best accuracy = {:.3f} on epoch {:3}\n".format(best_dev_acc, best_epoch))

    dev_labels, dev_pred_probs, dev_indices, dec_acc = test(args, model, dev_loader, name='Dev')
    np.savez(os.path.join(output_dir, 'dev.npz'), labels=dev_labels, pred_probs=dev_pred_probs, indices=dev_indices)

    test_labels, test_pred_probs, test_indices, test_acc = test(args, model, test_loader, name='Test')
    np.savez(os.path.join(output_dir, 'test.npz'), labels=test_labels, pred_probs=test_pred_probs, indices=test_indices)

    ood_labels, ood_pred_probs, ood_indices, ood_acc = test(args, model, ood_loader, name='OOD')
    np.savez(os.path.join(output_dir, 'ood.npz'), labels=ood_labels, pred_probs=ood_pred_probs, indices=ood_indices)


if __name__ == '__main__':
    main()
