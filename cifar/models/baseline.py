import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cifar.models.resnet import resnet18


class ResNetBaseline(object):
    def __init__(self, args):
        self.device = args.device
        self.n_classes = args.n_classes
        self.eps = args.eps

        self.model = ResNetBaselineModule(args).to(self.device)
        self.optim = optim.Adam((x for x in self.model.parameters() if x.requires_grad), args.lr)

    def fit(self, x, y):
        self.model.train()
        self.optim.zero_grad()
        output_dict = self.model(x, y)
        output_dict['loss'].backward()
        self.optim.step()
        return output_dict

    def evaluate(self, test_loader, ref_loader=None):
        self.model.eval()
        with torch.no_grad():
            test_zs, test_ys, test_is = zip(*[(self.model.get_representation(x.to(self.device)).cpu(), y, i)
                                            for x, y, i in test_loader])
            probs = [self.model.classify(z.to(self.device)).cpu() for z in test_zs]

            test_zs = torch.cat(test_zs, dim=0)
            test_ys = torch.cat(test_ys, dim=0)
            test_is = torch.cat(test_is, dim=0)
            probs = torch.cat(probs, dim=0)

            total_loss = self.model.criterion(probs, test_ys)
            correct = torch.eq(probs.argmax(dim=1), test_ys).sum().item()

            output_dict = {
                    'zs': test_zs,
                    'ys': test_ys,
                    'is': test_is,
                    'probs': probs,
                    'total_loss': total_loss,
                    'loss': total_loss.div(len(test_loader.sampler)),
                    'correct': correct,
                    'accuracy': 100 * correct / len(test_loader.sampler),
                    }
        return output_dict

    def embed(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            test_zs, test_ys, test_is = zip(*[(self.model.get_representation(x.to(self.device)).cpu(), y, i)
                                            for x, y, i in test_loader])
            test_zs = torch.cat(test_zs, dim=0)
            test_ys = torch.cat(test_ys, dim=0)
            test_is = torch.cat(test_is, dim=0)

            output_dict = {
                    'zs': test_zs,
                    'ys': test_ys,
                    'is': test_is,
                    }
        return output_dict

    def save(self, filepath):
        print("Saving model to {}".format(filepath))
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        print("Loading model from {}".format(filepath))
        self.model.load_state_dict(torch.load(filepath))


class ResNetBaselineModule(nn.Module):
    def __init__(self, args):
        super(ResNetBaselineModule, self).__init__()
        self.z_dim = args.z_dim
        self.n_classes = args.n_classes

        self.resnet = resnet18(self.z_dim)
        self.classifying_layer = nn.Linear(self.z_dim, self.n_classes)

        self.criterion = nn.NLLLoss(size_average=False)

    def get_representation(self, x):
        z = self.resnet(x)
        return z

    def classify(self, z):
        probs = F.log_softmax(self.classifying_layer(z), dim=1)
        return probs

    def forward(self, x, y):
        z = self.get_representation(x)
        probs = self.classify(z)

        total_loss = self.criterion(probs, y)

        output_dict = {
                'probs': probs,
                'loss': total_loss.div(x.shape[0]),
                'total_loss': total_loss,
                }
        return output_dict


