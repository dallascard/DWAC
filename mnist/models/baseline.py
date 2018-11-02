import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MNISTBaseline(object):

    def __init__(self, args):
        self.device = args.device
        self.n_classes = args.n_classes
        self.eps = args.eps

        self.model = MNISTBaselineModule().to(self.device)
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

    def get_representation(self, x):
        self.model.eval()
        with torch.no_grad():
            z = self.model.get_representation(x)
            output_dict = {'z': z}
            return output_dict

    def save(self, filepath):
        print("Saving model to {}".format(filepath))
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        print("Loading model from {}".format(filepath))
        self.model.load_state_dict(torch.load(filepath))


class MNISTBaselineModule(nn.Module):

    def __init__(self):
        super(MNISTBaselineModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

        self.criterion = nn.NLLLoss(size_average=False)

    def get_representation(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv2_dropout(x)
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        return x

    def classify(self, z):
        probs = F.log_softmax(self.fc2(z), dim=1)
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