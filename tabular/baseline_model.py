import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TabularBaseline(object):

    def __init__(self, args):
        self.device = args.device
        self.model = MLP(args).to(self.device)
        self.optim = optim.Adam((x for x in self.model.parameters() if x.requires_grad), args.lr)

    def fit(self, x, y):
        self.model.train()
        self.optim.zero_grad()
        output_dict = self.model(x, y)
        output_dict['loss'].backward()
        self.optim.step()
        return output_dict

    def evaluate(self, test_loader, ref_loader):
        self.model.eval()
        with torch.no_grad():

            test_zs, test_ys, test_is = zip(*[(self.model.get_representation(x.float().to(self.device)), y.to(self.device), i)
                                              for x, y, i in test_loader])
            test_zs = torch.cat(test_zs, dim=0)
            test_ys = torch.cat(test_ys, dim=0)
            test_is = torch.cat(test_is, dim=0)

            log_probs = F.log_softmax(test_zs, dim=1)

            total_loss = self.model.criterion(log_probs, test_ys)
            correct = torch.eq(log_probs.argmax(dim=1), test_ys).sum().item()

            output_dict = {
                'zs': test_zs,
                'ys': test_ys,
                'is': test_is,
                'probs': log_probs,
                'confs': torch.ones(log_probs.size()),
                'total_loss': total_loss,
                'loss': total_loss.div(len(test_loader.sampler)),
                'correct': correct,
                'accuracy': correct / len(test_loader.sampler),
            }
            return output_dict

    def predict(self, x, ref_loader=None):
        self.model.eval()
        with torch.no_grad():
            output_dict = self.model(x=x, y=None)
        return output_dict

    def embed(self, x):
        self.model.eval()
        with torch.no_grad():
            z = self.model.get_representation(x)
            output_dict = {'z': z}
        return output_dict

    def save(self, filepath):
        print("Saving model to ", filepath)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))


class MLP(nn.Module):

    def __init__(self, args):
        super(MLP, self).__init__()
        self.x_dim = args.xdim
        self.dh1 = args.dh1
        self.dh2 = args.dh2
        self.n_classes = args.n_classes
        self.dropout_prob = args.dropout
        self.fc1 = nn.Linear(self.x_dim, self.dh1)
        self.fc1_dropout = nn.Dropout(p=self.dropout_prob)
        if args.dh2 > 0:
            self.fc2 = nn.Linear(self.dh1, self.dh2)
            self.fc2_dropout = nn.Dropout(p=self.dropout_prob)
            self.fc3 = nn.Linear(self.dh2, self.n_classes)
        else:
            self.fc2 = nn.Linear(self.dh1, self.n_classes)

        self.criterion = nn.NLLLoss(size_average=False)

    def forward(self, x, y):
        z = self.get_representation(x)
        log_probs = F.log_softmax(z, dim=1)
        output_dict = {'probs': log_probs, 'z': z}

        if y is not None:
            total_loss = self.criterion(log_probs, y)
            loss = total_loss / x.shape[0]
            output_dict['total_loss'] = total_loss
            output_dict['loss'] = loss

        return output_dict

    def get_representation(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        if self.dh2 > 0:
            x = self.fc2_dropout(x)
            x = self.fc3(x)
        return x
