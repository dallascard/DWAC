import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MNISTDwac(object):

    def __init__(self, args):
        self.device = args.device
        self.n_classes = args.n_classes
        self.eps = args.eps

        self.model = MNISTDwacModule(args).to(self.device)
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
            ref_data = [(self.model.get_representation(x.to(self.device)).cpu(), y)
                                            for x, y, i in ref_loader]
            test_zs, test_ys, test_is = zip(*[(self.model.get_representation(x.to(self.device)).cpu(), y, i)
                                            for x, y, i in test_loader])
            class_dists = []

            for z in test_zs:
                z = z.to(self.device)
                z_norm = z.pow(2).sum(dim=1)

                batch_class_dists = torch.zeros([z.shape[0], self.n_classes], device=z.device)
                for ref_z, ref_y in ref_data:
                    ref_z, ref_y = ref_z.to(self.device), ref_y.to(self.device)
                    batch_output = self.model.classify(z, z_norm, ref_z, ref_y)
                    batch_class_dists.add_(batch_output['class_dists'])
                class_dists.append(batch_class_dists.cpu())

            test_zs = torch.cat(test_zs, dim=0)
            test_ys = torch.cat(test_ys, dim=0)
            test_is = torch.cat(test_is, dim=0)

            class_dists = torch.cat(class_dists, dim=0)
            probs = class_dists.div(class_dists.sum(dim=1, keepdim=True)).log()

            total_loss = self.model.criterion(probs, test_ys)
            correct = torch.eq(probs.argmax(dim=1), test_ys).sum().item()

            output_dict = {
                    'zs': test_zs,
                    'ys': test_ys,
                    'is': test_is,
                    'probs': probs,
                    'confs': class_dists,
                    'total_loss': total_loss,
                    'loss': total_loss / len(test_loader.sampler),
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
        print("Saving model to ", filepath)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))


class MNISTDwacModule(nn.Module):
    def __init__(self, args):
        super(MNISTDwacModule, self).__init__()
        self.eps = args.eps
        self.gamma = args.gamma
        self.z_dim = args.z_dim
        self.n_classes = args.n_classes

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, self.z_dim)

        self.criterion = nn.NLLLoss(size_average=False)

        if args.kernel == 'laplace':
            print("Using Laplace kernel")
            self.distance_metric = self._laplacian_kernel
        elif args.kernel == 'invquad':
            print("Using Inverse Quadratic kernel with smoothing parameter {:.3f}".format(self.gamma))
            self.distance_metric = self._inverse_quadratic
        elif args.kernel == 'gaussian':
            print("Using Guassian kernel")
            self.distance_metric = self._gaussian_kernel
        elif args.kernel == 'sigmoid':
            print("Using Sigmoid kernel")
            self.distance_metric = self._sigmoid
        elif args.kernel == 'softplus':
            print("Using Softplus kernel")
            self.distance_metric = self._softplus
        elif args.kernel == 'relu':
            print("Using ReLU kernel")
            self.distance_metric = self._relu
        else:
            raise ValueError('Invalid Kernel')

    def get_representation(self, x):
        z = F.relu(self.conv1(x))
        z = F.relu(F.max_pool2d(self.conv2(z), 2))
        z = self.conv2_dropout(z)
        z = z.view(z.shape[0], 9216)
        z = F.relu(self.fc1(z))
        z = self.fc1_dropout(z)
        z = self.fc2(z)
        return z

    def forward(self, x, y):
        z = self.get_representation(x)

        norm = z.pow(2).sum(dim=1)
        dists = torch.mm(z, z.t()).mul(-2).add(norm).t().add(norm).t()
        dists = self.distance_metric(dists)
        dists = dists.mul((1 != torch.eye(z.shape[0], device=z.device)).float())

        class_mask = torch.zeros(z.shape[0], 
                                 self.n_classes,
                                 device=z.device)
        class_mask.scatter_(1, y.view(z.shape[0], 1), 1)
        class_dists = torch.mm(dists, class_mask).add(self.eps)  # [batch_size, n_classes]
        probs = torch.div(class_dists.t(), class_dists.sum(dim=1)).log().t()

        total_loss = self.criterion(probs, y)
        output_dict = {
                'probs': probs,
                'loss': total_loss.div(x.shape[0]),
                'total_loss': total_loss,
                }
        return output_dict

    def classify(self, z, z_norm, ref_z, ref_y):
        ref_norm = ref_z.pow(2).sum(dim=1)
        dists = torch.mm(z, ref_z.t()).mul(-2).add(ref_norm).t().add(z_norm).t()
        dists = self.distance_metric(dists)

        class_mask = torch.zeros(ref_z.shape[0], 
                                 self.n_classes,
                                 device=ref_z.device)
        class_mask.scatter_(1, ref_y.view(ref_z.shape[0], 1), 1)
        class_dists = torch.mm(dists, class_mask)

        output_dict = {
                'class_dists': class_dists,
                }
        return output_dict

    def _gaussian_kernel(self, dists):
            return dists.mul_(-1 * self.gamma).exp_()

    def _laplacian_kernel(self, dists):
            return dists.pow_(0.5).mul_(-0.5 * self.gamma).exp_()

    def _inverse_quadratic(self, dists):
        return 1.0 / (self.gamma + dists)

    def _sigmoid(self, dists):
        return F.sigmoid(self.gamma - dists)

    def _softplus(self, dists):
        return F.softplus(self.gamma - dists)

    def _relu(self, dists):
        return F.relu(self.gamma - dists)
