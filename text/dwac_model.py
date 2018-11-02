import torch
import torch.nn as nn
import torch.optim as optim


class AttentionCnnDwac(object):
    def __init__(self, args, vocab, embeddings_matrix):
        self.device = args.device
        self.n_classes = args.n_classes
        self.eps = args.eps

        self.model = AttentionCnnDwacModule(args, vocab, embeddings_matrix).to(self.device)
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
            ref_zs = [(self.model.get_representation(x.to(self.device))[0].cpu(), y)
                      for x, y, i in ref_loader]

            test_reps, test_ys, test_is = zip(*[(self.model.get_representation(x.to(self.device)), y, i)
                                              for x, y, i in test_loader])
            test_zs, test_alphas = zip(*[(rep[0].cpu(), rep[1]) for rep in test_reps])
            class_dists = []

            for z in test_zs:
                z = z.to(self.device)
                z_norm = z.pow(2).sum(dim=1)

                batch_class_dists = torch.zeros([z.shape[0], self.n_classes], device=z.device)
                for ref_z, ref_y in ref_zs:
                    ref_z, ref_y = ref_z.to(self.device), ref_y.to(self.device)
                    batch_output = self.model.classify_against_ref(z, z_norm, ref_z, ref_y)
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
                'att': test_alphas,
                'total_loss': total_loss,
                'loss': total_loss.div(len(test_loader.sampler)),
                'correct': correct,
                'accuracy': correct / len(test_loader.sampler),
            }
            return output_dict

    def embed(self, x):
        self.model.eval()
        with torch.no_grad():
            z, alpha = self.model.get_representation(x)
            output_dict = {'z': z, 'att': alpha}
        return output_dict

    def save(self, filepath):
        print("Saving model to ", filepath)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))


class AttentionCnnDwacModule(nn.Module):
    def __init__(self, args, vocab, embeddings_matrix):
        super(AttentionCnnDwacModule, self).__init__()
        self.device = args.device

        self.eps = args.eps
        self.gamma = args.gamma

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = args.embedding_dim

        if args.kernel == 'laplace':
            print("Using Laplace kernel")
            self.distance_metric = self._laplacian_kernel
        elif args.kernel == 'invquad':
            print("Using Inverse Quadratic kernel with smoothing parameter {:.3f}".format(self.gamma))
            self.distance_metric = self._inverse_quadratic
        else:
            print("Using Guassian kernel")
            self.distance_metric = self._gaussian_kernel

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim, self.vocab.pad_idx)
        if embeddings_matrix is not None:
            self.embedding_layer.weight = nn.Parameter(
                embeddings_matrix,
                requires_grad=args.update_embeddings)
        else:
            self.embedding_layer.weight.requires_grad = args.update_embeddings
        self.embedding_dropout = nn.Dropout(p=0.5)

        self.kernel_size = args.kernel_size
        self.hidden_dim = args.hidden_dim
        self.z_dim = args.z_dim
        self.n_classes = args.n_classes

        self.conv1_layer = nn.Conv1d(
            self.embedding_dim,
            self.hidden_dim,
            kernel_size = self.kernel_size,
            padding = self.kernel_size // 2,
        )

        self.attn_layer = nn.Linear(self.hidden_dim, 1)
        self.output_layer = nn.Linear(self.hidden_dim, self.z_dim)

        self.criterion = nn.NLLLoss(size_average=False)

    def get_representation(self, x):
        batch_size, max_len = x.shape
        padding_mask = (x != self.vocab.pad_idx).float().view([batch_size, max_len, 1])
        x = self.embedding_layer(x)
        x = self.embedding_dropout(x)
        x = x.transpose(1, 2)
        x = torch.tanh(self.conv1_layer(x))
        x = x.transpose(1, 2)
        a = self.attn_layer(x).exp().mul(padding_mask)
        an = a.sum(dim=1).pow(-1).view([batch_size, 1, 1])
        alpha = torch.bmm(a, an)
        x = x.mul(alpha)
        z = x.sum(dim=1)
        z = self.output_layer(z)
        return z, alpha.squeeze(2)

    def forward(self, x, y):
        z, att = self.get_representation(x)

        norm = z.pow(2).sum(dim=1)
        fast_dists = torch.mm(z, z.t()).mul(-2).add(norm).t().add(norm).t()
        fast_dists = self.distance_metric(fast_dists)
        fast_dists = fast_dists.mul((1 != torch.eye(z.shape[0], device=z.device)).float())

        class_mask = torch.zeros(z.shape[0],
                                 self.n_classes,
                                 device=z.device)
        class_mask.scatter_(1, y.view(z.shape[0], 1), 1)
        class_dists = torch.mm(fast_dists, class_mask).add(self.eps)  # [batch_size, n_classes]

        probs = torch.div(class_dists.t(), class_dists.sum(dim=1)).log().t()

        total_loss = self.criterion(probs, y)

        output_dict = {
            'probs': probs,
            'loss': total_loss.div(x.shape[0]),
            'total_loss': total_loss,
        }
        return output_dict

    def _gaussian_kernel(self, dists):
        return dists.mul_(-0.5 * self.gamma).exp_()

    def _laplacian_kernel(self, dists):
        return dists.pow_(0.5).mul_(-0.5 * self.gamma).exp_()

    def _inverse_quadratic(self, dists):
        return 1.0 / (self.gamma + dists)

    def classify_against_ref(self, z, z_norm, ref_z, ref_y):
        ref_norm = ref_z.pow(2).sum(dim=1)

        fast_dists = torch.mm(z, ref_z.t()).mul(-2).add(ref_norm).t().add(z_norm).t()

        fast_dists = self.distance_metric(fast_dists)

        class_mask = torch.zeros(ref_z.shape[0],
                                 self.n_classes,
                                 device=ref_z.device)
        class_mask.scatter_(1, ref_y.view(ref_z.shape[0], 1), 1)
        class_dists = torch.mm(fast_dists, class_mask)

        output_dict = {
            'class_dists': class_dists,
        }

        return output_dict
