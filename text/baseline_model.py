import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TextBaseline(object):
    def __init__(self, args, vocab, embeddings_matrix):
        self.device = args.device

        self.model = BaselineModel(args, vocab, embeddings_matrix).to(self.device)
        self.optim = optim.Adam((x for x in self.model.parameters() if x.requires_grad), args.lr)

    def fit(self, x, y):
        self.model.train()
        self.optim.zero_grad()
        output_dict = self.model(x, y)
        output_dict['loss'].backward()
        self.optim.step()
        return output_dict

    def evaluate(self, x, y, ref_loader):
        self.model.eval()
        with torch.no_grad():
            output_dict = self.model(x, y)
        return output_dict

    def predict(self, x, ref_loader=None):
        self.model.eval()
        with torch.no_grad():
            output_dict = self.model(x=x, y=None)
        return output_dict

    def embed(self, x):
        self.model.eval()
        with torch.no_grad():
            output_dict = self.model(x=x, y=None)
        return output_dict

    def save(self, filepath):
        print("Saving model to ", filepath)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))


class BaselineModel(nn.Module):
    def __init__(self, args, vocab, embeddings_matrix):
        super(BaselineModel, self).__init__()
        self.device = args.device

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = args.embedding_dim

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
        self.n_classes = args.n_classes

        self.conv1_layer = nn.Conv1d(
            self.embedding_dim,
            self.hidden_dim,
            kernel_size = self.kernel_size,
            padding = self.kernel_size // 2,
        )

        self.attn_layer = nn.Linear(self.hidden_dim, 1)
        self.classification_layer = nn.Linear(self.hidden_dim, self.n_classes)


        self.criterion = nn.NLLLoss(size_average=False)

    def get_representation(self, x):
        batch_size, max_len = x.shape
        padding_mask = (x != self.vocab.pad_idx).float().view([batch_size, max_len, 1])
        x = self.embedding_layer(x)
        x = self.embedding_dropout(x)

        x = x.transpose(1, 2)
        x = torch.tanh(self.conv1_layer(x))
        x = x.transpose(1, 2)

        #alpha = F.softmax(self.attn_layer(x).mul_(padding_mask), dim=1)
        a = self.attn_layer(x).exp().mul(padding_mask)
        an = a.sum(dim=1).pow(-1).view([batch_size, 1, 1])
        alpha = torch.bmm(a, an)
        x = x.mul(alpha)
        z = x.sum(dim=1)
        return z, alpha.squeeze(2)

    def forward(self, x, y):
        z, alpha = self.get_representation(x)
        probs = F.log_softmax(self.classification_layer(z), dim=1)
        output_dict = {'z': z, 'probs': probs, 'att': alpha}

        if y is not None:
            total_loss = self.criterion(probs, y)
            loss = total_loss / x.shape[0]
            output_dict['total_loss'] = total_loss
            output_dict['loss'] = loss

        return output_dict
