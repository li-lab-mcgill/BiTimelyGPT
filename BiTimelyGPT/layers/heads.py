import torch.nn as nn
from layers.snippets import SigmoidRange


class PretrainHead(nn.Module):
    def __init__(self, d_model, n_output):
        super(PretrainHead, self).__init__()
        self.head = nn.Linear(d_model, n_output)  # +1 to include the padding token as a class

    def forward(self, x):
        """
        x: tensor [batch_size x seq_len x d_model]
        output: tensor [batch_size x seq_len x n_output]
        """
        return self.head(x)


class ClfHead(nn.Module):
    def __init__(self, d_model, n_output):
        super(ClfHead, self).__init__()
        self.head = nn.Linear(d_model, n_output)

    def forward(self, x):
        """
        x: tensor [batch_size x seq_len x d_model]
        output: tensor [batch_size x num_classes]
        """
        # x = x[:, 0, :]     # Only use the [sos] token
        x = x.mean(dim=1)    # Average pool over the hidden states
        return self.head(x)


class RegrHead(nn.Module):
    def __init__(self, d_model, output_dim, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.regr_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        # todo: check average-first-last pooling
        x = x.mean(dim=1)         # Average pool over the sequence dimension
        y = self.regr_layer(x)
        if self.y_range: y = SigmoidRange(*self.y_range)(y)

        return y


