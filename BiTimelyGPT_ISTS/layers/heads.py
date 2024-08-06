import torch.nn as nn
from layers.snippets import SigmoidRange

class PretrainHead(nn.Module):
    def __init__(self, d_model, n_output):
        super(PretrainHead, self).__init__()
        self.head = nn.Linear(d_model, n_output + 1)  # +1 to include the padding token as a class

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
