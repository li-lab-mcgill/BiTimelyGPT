import time
import pandas as pd
import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbeddingLearnable(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbeddingLearnable, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TokenEmbeddingFixed(nn.Module):
    def __init__(self, embedding_path):
        super(TokenEmbeddingFixed, self).__init__()

        # Load pre-trained embeddings
        embedding_df = pd.read_csv(embedding_path, index_col=0)
        embeddings = torch.tensor(embedding_df.values, dtype=torch.float32)
        # Create PheCode to index mapping
        self.phecode_to_index = {int(phecode): idx for idx, phecode in enumerate(embedding_df.index)}
        # Add a padding index entry
        self.padding_idx = len(self.phecode_to_index)
        embeddings = torch.cat([embeddings, torch.zeros(1, embeddings.shape[1])], dim=0)  # Add a zero vector for padding
        # Create embedding layer with pre-trained weights
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=self.padding_idx)

    def get_num_tokens(self):
        return len(self.phecode_to_index)  # Exclude the padding index

    def map_phecodes_to_indices(self, phecodes):
        indices = [self.phecode_to_index.get(int(phecode), self.padding_idx) for phecode in phecodes]
        return torch.tensor(indices, dtype=torch.int64)

    def forward(self, x):
        # Convert PheCode to indices using map_phecodes_to_indices
        indices = self.map_phecodes_to_indices(x).to(x.device)
        return self.emb(indices).detach()