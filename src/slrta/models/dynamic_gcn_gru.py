import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MeanGraphConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = MeanGraphConv(in_dim, hidden_dim)
        self.conv2 = MeanGraphConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class DynamicGCNGRU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, dropout)
        self.gru_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward_step(self, data, h_prev=None):
        z = self.encoder(data.x, data.edge_index)
        if h_prev is None:
            h_prev = torch.zeros_like(z)
        h_t = self.gru_cell(z, h_prev)
        logits_t = self.head(h_t).squeeze(-1)
        return logits_t, h_t

    def forward_embeddings(self, snapshots):
        logits = []
        h = None
        for data in snapshots:
            logit_t, h = self.forward_step(data, h)
            logits.append(logit_t)
        return torch.stack(logits, dim=0)
