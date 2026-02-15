import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MeanGraphConv


class StaticGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = MeanGraphConv(in_dim, hidden_dim)
        self.conv2 = MeanGraphConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.head(x).squeeze(-1)
        return x
