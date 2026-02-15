import torch
import torch.nn as nn


class NormalizedGCNConv(nn.Module):
    """Plain PyTorch normalized GCN: D^-1/2 A_hat D^-1/2 XW over edge_index."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        n = h.size(0)

        if edge_index.numel() == 0:
            src = torch.arange(n, device=x.device, dtype=torch.long)
            dst = src
        else:
            src = edge_index[0]
            dst = edge_index[1]
            self_loop = torch.arange(n, device=x.device, dtype=torch.long)
            src = torch.cat([src, self_loop], dim=0)
            dst = torch.cat([dst, self_loop], dim=0)

        deg = torch.zeros(n, device=x.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device, dtype=h.dtype))
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        out = torch.zeros_like(h)
        out.index_add_(0, dst, h[src] * norm.unsqueeze(1))
        return out + self.bias


MeanGraphConv = NormalizedGCNConv
