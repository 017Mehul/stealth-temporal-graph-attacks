import torch
import torch.nn as nn


class NormalizedGCNConv(nn.Module):
    """Directed normalized message passing using D_in^-1/2 A_hat D_out^-1/2 XW."""

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

        ones = torch.ones(src.size(0), device=x.device, dtype=h.dtype)
        deg_out = torch.zeros(n, device=x.device, dtype=h.dtype)
        deg_in = torch.zeros(n, device=x.device, dtype=h.dtype)
        deg_out.index_add_(0, src, ones)
        deg_in.index_add_(0, dst, ones)
        deg_out_inv_sqrt = deg_out.clamp(min=1.0).pow(-0.5)
        deg_in_inv_sqrt = deg_in.clamp(min=1.0).pow(-0.5)
        norm = deg_out_inv_sqrt[src] * deg_in_inv_sqrt[dst]

        out = torch.zeros_like(h)
        out.index_add_(0, dst, h[src] * norm.unsqueeze(1))
        return out + self.bias


MeanGraphConv = NormalizedGCNConv
