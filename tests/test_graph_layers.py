import math

import torch

from slrta.models.layers import NormalizedGCNConv


def test_directed_gcn_uses_source_out_degree_and_target_in_degree():
    layer = NormalizedGCNConv(1, 1)
    with torch.no_grad():
        layer.lin.weight.fill_(1.0)
        layer.bias.zero_()

    x = torch.tensor([[1.0], [2.0], [4.0]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    actual = layer(x, edge_index).squeeze(-1)

    expected_node0 = 1.0 / math.sqrt(2.0)
    expected_node1 = 1.0 * 0.5 + 2.0 / math.sqrt(2.0)
    expected_node2 = 4.0
    assert torch.allclose(
        actual,
        torch.tensor([expected_node0, expected_node1, expected_node2]),
        atol=1e-6,
    )
