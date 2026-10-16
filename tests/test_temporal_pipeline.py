import pandas as pd
import torch

from slrta.attack.low_rate import _append_edge_to_future_snapshots, _recompute_cumulative_features
from slrta.data.clean import load_elliptic
from slrta.data.io import load_snapshots
from slrta.data.snapshots import Data, build_dynamic_snapshots
from slrta.models.dynamic_gcn_gru import DynamicGCNGRU


def test_elliptic_loader_maps_both_endpoint_labels_and_timestamps(tmp_path):
    pd.DataFrame({"txId1": [10, 20], "txId2": [20, 30]}).to_csv(
        tmp_path / "elliptic_txs_edgelist.csv", index=False
    )
    pd.DataFrame(
        {"txId": [10, 20, 30], "class": ["1", "2", "unknown"]}
    ).to_csv(tmp_path / "elliptic_txs_classes.csv", index=False)
    pd.DataFrame([[10, 1, 0.0], [20, 2, 0.0], [30, 3, 0.0]]).to_csv(
        tmp_path / "elliptic_txs_features.csv", index=False, header=False
    )

    result = load_elliptic(str(tmp_path))

    assert result["source_label"].tolist() == [1, 0]
    assert result["target_label"].tolist() == [0, -1]
    assert result["timestamp"].tolist() == [1, 2]
    assert result["amount"].tolist() == [1.0, 1.0]


def test_cumulative_snapshot_frequency_uses_elapsed_timestamps(tmp_path):
    frame = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "amount": [1.0, 1000.0],
            "timestamp": [1, 2],
            "label": [0, 0],
            "source_label": [0, 0],
            "target_label": [0, 1],
        }
    )
    build_dynamic_snapshots(frame, str(tmp_path), window_size=1)
    snapshots = load_snapshots(str(tmp_path))

    # B participates in one edge at t=1, then two cumulative edges across two times.
    assert abs(snapshots[0].x[1, 4].item() - snapshots[1].x[1, 4].item()) < 1e-6
    assert snapshots[0].feature_mean.shape[0] == snapshots[0].x.shape[1]
    assert snapshots[0].feature_std.shape[0] == snapshots[0].x.shape[1]
    # The future-only amount of 1000 must not influence the training-prefix scaler.
    assert snapshots[0].feature_mean[2].item() == 1.0
    assert snapshots[0].active_mask.tolist() == [True, True, False]
    assert snapshots[1].active_mask.tolist() == [True, True, True]


def test_injected_edge_persists_from_intervention_time_forward():
    snapshots = [
        Data(
            x=torch.zeros((3, 11)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 2)),
            y=torch.tensor([0.0, 0.0, 1.0]),
            active_mask=torch.ones(3, dtype=torch.bool),
            time_values=torch.tensor([t]),
        )
        for t in (1, 2, 3)
    ]

    _append_edge_to_future_snapshots(snapshots, 1, 0, 2, 1.5, 2)

    assert snapshots[0].edge_index.size(1) == 0
    assert snapshots[1].edge_index[:, -1].tolist() == [0, 2]
    assert snapshots[2].edge_index[:, -1].tolist() == [0, 2]
    assert snapshots[1].edge_attr[-1].tolist() == [1.5, 2.0]


def test_feature_recomputation_updates_cumulative_degree():
    snapshots = [
        Data(
            x=torch.zeros((3, 11)),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.tensor([[1.0, 1.0]]),
            y=torch.tensor([0.0, 0.0, 1.0]),
            time_values=torch.tensor([1]),
        ),
        Data(
            x=torch.zeros((3, 11)),
            edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            edge_attr=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            y=torch.tensor([0.0, 0.0, 1.0]),
            time_values=torch.tensor([1, 2]),
        ),
    ]

    _recompute_cumulative_features(snapshots)

    assert snapshots[0].x[0, 6].item() == 1.0
    assert snapshots[1].x[0, 6].item() == 2.0
    assert snapshots[1].x[0, 4].item() == 1.0


def test_inactive_nodes_do_not_accumulate_gru_state():
    model = DynamicGCNGRU(in_dim=11, hidden_dim=4, dropout=0.0)
    data = Data(
        x=torch.zeros((2, 11)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        active_mask=torch.tensor([True, False]),
    )
    previous = torch.ones((2, 4))

    _, hidden = model.forward_step(data, previous)

    assert torch.count_nonzero(hidden[1]).item() == 0



def test_snapshot_builder_rejects_invalid_window_size(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="window_size"):
        build_dynamic_snapshots(
            pd.DataFrame(
                {"source": ["A"], "target": ["B"], "amount": [1.0],
                 "timestamp": [1], "label": [0]}
            ),
            str(tmp_path),
            window_size=0,
        )
