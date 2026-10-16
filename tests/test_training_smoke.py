import json

import torch
import yaml

from slrta.data.snapshots import Data
from slrta.train_baseline import main as train_baseline
from slrta.train_dynamic import main as train_dynamic


def _write_tiny_training_fixture(root):
    graph_dir = root / "graphs"
    graph_dir.mkdir(parents=True)
    torch.manual_seed(7)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    for step in range(10):
        data = Data(
            x=torch.randn(4, 11),
            edge_index=edge_index,
            edge_attr=torch.ones((4, 2)),
            y=y.clone(),
            labeled_mask=torch.ones(4, dtype=torch.bool),
            active_mask=torch.ones(4, dtype=torch.bool),
            time_values=torch.tensor([step + 1]),
        )
        data.snapshot_id = step + 1
        torch.save(data, graph_dir / f"graph_{step + 1}.pt")

    cfg = {
        "project": {"seed": 7, "device": "cpu"},
        "paths": {
            "raw_dir": str(root / "raw"),
            "processed_csv": str(root / "processed" / "clean.csv"),
            "dynamic_dir": str(graph_dir),
            "checkpoints_dir": str(root / "checkpoints"),
            "metrics_dir": str(root / "metrics"),
            "plots_dir": str(root / "plots"),
        },
        "training": {
            "epochs": 2,
            "lr": 0.005,
            "weight_decay": 0.0005,
            "dropout": 0.0,
            "hidden_dim": 8,
            "truncate_bptt": 2,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
        },
    }
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path, root


def test_dynamic_training_smoke_test(tmp_path):
    cfg_path, root = _write_tiny_training_fixture(tmp_path / "dynamic")
    train_dynamic(str(cfg_path))

    result = json.loads((root / "metrics" / "dynamic_results.json").read_text())
    assert result["test_metrics"]["n_samples"] > 0
    assert (root / "checkpoints" / "dynamic_gcn_gru_best.pt").exists()


def test_static_baseline_training_smoke_test(tmp_path):
    cfg_path, root = _write_tiny_training_fixture(tmp_path / "static")
    train_baseline(str(cfg_path))

    result = json.loads((root / "metrics" / "baseline_results.json").read_text())
    assert result["test_metrics"]["n_samples"] > 0
    assert (root / "checkpoints" / "static_gcn_best.pt").exists()
