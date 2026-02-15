from pathlib import Path

import torch


def load_snapshots(dynamic_dir: str):
    dynamic_path = Path(dynamic_dir)
    files = sorted(dynamic_path.glob("graph_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"No snapshots found in {dynamic_dir}")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def split_indices(n: int, train_ratio: float, val_ratio: float):
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_val = max(1, n - n_train - n_test)
    train = list(range(0, n_train))
    val = list(range(n_train, n_train + n_val))
    test = list(range(n_train + n_val, n))
    return train, val, test
