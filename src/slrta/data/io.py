from pathlib import Path

import torch


def load_snapshots(dynamic_dir: str):
    dynamic_path = Path(dynamic_dir)
    files = sorted(dynamic_path.glob("graph_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"No snapshots found in {dynamic_dir}")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def split_indices(n: int, train_ratio: float, val_ratio: float):
    """Return contiguous chronological splits and reject impossible configurations."""
    if n < 3:
        raise ValueError(f"At least 3 snapshots are required for train/val/test; got {n}")
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must each be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        raise ValueError(
            f"Ratios leave no test snapshots (n={n}, train={n_train}, val={n_val}); "
            "adjust train_ratio/val_ratio."
        )
    return list(range(n_train)), list(range(n_train, n_train + n_val)), list(range(n_train + n_val, n))
