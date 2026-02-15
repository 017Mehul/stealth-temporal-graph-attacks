from pathlib import Path

import numpy as np
import pandas as pd
import torch
try:
    from torch_geometric.data import Data  # type: ignore
except ImportError:
    class Data:  # Minimal fallback when torch-geometric is unavailable.
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to(self, device):
            out = Data()
            for k, v in self.__dict__.items():
                if torch.is_tensor(v):
                    setattr(out, k, v.to(device))
                else:
                    setattr(out, k, v)
            return out


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mu) / std


def compute_node_features(window_df: pd.DataFrame, node_to_idx: dict, window_size: int) -> np.ndarray:
    n = len(node_to_idx)
    # features: in_deg, out_deg, total_amount, avg_amount, freq, tac, degree, log_degree
    feats = np.zeros((n, 8), dtype=np.float32)

    if window_df.empty:
        return feats

    src = window_df["source"].astype(str).to_numpy()
    dst = window_df["target"].astype(str).to_numpy()
    amt = window_df["amount"].astype(float).to_numpy()
    ts = window_df["timestamp"].astype(int).to_numpy()

    in_deg = {}
    out_deg = {}
    total_amt = {}
    count_amt = {}
    time_sets = {}

    for s, d, a, t in zip(src, dst, amt, ts):
        out_deg[s] = out_deg.get(s, 0) + 1
        in_deg[d] = in_deg.get(d, 0) + 1

        total_amt[s] = total_amt.get(s, 0.0) + a
        total_amt[d] = total_amt.get(d, 0.0) + a

        count_amt[s] = count_amt.get(s, 0) + 1
        count_amt[d] = count_amt.get(d, 0) + 1

        time_sets.setdefault(s, set()).add(t)
        time_sets.setdefault(d, set()).add(t)

    all_nodes = set(src).union(set(dst))
    for node in all_nodes:
        idx = node_to_idx[node]
        i_deg = in_deg.get(node, 0)
        o_deg = out_deg.get(node, 0)
        tot = total_amt.get(node, 0.0)
        c = count_amt.get(node, 0)
        avg = tot / c if c > 0 else 0.0
        freq = c / max(window_size, 1)
        tac = len(time_sets.get(node, set()))
        degree = i_deg + o_deg
        log_degree = float(np.log1p(degree))
        feats[idx] = np.array([i_deg, o_deg, tot, avg, freq, tac, degree, log_degree], dtype=np.float32)

    return feats


def build_dynamic_snapshots(clean_df: pd.DataFrame, out_dir: str, window_size: int = 1, normalize_features: bool = True) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    nodes = pd.Index(pd.concat([clean_df["source"], clean_df["target"]], ignore_index=True).astype(str).unique())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    label_votes = {}
    for _, row in clean_df.iterrows():
        s, y = str(row["source"]), int(row["label"])
        if y in (0, 1):
            label_votes.setdefault(s, []).append(y)

    y_global = np.full((len(nodes),), -1, dtype=np.int64)
    for node, ys in label_votes.items():
        idx = node_to_idx[node]
        y_global[idx] = 1 if (sum(ys) >= len(ys) / 2) else 0

    ts_values = sorted(clean_df["timestamp"].unique().tolist())
    windows = []
    start = 0
    while start < len(ts_values):
        windows.append(ts_values[start : start + window_size])
        start += window_size

    # Build cumulative snapshots: include edges with timestamp <= current window max
    for t, ts_window in enumerate(windows, start=1):
        cutoff = max(ts_window)
        wdf = clean_df[clean_df["timestamp"] <= cutoff].copy()

        # For cumulative snapshots, window_size is the number of original timestamps in the window
        x_np = compute_node_features(wdf, node_to_idx, window_size=len(ts_window))
        if normalize_features:
            x_np = _zscore(x_np)

        if wdf.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            src_idx = wdf["source"].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
            dst_idx = wdf["target"].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
            edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
            edge_attr = torch.tensor(wdf[["amount", "timestamp"]].to_numpy(dtype=np.float32), dtype=torch.float)

        y = torch.tensor(y_global, dtype=torch.float)
        labeled_mask = (y >= 0)

        data = Data(
            x=torch.tensor(x_np, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        data.labeled_mask = labeled_mask
        data.snapshot_id = t
        data.time_values = torch.tensor(ts_window, dtype=torch.long)

        torch.save(data, out_path / f"graph_{t}.pt")

    metadata = {
        "num_nodes": len(nodes),
        "num_snapshots": len(windows),
        "node_to_idx": node_to_idx,
        "idx_to_node": {i: n for n, i in node_to_idx.items()},
    }
    torch.save(metadata, out_path / "metadata.pt")
