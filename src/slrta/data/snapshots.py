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


def compute_node_features(window_df: pd.DataFrame, node_to_idx: dict, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(node_to_idx)
    # features: in_deg, out_deg, total_amount, avg_amount, freq, tac, degree, log_degree
    feats = np.zeros((n, 8), dtype=np.float32)
    degree_arr = np.zeros((n,), dtype=np.float32)

    if window_df.empty:
        return feats, degree_arr

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
        degree_arr[idx] = float(degree)

    return feats, degree_arr


def build_dynamic_snapshots(
    clean_df: pd.DataFrame, out_dir: str, window_size: int = 1,
    normalize_features: bool = True, train_ratio: float = 0.7,
) -> None:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if clean_df.empty:
        raise ValueError("Cannot build snapshots from an empty transaction dataframe")
    required_columns = {"source", "target", "amount", "timestamp", "label"}
    missing = required_columns.difference(clean_df.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # Remove stale generated snapshots so reruns cannot silently mix datasets.
    for stale_snapshot in out_path.glob("graph_*.pt"):
        stale_snapshot.unlink()

    nodes = pd.Index(pd.concat([clean_df["source"], clean_df["target"]], ignore_index=True).astype(str).unique())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Build one label per transaction node, never a degree-weighted vote.
    node_labels = {}
    if "source_label" in clean_df.columns and "target_label" in clean_df.columns:
        for _, row in clean_df[["source", "target", "source_label", "target_label"]].drop_duplicates().iterrows():
            for node, label in ((str(row["source"]), int(row["source_label"])),
                                (str(row["target"]), int(row["target_label"]))):
                if label in (0, 1):
                    if node in node_labels and node_labels[node] != label:
                        raise ValueError(f"Conflicting ground-truth labels for node {node}")
                    node_labels[node] = label
    else:
        for _, row in clean_df[["source", "label"]].drop_duplicates().iterrows():
            label = int(row["label"])
            if label in (0, 1):
                node = str(row["source"])
                if node in node_labels and node_labels[node] != label:
                    raise ValueError(f"Conflicting ground-truth labels for node {node}")
                node_labels[node] = label
    y_global = np.full((len(nodes),), -1, dtype=np.int64)
    for node, label in node_labels.items():
        y_global[node_to_idx[node]] = label

    # A node's first-seen time is the first transaction timestamp at which it
    # appears in the observed edge stream. Earlier snapshots must not supervise
    # a node that has not appeared yet.
    first_seen = {}
    if {"source_timestamp", "target_timestamp"}.issubset(clean_df.columns):
        for _, row in clean_df[["source", "target", "source_timestamp", "target_timestamp"]].iterrows():
            for node, ts_value in (
                (str(row["source"]), int(row["source_timestamp"])),
                (str(row["target"]), int(row["target_timestamp"])),
            ):
                first_seen[node] = min(first_seen.get(node, ts_value), ts_value)
    else:
        for _, row in clean_df[["source", "target", "timestamp"]].iterrows():
            ts = int(row["timestamp"])
            for node in (str(row["source"]), str(row["target"])):
                first_seen[node] = min(first_seen.get(node, ts), ts)
    first_seen_arr = np.full((len(nodes),), np.iinfo(np.int64).max, dtype=np.int64)
    for node, ts in first_seen.items():
        first_seen_arr[node_to_idx[node]] = ts

    ts_values = sorted(clean_df["timestamp"].unique().tolist())
    windows = []
    start = 0
    while start < len(ts_values):
        windows.append(ts_values[start : start + window_size])
        start += window_size

    # Temporal states for appended features:
    # 1) cumulative transaction count per node
    # 2) degree growth rate: deg_t - deg_{t-1}
    # 3) historical neighbor illicit ratio
    cumulative_tx_count = np.zeros((len(nodes),), dtype=np.float32)
    prev_degree = np.zeros((len(nodes),), dtype=np.float32)
    historical_neighbors = [set() for _ in range(len(nodes))]
    # Ground-truth labels are supervision only, never model input features.

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    snapshot_objects = []
    # Build cumulative snapshots: include edges with timestamp <= current window max
    for t, ts_window in enumerate(windows, start=1):
        cutoff = max(ts_window)
        wdf = clean_df[clean_df["timestamp"] <= cutoff].copy()
        current_df = clean_df[clean_df["timestamp"].isin(ts_window)].copy()

        # Cumulative edge counts must be normalized by elapsed timestamps, not just
        # the width of the latest window. Otherwise frequency grows spuriously with
        # history length (especially when window_size=1).
        elapsed_steps = int(wdf["timestamp"].nunique()) if not wdf.empty else 1
        base_feats, current_degree = compute_node_features(
            wdf, node_to_idx, window_size=max(1, elapsed_steps)
        )

        # Disable the former illicit-neighbor ratio: it used ground-truth labels
        # from the full dataset and leaked targets into model features.
        hist_illicit_ratio = np.zeros((len(nodes),), dtype=np.float32)

        # Update cumulative tx count using current window transactions.
        if not current_df.empty:
            src_idx_cur = current_df["source"].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
            dst_idx_cur = current_df["target"].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
            np.add.at(cumulative_tx_count, src_idx_cur, 1.0)
            np.add.at(cumulative_tx_count, dst_idx_cur, 1.0)
        degree_growth = current_degree - prev_degree
        prev_degree = current_degree.copy()

        temporal_feats = np.column_stack([cumulative_tx_count, degree_growth, hist_illicit_ratio]).astype(np.float32)
        x_np = np.concatenate([base_feats, temporal_feats], axis=1)
        # Do not fit a separate scaler per snapshot: that makes feature scales
        # incomparable and lets the full node universe affect each snapshot.
        # A training-only scaler can be added as a separate fitted transform.

        if wdf.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            src_idx = wdf["source"].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
            dst_idx = wdf["target"].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
            edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
            edge_attr = torch.tensor(wdf[["amount", "timestamp"]].to_numpy(dtype=np.float32), dtype=torch.float)

        # Ingest current edges into historical neighbor sets for next snapshot.
        if not current_df.empty:
            src_idx_cur = current_df["source"].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
            dst_idx_cur = current_df["target"].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
            for s_i, d_i in zip(src_idx_cur, dst_idx_cur):
                historical_neighbors[s_i].add(int(d_i))
                historical_neighbors[d_i].add(int(s_i))

        visible = first_seen_arr <= cutoff
        y_snapshot = y_global.copy()
        y_snapshot[~visible] = -1
        y = torch.tensor(y_snapshot, dtype=torch.float)
        labeled_mask = (y >= 0)

        data = Data(
            x=torch.tensor(x_np, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        data.labeled_mask = labeled_mask
        data.active_mask = torch.tensor(visible, dtype=torch.bool)
        data.snapshot_id = t
        data.time_values = torch.tensor(ts_window, dtype=torch.long)

        snapshot_objects.append(data)

    # Fit one scaler on active nodes in the chronological training prefix only.
    # Reuse these statistics for validation/test and for attack feature updates.
    train_count = max(1, int(len(snapshot_objects) * train_ratio))
    if normalize_features:
        fit_rows = [
            snap.x[snap.active_mask] for snap in snapshot_objects[:train_count]
            if bool(snap.active_mask.any())
        ]
        if not fit_rows:
            raise ValueError("No active training nodes available to fit feature scaler")
        fit_x = torch.cat(fit_rows, dim=0)
        feature_mean = fit_x.mean(dim=0)
        feature_std = fit_x.std(dim=0, unbiased=False)
        feature_std[feature_std < 1e-6] = 1.0
    else:
        feature_mean = torch.zeros(snapshot_objects[0].x.size(1), dtype=torch.float32)
        feature_std = torch.ones(snapshot_objects[0].x.size(1), dtype=torch.float32)

    for t, snap in enumerate(snapshot_objects, start=1):
        if normalize_features:
            snap.x = (snap.x - feature_mean) / feature_std
            # Keep future/inactive nodes at zero after applying the scaler.
            snap.x[~snap.active_mask] = 0.0
        snap.feature_mean = feature_mean.clone()
        snap.feature_std = feature_std.clone()
        torch.save(snap, out_path / f"graph_{t}.pt")

    metadata = {
        "num_nodes": len(nodes),
        "num_snapshots": len(windows),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "features_normalized": bool(normalize_features),
        "scaler_fit_snapshots": train_count,
        "node_to_idx": node_to_idx,
        "idx_to_node": {i: n for n, i in node_to_idx.items()},
    }
    torch.save(metadata, out_path / "metadata.pt")
