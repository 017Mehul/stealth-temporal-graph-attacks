from pathlib import Path

import pandas as pd


CANONICAL = {
    "source": ["source", "src", "from", "from_id", "sender", "txid1"],
    "target": ["target", "dst", "to", "to_id", "receiver", "txid2"],
    "amount": ["amount", "value", "transaction_amount", "amt"],
    "timestamp": ["timestamp", "time", "time_step", "timestep", "ts"],
    "label": ["label", "class", "fraud", "is_fraud", "y"],
}


def _resolve_col(df: pd.DataFrame, name: str) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in CANONICAL[name]:
        if cand in cols:
            return cols[cand]
    raise ValueError(f"Could not resolve required column '{name}' from columns: {list(df.columns)}")


def _resolve_elliptic_root(raw_dir: str) -> Path:
    raw = Path(raw_dir)
    candidates = [raw, raw / "elliptic_bitcoin_dataset"]
    required = ["elliptic_txs_edgelist.csv", "elliptic_txs_classes.csv", "elliptic_txs_features.csv"]
    for cand in candidates:
        if all((cand / f).exists() for f in required):
            return cand
    raise FileNotFoundError(
        "Elliptic files missing. Expected elliptic_txs_edgelist.csv, "
        "elliptic_txs_classes.csv, and elliptic_txs_features.csv in data/raw/ "
        "or data/raw/elliptic_bitcoin_dataset/."
    )


def load_elliptic(raw_dir: str) -> pd.DataFrame:
    """Load Elliptic edges while retaining labels for both endpoint transactions."""
    raw = _resolve_elliptic_root(raw_dir)
    edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv").rename(
        columns={"txId1": "source", "txId2": "target"}
    )
    classes = pd.read_csv(raw / "elliptic_txs_classes.csv").rename(
        columns={"txId": "node_id", "class": "raw_label"}
    )
    features = pd.read_csv(raw / "elliptic_txs_features.csv", header=None, usecols=[0, 1])
    features.columns = ["node_id", "timestamp"]
    features["node_id"] = features["node_id"].astype(str)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    label_map = {"1": 1, "2": 0, "unknown": -1, 1: 1, 2: 0}
    classes["label"] = classes["raw_label"].map(label_map).fillna(-1).astype(int)
    classes["node_id"] = classes["node_id"].astype(str)
    labels = classes.set_index("node_id")["label"].to_dict()
    times = features.set_index("node_id")["timestamp"].to_dict()

    edges["source_label"] = edges["source"].map(labels).fillna(-1).astype(int)
    edges["target_label"] = edges["target"].map(labels).fillna(-1).astype(int)
    # Keep the legacy label column for generic consumers; snapshot construction
    # uses source_label/target_label to build a unique node label map.
    edges["label"] = edges["source_label"]
    edges["timestamp"] = edges["source"].map(times).fillna(0).astype(int)
    # Elliptic does not provide transaction amounts; use a documented placeholder.
    edges["amount"] = 1.0
    return edges[["source", "target", "amount", "timestamp", "label", "source_label", "target_label"]]


def load_generic_transactions(raw_dir: str) -> pd.DataFrame:
    path = Path(raw_dir) / "transactions.csv"
    if not path.exists():
        raise FileNotFoundError("Generic mode expects data/raw/transactions.csv")
    df = pd.read_csv(path)
    cols = [_resolve_col(df, key) for key in ("source", "target", "amount", "timestamp", "label")]
    out = df[cols].copy()
    out.columns = ["source", "target", "amount", "timestamp", "label"]
    return out


def clean_transactions(df: pd.DataFrame, remove_isolated_nodes: bool = False) -> pd.DataFrame:
    required = ["source", "target", "timestamp", "label"]
    out = df.copy().dropna(subset=required)
    out["source"] = out["source"].astype(str)
    out["target"] = out["target"].astype(str)
    out["amount"] = pd.to_numeric(out.get("amount", 0.0), errors="coerce").fillna(0.0)
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0).astype(int)
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(-1).astype(int)
    for col in ("source_label", "target_label"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(-1).astype(int)
    if remove_isolated_nodes:
        nodes = pd.concat([out["source"], out["target"]], ignore_index=True)
        degree = nodes.value_counts()
        keep = set(degree[degree > 1].index)
        out = out[out["source"].isin(keep) & out["target"].isin(keep)].copy()
    return out.sort_values(["timestamp", "source", "target"]).reset_index(drop=True)
