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
        "Elliptic files missing. Expected files in either data/raw/ or "
        "data/raw/elliptic_bitcoin_dataset/: elliptic_txs_edgelist.csv, "
        "elliptic_txs_classes.csv, elliptic_txs_features.csv."
    )


def load_elliptic(raw_dir: str) -> pd.DataFrame:
    raw = _resolve_elliptic_root(raw_dir)
    edges_path = raw / "elliptic_txs_edgelist.csv"
    classes_path = raw / "elliptic_txs_classes.csv"
    features_path = raw / "elliptic_txs_features.csv"

    edges = pd.read_csv(edges_path)
    classes = pd.read_csv(classes_path)
    feats = pd.read_csv(features_path, header=None)

    # Elliptic features format: txId, time_step, ...
    feats = feats.rename(columns={0: "txId", 1: "timestamp"})[["txId", "timestamp"]]
    classes = classes.rename(columns={"txId": "node_id", "class": "label"})

    label_map = {"1": 1, "2": 0, "unknown": -1, 1: 1, 2: 0}
    classes["label"] = classes["label"].map(label_map).fillna(-1).astype(int)

    edges = edges.rename(columns={"txId1": "source", "txId2": "target"})
    edges = edges.merge(feats.rename(columns={"txId": "source"}), on="source", how="left")
    edges = edges.merge(classes.rename(columns={"node_id": "source", "label": "source_label"}), on="source", how="left")
    edges = edges.merge(classes.rename(columns={"node_id": "target", "label": "target_label"}), on="target", how="left")

    # Elliptic does not include amount; use 1.0 as placeholder edge amount.
    edges["amount"] = 1.0
    edges["label"] = edges["source_label"].fillna(-1).astype(int)
    edges["timestamp"] = edges["timestamp"].fillna(0).astype(int)

    return edges[["source", "target", "amount", "timestamp", "label"]]


def load_generic_transactions(raw_dir: str) -> pd.DataFrame:
    path = Path(raw_dir) / "transactions.csv"
    if not path.exists():
        raise FileNotFoundError("Generic mode expects data/raw/transactions.csv")

    df = pd.read_csv(path)
    src = _resolve_col(df, "source")
    dst = _resolve_col(df, "target")
    amt = _resolve_col(df, "amount")
    ts = _resolve_col(df, "timestamp")
    lb = _resolve_col(df, "label")

    out = df[[src, dst, amt, ts, lb]].copy()
    out.columns = ["source", "target", "amount", "timestamp", "label"]
    return out


def clean_transactions(df: pd.DataFrame, remove_isolated_nodes: bool = False) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["source", "target", "timestamp", "label"]).copy()
    out["source"] = out["source"].astype(str)
    out["target"] = out["target"].astype(str)
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0).astype(int)
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(-1).astype(int)

    if remove_isolated_nodes:
        nodes = pd.concat([out["source"], out["target"]], ignore_index=True)
        deg = nodes.value_counts()
        keep = set(deg[deg > 1].index)
        out = out[out["source"].isin(keep) & out["target"].isin(keep)].copy()

    out = out.sort_values(["timestamp", "source", "target"]).reset_index(drop=True)
    return out
