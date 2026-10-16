import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from slrta.config import ensure_dirs, load_config, set_seed
from slrta.data.clean import clean_transactions, load_elliptic, load_generic_transactions
from slrta.data.snapshots import build_dynamic_snapshots



def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(cfg):
    dataset_type = cfg["data"]["dataset_type"].lower()
    raw = Path(cfg["paths"]["raw_dir"])
    if dataset_type == "elliptic":
        required = [
            "elliptic_txs_edgelist.csv",
            "elliptic_txs_classes.csv",
            "elliptic_txs_features.csv",
        ]
        roots = [raw, raw / "elliptic_bitcoin_dataset"]
        root = next((candidate for candidate in roots if all((candidate / name).exists() for name in required)), None)
        if root is None:
            raise FileNotFoundError("Cannot build provenance manifest: Elliptic input files are missing.")
        return [root / name for name in required]
    generic = raw / "transactions.csv"
    if not generic.exists():
        raise FileNotFoundError(f"Cannot build provenance manifest: missing {generic}")
    return [generic]


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    set_seed(cfg["project"]["seed"])

    dataset_type = cfg["data"]["dataset_type"].lower()
    raw_dir = cfg["paths"]["raw_dir"]

    if dataset_type == "elliptic":
        df = load_elliptic(raw_dir)
    elif dataset_type == "generic":
        df = load_generic_transactions(raw_dir)
    else:
        raise ValueError("dataset_type must be one of: elliptic, generic")

    clean_df = clean_transactions(df, remove_isolated_nodes=cfg["data"].get("remove_isolated_nodes", False))
    clean_df.to_csv(cfg["paths"]["processed_csv"], index=False)

    source_manifest = []
    for source_path in _source_files(cfg):
        source_manifest.append({
            "file_name": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        })

    build_dynamic_snapshots(
        clean_df,
        cfg["paths"]["dynamic_dir"],
        window_size=cfg["data"]["window_size"],
        normalize_features=cfg["data"].get("normalize_features", True),
        train_ratio=cfg["training"]["train_ratio"],
    )

    manifest = {
        "dataset_type": dataset_type,
        "source_files": source_manifest,
        "cleaned_rows": int(len(clean_df)),
        "unique_timestamps": int(clean_df["timestamp"].nunique()),
        "num_snapshots": int(clean_df["timestamp"].nunique() // max(1, int(cfg["data"]["window_size"])) + bool(clean_df["timestamp"].nunique() % max(1, int(cfg["data"]["window_size"])))),
        "preprocessing": {
            "window_size": int(cfg["data"]["window_size"]),
            "normalize_features": bool(cfg["data"].get("normalize_features", True)),
            "remove_isolated_nodes": bool(cfg["data"].get("remove_isolated_nodes", False)),
            "train_ratio_for_scaler": float(cfg["training"]["train_ratio"]),
        },
    }
    manifest_path = Path(cfg["paths"]["processed_csv"]).parent / "dataset_provenance.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved cleaned transactions: {cfg['paths']['processed_csv']}")
    print(f"Saved dynamic graphs in: {cfg['paths']['dynamic_dir']}")
    print(f"Saved dataset provenance: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
