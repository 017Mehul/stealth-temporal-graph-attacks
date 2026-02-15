import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from slrta.config import ensure_dirs, load_config, set_seed
from slrta.data.clean import clean_transactions, load_elliptic, load_generic_transactions
from slrta.data.snapshots import build_dynamic_snapshots


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

    build_dynamic_snapshots(
        clean_df,
        cfg["paths"]["dynamic_dir"],
        window_size=cfg["data"]["window_size"],
        normalize_features=cfg["data"].get("normalize_features", True),
    )

    print(f"Saved cleaned transactions: {cfg['paths']['processed_csv']}")
    print(f"Saved dynamic graphs in: {cfg['paths']['dynamic_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
