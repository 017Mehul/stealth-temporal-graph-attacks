import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from slrta.attack.low_rate import run_low_rate_attack, save_attack_plots
from slrta.config import ensure_dirs, load_config
from slrta.data.io import load_snapshots, require_first_seen_masks
from slrta.models.dynamic_gcn_gru import MODEL_VERSION, DynamicGCNGRU


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    requested_device = cfg["project"].get("device", "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in config but is unavailable; set project.device: cpu")
    device = torch.device(requested_device)
    snapshots = load_snapshots(cfg["paths"]["dynamic_dir"])
    require_first_seen_masks(snapshots, cfg.get("data", {}).get("dataset_type", "generic"))

    in_dim = snapshots[0].x.size(1)
    model = DynamicGCNGRU(in_dim, cfg["training"]["hidden_dim"], cfg["training"]["dropout"]).to(device)

    ckpt = Path(cfg["paths"]["checkpoints_dir"]) / "dynamic_gcn_gru_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt}. Run scripts/train_dynamic.py first.")

    threshold_path = Path(cfg["paths"]["metrics_dir"]) / "dynamic_results.json"
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Missing {threshold_path}. Run scripts/train_dynamic.py before attack evaluation."
        )
    threshold_results = json.loads(threshold_path.read_text(encoding="utf-8"))
    if cfg.get("data", {}).get("dataset_type") == "elliptic":
        manifest_path = Path(cfg["paths"]["processed_csv"]).parent / "dataset_provenance.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Rerun preprocessing before attack evaluation."
            )
        current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trained_manifest = threshold_results.get("dataset_provenance", {}).get("source_manifest")
        if trained_manifest != current_manifest:
            raise RuntimeError(
                "Dataset provenance differs from the one used for training. "
                "Retrain on the current preprocessed dataset before attack evaluation."
            )
    if threshold_results.get("model_version") != MODEL_VERSION:
        raise RuntimeError(
            "Dynamic model results were produced by a different architecture version. "
            "Retrain the temporal model before running the attack."
        )
    actual_hash = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if threshold_results.get("checkpoint_sha256") != actual_hash:
        raise RuntimeError(
            "The temporal checkpoint does not match dynamic_results.json. "
            "Retrain the model or restore the matching checkpoint/metrics pair."
        )
    cfg.setdefault("attack", {})["decision_threshold"] = float(
        threshold_results["selected_threshold"]
    )
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    result = run_low_rate_attack(model, snapshots, cfg, device)
    save_attack_plots(result, cfg["paths"]["plots_dir"])

    out_path = Path(cfg["paths"]["metrics_dir"]) / "attack_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved attack metrics: {out_path}")
