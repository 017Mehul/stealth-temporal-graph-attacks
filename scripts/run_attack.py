import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from slrta.attack.low_rate import run_low_rate_attack, save_attack_plots
from slrta.config import ensure_dirs, load_config
from slrta.data.io import load_snapshots
from slrta.models.dynamic_gcn_gru import DynamicGCNGRU


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    device = torch.device(cfg["project"].get("device", "cpu"))
    snapshots = load_snapshots(cfg["paths"]["dynamic_dir"])

    in_dim = snapshots[0].x.size(1)
    model = DynamicGCNGRU(in_dim, cfg["training"]["hidden_dim"], cfg["training"]["dropout"]).to(device)

    ckpt = Path(cfg["paths"]["checkpoints_dir"]) / "dynamic_gcn_gru_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt}. Run scripts/train_dynamic.py first.")

    model.load_state_dict(torch.load(ckpt, map_location=device))

    result = run_low_rate_attack(model, snapshots, cfg, device)
    save_attack_plots(result, cfg["paths"]["plots_dir"])

    out_path = Path(cfg["paths"]["metrics_dir"]) / "attack_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved attack metrics: {out_path}")
