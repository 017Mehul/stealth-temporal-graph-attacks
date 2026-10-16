import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slrta.attack.low_rate import run_low_rate_attack, save_attack_plots
from slrta.config import load_config
from slrta.data.io import load_snapshots
from slrta.models.dynamic_gcn_gru import DynamicGCNGRU
from slrta.train_baseline import main as train_baseline
from slrta.train_dynamic import main as train_dynamic


METRICS = ("f1", "precision", "recall", "accuracy", "roc_auc", "pr_auc")


def _mean_std(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not finite:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "n": len(finite),
    }


def run_seed(base_config_path, seed, root_out):
    cfg = load_config(base_config_path)
    cfg["project"]["seed"] = int(seed)
    run_root = root_out / f"seed_{seed}"
    cfg["paths"]["checkpoints_dir"] = str(run_root / "checkpoints")
    cfg["paths"]["metrics_dir"] = str(run_root / "metrics")
    cfg["paths"]["plots_dir"] = str(run_root / "plots")
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = run_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    train_baseline(str(config_path))
    train_dynamic(str(config_path))

    snapshots = load_snapshots(cfg["paths"]["dynamic_dir"])
    model = DynamicGCNGRU(
        snapshots[0].x.size(1),
        cfg["training"]["hidden_dim"],
        cfg["training"]["dropout"],
    ).to(torch.device(cfg["project"].get("device", "cpu")))
    checkpoint = Path(cfg["paths"]["checkpoints_dir"]) / "dynamic_gcn_gru_best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=cfg["project"].get("device", "cpu"), weights_only=True))
    result = run_low_rate_attack(model, snapshots, cfg, torch.device(cfg["project"].get("device", "cpu")))
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    (metrics_dir / "attack_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_attack_plots(result, cfg["paths"]["plots_dir"])
    return {
        "seed": int(seed),
        "baseline": json.loads((metrics_dir / "baseline_results.json").read_text(encoding="utf-8")),
        "dynamic": json.loads((metrics_dir / "dynamic_results.json").read_text(encoding="utf-8")),
        "attack": result,
        "run_dir": str(run_root),
    }


def summarize(runs):
    summary = {"seeds": [run["seed"] for run in runs], "n_seeds": len(runs), "models": {}, "attack": {}}
    for model_name in ("baseline", "dynamic"):
        summary["models"][model_name] = {}
        for metric in METRICS:
            summary["models"][model_name][metric] = _mean_std(
                run[model_name]["test_metrics"].get(metric) for run in runs
            )
    summary["attack"]["success_rate_over_steps"] = _mean_std(
        run["attack"].get("attack_success_rate_over_steps") for run in runs
    )
    summary["attack"]["injected_edges"] = _mean_std(
        run["attack"].get("injected_edges") for run in runs
    )
    summary["attack"]["clean_f1"] = _mean_std(
        run["attack"]["clean_metrics"].get("f1") for run in runs
    )
    summary["attack"]["attacked_f1"] = _mean_std(
        run["attack"]["attacked_metrics"].get("f1") for run in runs
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run reproducible multi-seed Elliptic experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated integer seeds.")
    parser.add_argument("--output", default="outputs/multiseed")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain at least one unique integer seed")
    root_out = Path(args.output)
    root_out.mkdir(parents=True, exist_ok=True)
    runs = [run_seed(args.config, seed, root_out) for seed in seeds]
    summary = summarize(runs)
    (root_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved multi-seed summary: {root_out / 'summary.json'}")


if __name__ == "__main__":
    main()
