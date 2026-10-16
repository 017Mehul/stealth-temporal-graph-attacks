import argparse
import json
from pathlib import Path

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

from slrta.config import ensure_dirs, load_config, set_seed
from slrta.data.io import labeled_node_mask, load_snapshots, require_first_seen_masks, split_indices
from slrta.eval.metrics import compute_binary_metrics
from slrta.eval.thresholds import select_f1_threshold
from slrta.models.static_gcn import StaticGCN


def evaluate(model, snapshots, indices, device, threshold=0.5, newly_observed_only=True):
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for i in indices:
            data = snapshots[i].to(device)
            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits)
            mask = labeled_node_mask(data, newly_observed_only=newly_observed_only)
            if mask.sum() == 0:
                continue
            y_true_all.append(data.y[mask].cpu().numpy().astype(int))
            y_prob_all.append(probs[mask].cpu().numpy())

    if not y_true_all:
        return compute_binary_metrics([], [], threshold=threshold)

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return compute_binary_metrics(y_true, y_prob, threshold=threshold)


def collect_probs_and_labels(model, snapshots, indices, device, newly_observed_only=True):
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for i in indices:
            data = snapshots[i].to(device)
            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits)
            mask = labeled_node_mask(data, newly_observed_only=newly_observed_only)
            if mask.sum() == 0:
                continue
            y_true_all.append(data.y[mask].cpu().numpy().astype(int))
            y_prob_all.append(probs[mask].cpu().numpy())

    if not y_true_all:
        return np.array([], dtype=int), np.array([], dtype=float)

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return y_true, y_prob



def collect_per_snapshot_metrics(model, snapshots, indices, device, threshold, newly_observed_only=True):
    """Return metrics for each snapshot separately, preserving chronological labels."""
    model.eval()
    results = []
    with torch.no_grad():
        for idx in indices:
            data = snapshots[idx].to(device)
            probs = torch.sigmoid(model(data.x, data.edge_index))
            mask = labeled_node_mask(data, newly_observed_only=newly_observed_only)
            y_true = data.y[mask].cpu().numpy().astype(int)
            y_prob = probs[mask].cpu().numpy()
            metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
            metrics["snapshot_index"] = int(idx)
            results.append(metrics)
    return results


def class_distribution(snapshots, indices, newly_observed_only=True):
    labels = []
    for idx in indices:
        data = snapshots[idx]
        mask = labeled_node_mask(data, newly_observed_only=newly_observed_only)
        if mask.any():
            labels.append(data.y[mask].cpu().numpy().astype(int))
    values = np.concatenate(labels) if labels else np.array([], dtype=int)
    return {"negative": int((values == 0).sum()), "positive": int((values == 1).sum())}


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    set_seed(cfg["project"]["seed"])

    requested_device = cfg["project"].get("device", "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in config but is unavailable; set project.device: cpu")
    device = torch.device(requested_device)
    snapshots = load_snapshots(cfg["paths"]["dynamic_dir"])
    require_first_seen_masks(snapshots, cfg.get("data", {}).get("dataset_type", "generic"))
    train_idx, val_idx, test_idx = split_indices(
        len(snapshots),
        cfg["training"]["train_ratio"],
        cfg["training"]["val_ratio"],
    )

    in_dim = snapshots[0].x.size(1)
    model = StaticGCN(in_dim, cfg["training"]["hidden_dim"], cfg["training"]["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    best_val_score = -float("inf")
    best_path = Path(cfg["paths"]["checkpoints_dir"]) / "static_gcn_best.pt"
    if best_path.exists():
        best_path.unlink()
    train_losses = []

    # Compute and print class distributions for train and validation sets
    y_train_all, y_val_all = [], []
    for i in train_idx:
        data = snapshots[i]
        mask = labeled_node_mask(data)
        if mask.sum() == 0:
            continue
        y_train_all.append(data.y[mask].cpu().numpy().astype(int))

    for i in val_idx:
        data = snapshots[i]
        mask = labeled_node_mask(data)
        if mask.sum() == 0:
            continue
        y_val_all.append(data.y[mask].cpu().numpy().astype(int))

    if y_train_all:
        y_train = np.concatenate(y_train_all)
    else:
        y_train = np.array([], dtype=int)
    if y_train.size == 0:
        raise RuntimeError("No labeled training nodes found in the chronological training split.")
    if not y_val_all:
        raise RuntimeError("No labeled validation nodes; cannot select a checkpoint reliably.")

    if y_val_all:
        y_val = np.concatenate(y_val_all)
    else:
        y_val = np.array([], dtype=int)

    print("Train class distribution:", np.bincount(y_train))
    print("Val class distribution:", np.bincount(y_val))

    # Compute pos_weight from training distribution and create loss
    try:
        num_positive = int((y_train == 1).sum())
        num_negative = int((y_train == 0).sum())
    except Exception:
        num_positive = 0
        num_negative = 0

    print("Train positives (illicit):", num_positive)
    print("Train negatives (licit):", num_negative)

    if num_positive > 0:
        pos_weight = torch.tensor(float(num_negative) / float(num_positive), dtype=torch.float, device=device)
    else:
        pos_weight = torch.tensor(1.0, dtype=torch.float, device=device)

    print("Using pos_weight:", float(pos_weight.item()))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in tqdm(range(cfg["training"]["epochs"]), desc="StaticGCN Training"):
        model.train()
        total_loss = 0.0
        for i in train_idx:
            data = snapshots[i].to(device)
            mask = labeled_node_mask(data)
            if mask.sum() == 0:
                continue

            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[mask], data.y[mask])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_losses.append(total_loss / max(1, len(train_idx)))
        val_metrics = evaluate(model, snapshots, val_idx, device)
        checkpoint_score = val_metrics["pr_auc"]
        if checkpoint_score is None:
            checkpoint_score = val_metrics["f1"]
        print(
            f"epoch={epoch + 1}/{cfg['training']['epochs']} "
            f"loss={train_losses[-1]:.6f} val_pr_auc={val_metrics['pr_auc']}"
        )
        if checkpoint_score > best_val_score:
            best_val_score = checkpoint_score
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    # Calibrate on validation predictions only, then freeze the threshold for test.
    y_val_true, y_val_prob = collect_probs_and_labels(model, snapshots, val_idx, device)
    if y_val_prob.size == 0:
        raise RuntimeError("No labeled validation samples; cannot calibrate a threshold.")
    selected_threshold = select_f1_threshold(y_val_true, y_val_prob)
    val_metrics = evaluate(model, snapshots, val_idx, device, threshold=selected_threshold)
    test_metrics = evaluate(model, snapshots, test_idx, device, threshold=selected_threshold)
    all_active_test_metrics = evaluate(
        model, snapshots, test_idx, device,
        threshold=selected_threshold, newly_observed_only=False,
    )
    if test_metrics["n_samples"] == 0:
        raise RuntimeError("No labeled test nodes; refusing to report empty-split metrics.")
    print(f"Validation-selected threshold: {selected_threshold:.6f}")

    test_per_snapshot = collect_per_snapshot_metrics(
        model, snapshots, test_idx, device, threshold=selected_threshold
    )
    dataset_provenance = {
        "dataset_type": cfg.get("data", {}).get("dataset_type"),
        "raw_dir": cfg["paths"].get("raw_dir"),
        "dynamic_dir": cfg["paths"].get("dynamic_dir"),
        "num_snapshots": len(snapshots),
    }
    manifest_path = Path(cfg["paths"]["processed_csv"]).parent / "dataset_provenance.json"
    if manifest_path.exists():
        dataset_provenance["source_manifest"] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    out = {
        "model": "StaticGCN",
        "config": cfg,
        "dataset_provenance": dataset_provenance,
        "checkpoint_selection": "maximum validation PR-AUC; falls back to validation F1 if PR-AUC is undefined",
        "evaluation_cohort": "newly observed labeled nodes only; nodes are supervised/evaluated once at first appearance",
        "test_per_snapshot_metrics": test_per_snapshot,
        "all_active_test_metrics_secondary": all_active_test_metrics,
        "seed": int(cfg["project"]["seed"]),
        "selected_threshold": selected_threshold,
        "threshold_selection": "maximum validation F1; threshold frozen before test evaluation",
        "class_distribution": {
            "train": {"negative": int((y_train == 0).sum()), "positive": int((y_train == 1).sum())},
            "validation": {"negative": int((y_val_true == 0).sum()), "positive": int((y_val_true == 1).sum())},
            "test": class_distribution(snapshots, test_idx),
            "test_all_active_secondary": class_distribution(snapshots, test_idx, newly_observed_only=False),
        },
        "split": {"train": train_idx, "val": val_idx, "test": test_idx},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_path),
        "train_loss_curve": train_losses,
    }

    out_path = Path(cfg["paths"]["metrics_dir"]) / "baseline_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved baseline metrics: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
