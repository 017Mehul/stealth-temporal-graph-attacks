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
from slrta.data.io import load_snapshots, split_indices
from slrta.eval.metrics import compute_binary_metrics
from slrta.models.static_gcn import StaticGCN


def evaluate(model, snapshots, indices, device):
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for i in indices:
            data = snapshots[i].to(device)
            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits)
            mask = data.labeled_mask & (data.y >= 0)
            if mask.sum() == 0:
                continue
            y_true_all.append(data.y[mask].cpu().numpy().astype(int))
            y_prob_all.append(probs[mask].cpu().numpy())

    if not y_true_all:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "roc_auc": 0.5}

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return compute_binary_metrics(y_true, y_prob)


def collect_probs_and_labels(model, snapshots, indices, device):
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for i in indices:
            data = snapshots[i].to(device)
            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits)
            mask = data.labeled_mask & (data.y >= 0)
            if mask.sum() == 0:
                continue
            y_true_all.append(data.y[mask].cpu().numpy().astype(int))
            y_prob_all.append(probs[mask].cpu().numpy())

    if not y_true_all:
        return np.array([], dtype=int), np.array([], dtype=float)

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return y_true, y_prob


def main(config_path: str):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    set_seed(cfg["project"]["seed"])

    device = torch.device(cfg["project"].get("device", "cpu"))
    snapshots = load_snapshots(cfg["paths"]["dynamic_dir"])
    train_idx, val_idx, test_idx = split_indices(
        len(snapshots),
        cfg["training"]["train_ratio"],
        cfg["training"]["val_ratio"],
    )

    in_dim = snapshots[0].x.size(1)
    model = StaticGCN(in_dim, cfg["training"]["hidden_dim"], cfg["training"]["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    # criterion will be created after computing `y_train` so we can derive pos_weight
    criterion = None

    best_val_f1 = -1.0
    best_path = Path(cfg["paths"]["checkpoints_dir"]) / "static_gcn_best.pt"
    train_losses = []

    # Compute and print class distributions for train and validation sets
    y_train_all, y_val_all = [], []
    for i in train_idx:
        data = snapshots[i]
        mask = data.labeled_mask & (data.y >= 0)
        if mask.sum() == 0:
            continue
        y_train_all.append(data.y[mask].cpu().numpy().astype(int))

    for i in val_idx:
        data = snapshots[i]
        mask = data.labeled_mask & (data.y >= 0)
        if mask.sum() == 0:
            continue
        y_val_all.append(data.y[mask].cpu().numpy().astype(int))

    if y_train_all:
        y_train = np.concatenate(y_train_all)
    else:
        y_train = np.array([], dtype=int)

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
            mask = data.labeled_mask & (data.y >= 0)
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
        print(
            f"epoch={epoch + 1}/{cfg['training']['epochs']} "
            f"loss={train_losses[-1]:.6f} val_f1={val_metrics['f1']:.4f}"
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_metrics = evaluate(model, snapshots, val_idx, device)
    test_metrics = evaluate(model, snapshots, test_idx, device)

    # Collect probs and labels for validation to inspect prediction distribution
    y_val_true, y_val_prob = collect_probs_and_labels(model, snapshots, val_idx, device)
    if y_val_prob.size:
        print("Validation probs — mean:{:.6f} max:{:.6f} min:{:.6f}".format(
            float(y_val_prob.mean()), float(y_val_prob.max()), float(y_val_prob.min())
        ))
        preds_05 = (y_val_prob >= 0.5).sum()
        print("Predicted positives at 0.5:", int(preds_05))

        # Test several thresholds
        for thr in [0.5, 0.3, 0.2, 0.1, 0.05]:
            metrics_thr = compute_binary_metrics(y_val_true, y_val_prob, threshold=thr)
            print(f"Threshold={thr:.2f} -> f1={metrics_thr['f1']:.4f} prec={metrics_thr['precision']:.4f} rec={metrics_thr['recall']:.4f}")
    else:
        print("No labeled validation samples to compute probabilities.")

    out = {
        "model": "StaticGCN",
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
