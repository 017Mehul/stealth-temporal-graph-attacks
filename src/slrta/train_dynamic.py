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
from slrta.models.dynamic_gcn_gru import DynamicGCNGRU


def eval_dynamic(model, snapshots, indices, device):
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        seq = [snapshots[i].to(device) for i in indices]
        logits_tn = model.forward_embeddings(seq)
        probs_tn = torch.sigmoid(logits_tn)

        for local_t, data in enumerate(seq):
            mask = data.labeled_mask & (data.y >= 0)
            if mask.sum() == 0:
                continue
            y_true_all.append(data.y[mask].cpu().numpy().astype(int))
            y_prob_all.append(probs_tn[local_t][mask].cpu().numpy())

    if not y_true_all:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "roc_auc": 0.5}

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return compute_binary_metrics(y_true, y_prob)


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
    model = DynamicGCNGRU(in_dim, cfg["training"]["hidden_dim"], cfg["training"]["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()
    truncate_bptt = int(cfg["training"].get("truncate_bptt", 4))

    best_val_f1 = -1.0
    best_path = Path(cfg["paths"]["checkpoints_dir"]) / "dynamic_gcn_gru_best.pt"
    train_losses = []

    train_seq = [snapshots[i].to(device) for i in train_idx]

    for epoch in tqdm(range(cfg["training"]["epochs"]), desc="DynamicGCNGRU Training"):
        model.train()
        optimizer.zero_grad()
        h = None
        chunk_losses = []
        update_steps = 0
        epoch_loss_sum = 0.0

        for local_t, data in enumerate(train_seq):
            logits_t, h = model.forward_step(data, h)
            mask = data.labeled_mask & (data.y >= 0)
            if mask.sum() > 0:
                chunk_losses.append(criterion(logits_t[mask], data.y[mask]))

            should_update = ((local_t + 1) % truncate_bptt == 0) or ((local_t + 1) == len(train_seq))
            if should_update and chunk_losses:
                loss = torch.stack(chunk_losses).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss_sum += float(loss.item())
                update_steps += 1
                chunk_losses = []
                if h is not None:
                    h = h.detach()

        if update_steps == 0:
            train_losses.append(0.0)
            continue
        train_losses.append(epoch_loss_sum / update_steps)

        val_metrics = eval_dynamic(model, snapshots, val_idx, device)
        print(
            f"epoch={epoch + 1}/{cfg['training']['epochs']} "
            f"loss={train_losses[-1]:.6f} val_f1={val_metrics['f1']:.4f}"
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_metrics = eval_dynamic(model, snapshots, val_idx, device)
    test_metrics = eval_dynamic(model, snapshots, test_idx, device)

    out = {
        "model": "DynamicGCNGRU",
        "split": {"train": train_idx, "val": val_idx, "test": test_idx},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_path),
        "train_loss_curve": train_losses,
    }

    out_path = Path(cfg["paths"]["metrics_dir"]) / "dynamic_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved dynamic metrics: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
