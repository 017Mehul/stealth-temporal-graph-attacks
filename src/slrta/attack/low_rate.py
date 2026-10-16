import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from slrta.data.io import split_indices
from slrta.eval.metrics import compute_binary_metrics


def _to_device_snapshots(snapshots, device):
    return [s.to(device) for s in snapshots]


def predict_dynamic(model, snapshots, device):
    """Predict snapshots in order; caller must pass the full history prefix."""
    model.eval()
    with torch.no_grad():
        seq = _to_device_snapshots(snapshots, device)
        logits = model.forward_embeddings(seq)
        return torch.sigmoid(logits).cpu().numpy()


def _predict_indices(model, snapshots, indices, device):
    if not indices:
        raise ValueError("Evaluation indices are empty")
    probs_prefix = predict_dynamic(model, snapshots[: max(indices) + 1], device)
    return probs_prefix[np.asarray(indices, dtype=int)]


def evaluate_sequence_f1(model, snapshots, indices, device):
    probs = _predict_indices(model, snapshots, indices, device)
    y_true_all, y_prob_all = [], []
    for t, idx in enumerate(indices):
        data = snapshots[idx]
        mask = (data.labeled_mask & (data.y >= 0)).cpu().numpy()
        if mask.any():
            y_true_all.append(data.y.cpu().numpy()[mask].astype(int))
            y_prob_all.append(probs[t][mask])
    if not y_true_all:
        return compute_binary_metrics([], [])
    return compute_binary_metrics(np.concatenate(y_true_all), np.concatenate(y_prob_all))


def _append_edge_to_future_snapshots(snapshots, start_index, source, target, amount, timestamp):
    """Append one intervention to its time step and every later cumulative snapshot."""
    if start_index < 0 or start_index >= len(snapshots):
        raise IndexError("start_index is outside the snapshot sequence")
    new_edge = torch.tensor([[int(source)], [int(target)]], dtype=torch.long)
    new_attr = torch.tensor([[float(amount), float(timestamp)]], dtype=torch.float32)
    for future in snapshots[start_index:]:
        future.edge_index = torch.cat([future.edge_index, new_edge], dim=1)
        future.edge_attr = torch.cat([future.edge_attr, new_attr], dim=0)


def _recompute_cumulative_features(snapshots, start_index=0):
    """Recompute structural features for all cumulative snapshots after intervention."""
    if not snapshots:
        return
    num_nodes = int(snapshots[0].x.size(0))
    previous_degree = np.zeros(num_nodes, dtype=np.float32)
    for index, snap in enumerate(snapshots):
        edge_index = snap.edge_index.cpu().numpy()
        edge_attr = snap.edge_attr.cpu().numpy()
        src = edge_index[0] if edge_index.size else np.array([], dtype=np.int64)
        dst = edge_index[1] if edge_index.size else np.array([], dtype=np.int64)
        amount = edge_attr[:, 0] if edge_attr.size else np.array([], dtype=np.float32)
        timestamp = edge_attr[:, 1] if edge_attr.size else np.array([], dtype=np.float32)
        indeg = np.bincount(dst, minlength=num_nodes).astype(np.float32)
        outdeg = np.bincount(src, minlength=num_nodes).astype(np.float32)
        total = np.zeros(num_nodes, dtype=np.float32)
        counts = np.zeros(num_nodes, dtype=np.float32)
        time_sets = [set() for _ in range(num_nodes)]
        for s, d, a, ts in zip(src, dst, amount, timestamp):
            total[s] += a
            total[d] += a
            counts[s] += 1
            counts[d] += 1
            time_sets[s].add(int(ts))
            time_sets[d].add(int(ts))
        avg = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
        degree = indeg + outdeg
        # Cumulative frequency uses the number of distinct elapsed event times,
        # matching the preprocessing pipeline rather than the latest window width.
        frequency = counts / max(1, int(np.unique(timestamp).size))
        tac = np.asarray([len(x) for x in time_sets], dtype=np.float32)
        base = np.column_stack([indeg, outdeg, total, avg, frequency, tac, degree, np.log1p(degree)])
        # The temporal label-derived feature is intentionally held at zero.
        cumulative_count = degree
        growth = degree - previous_degree
        temporal = np.column_stack([cumulative_count, growth, np.zeros(num_nodes, dtype=np.float32)])
        snap.x = torch.tensor(np.concatenate([base, temporal], axis=1), dtype=torch.float32)
        previous_degree = degree


def run_low_rate_attack(model, snapshots, cfg, device):
    attack_cfg = cfg["attack"]
    _, _, test_idx = split_indices(
        len(snapshots), cfg["training"]["train_ratio"], cfg["training"]["val_ratio"]
    )
    if not test_idx:
        raise ValueError("No test snapshots available")
    clean_metrics = evaluate_sequence_f1(model, snapshots, test_idx, device)
    attacked = [copy.deepcopy(s) for s in snapshots]

    clean_probs = _predict_indices(model, attacked, test_idx, device)
    first_test = attacked[test_idx[0]]
    y0 = first_test.y.cpu().numpy().astype(int)
    fraud_candidates = np.where(y0 == 1)[0]
    benign_candidates = np.where(y0 == 0)[0]
    if not len(fraud_candidates):
        raise RuntimeError("No labeled illicit nodes in the first test snapshot.")
    if not len(benign_candidates):
        raise RuntimeError("No labeled licit candidates in the first test snapshot.")

    # Target selection uses the first test labels as a targeted benchmark oracle.
    # This is a white-box benchmark assumption, not a deployable attacker model.
    target_node = int(fraud_candidates[np.argmax(clean_probs[0, fraud_candidates])])

    # Use training-period amounts only; never estimate attack parameters from test edges.
    train_idx, _, _ = split_indices(
        len(snapshots), cfg["training"]["train_ratio"], cfg["training"]["val_ratio"]
    )
    train_amounts = []
    for idx in train_idx:
        edge_attr = attacked[idx].edge_attr
        if edge_attr.numel():
            train_amounts.extend(edge_attr[:, 0].cpu().numpy().tolist())
    if not train_amounts:
        train_amounts = [1.0]
    q_low, q_high = np.quantile(np.asarray(train_amounts), [0.25, 0.75])

    attack_duration = min(int(attack_cfg["attack_duration"]), len(test_idx))
    budget = int(attack_cfg["attack_budget_per_window"])
    if attack_duration < 1 or budget < 0:
        raise ValueError("attack_duration must be positive and attack_budget_per_window nonnegative")

    fraud_prob_over_time = []
    injected_edges = 0
    successful_steps = 0

    for local_t in range(attack_duration):
        snap_idx = test_idx[local_t]
        # Re-score with complete history through the current time step.
        current_probs = _predict_indices(model, attacked, [snap_idx], device)[0]
        benign_node = int(benign_candidates[np.argmin(current_probs[benign_candidates])])
        snap = attacked[snap_idx]
        ts_val = int(snap.time_values[-1].item()) if hasattr(snap, "time_values") and snap.time_values.numel() else int(local_t)

        for _ in range(budget):
            sampled_amt = float(np.random.uniform(q_low, q_high)) if q_high > q_low else float(q_low)
            # Snapshots are cumulative: an intervention persists into every future snapshot.
            _append_edge_to_future_snapshots(
                attacked, snap_idx, benign_node, target_node, sampled_amt, ts_val
            )
            injected_edges += 1
        _recompute_cumulative_features(attacked, snap_idx)
        current_prob = _predict_indices(model, attacked, [snap_idx], device)[0]
        p_target = float(current_prob[target_node])
        fraud_prob_over_time.append(p_target)
        successful_steps += int(p_target < 0.5)

    attacked_metrics = evaluate_sequence_f1(model, attacked, test_idx, device)
    return {
        "target_node_index": int(target_node),
        "clean_metrics": clean_metrics,
        "attacked_metrics": attacked_metrics,
        "attack_success": bool(fraud_prob_over_time and fraud_prob_over_time[-1] < 0.5),
        "attack_success_rate_over_steps": float(successful_steps / max(1, len(fraud_prob_over_time))),
        "injected_edges": int(injected_edges),
        "fraud_probability_over_time": fraud_prob_over_time,
        "test_indices": test_idx,
        "attacker_knowledge": "target-label oracle for benchmark analysis; parameters estimated from training snapshots only",
    }


def save_attack_plots(result: dict, plots_dir: str):
    out = Path(plots_dir)
    out.mkdir(parents=True, exist_ok=True)
    probs = result["fraud_probability_over_time"]
    x = list(range(1, len(probs) + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(x, probs, marker="o")
    plt.xlabel("Attack Time Step")
    plt.ylabel("Target Fraud Probability")
    plt.title("Target Probability Across Attack Steps")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fraud_probability_vs_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Clean", "Attacked"], [result["clean_metrics"]["f1"], result["attacked_metrics"]["f1"]])
    plt.ylabel("F1")
    plt.title("Clean vs Attacked F1")
    plt.tight_layout()
    plt.savefig(out / "clean_vs_attacked_f1.png", dpi=150)
    plt.close()
