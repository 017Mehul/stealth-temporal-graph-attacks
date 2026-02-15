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
    model.eval()
    with torch.no_grad():
        seq = _to_device_snapshots(snapshots, device)
        logits = model.forward_embeddings(seq)
        probs = torch.sigmoid(logits).cpu().numpy()  # [T, N]
    return probs


def evaluate_sequence_f1(model, snapshots, indices, device):
    probs = predict_dynamic(model, [snapshots[i] for i in indices], device)
    y_true_all, y_prob_all = [], []
    for t, idx in enumerate(indices):
        data = snapshots[idx]
        mask = (data.labeled_mask & (data.y >= 0)).cpu().numpy()
        y = data.y.cpu().numpy().astype(int)
        y_true_all.append(y[mask])
        y_prob_all.append(probs[t][mask])

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return compute_binary_metrics(y_true, y_prob)


def run_low_rate_attack(model, snapshots, cfg, device):
    attack_cfg = cfg["attack"]
    train_idx, val_idx, test_idx = split_indices(
        len(snapshots),
        cfg["training"]["train_ratio"],
        cfg["training"]["val_ratio"],
    )

    clean_metrics = evaluate_sequence_f1(model, snapshots, test_idx, device)

    attacked = [copy.deepcopy(s) for s in snapshots]

    probs_clean = predict_dynamic(model, [attacked[i] for i in test_idx], device)
    first_test = attacked[test_idx[0]]
    y0 = first_test.y.cpu().numpy().astype(int)

    fraud_candidates = np.where(y0 == 1)[0]
    if len(fraud_candidates) == 0:
        raise RuntimeError("No labeled fraud nodes in test split for target selection.")

    # Choose high-risk fraud node: highest predicted fraud probability at first test step.
    first_probs = probs_clean[0]
    target_node = fraud_candidates[np.argmax(first_probs[fraud_candidates])]

    benign_candidates = np.where(y0 == 0)[0]
    if len(benign_candidates) == 0:
        raise RuntimeError("No benign candidates in test split for attack injection.")

    # Benign amount distribution from test snapshots.
    benign_amounts = []
    for idx in test_idx:
        s = attacked[idx]
        if s.edge_attr.size(0) == 0:
            continue
        benign_amounts.extend(s.edge_attr[:, 0].cpu().numpy().tolist())
    if not benign_amounts:
        benign_amounts = [1.0]

    q_low, q_high = np.quantile(np.array(benign_amounts), [0.25, 0.75])

    attack_duration = min(int(attack_cfg["attack_duration"]), len(test_idx))
    budget = int(attack_cfg["attack_budget_per_window"])

    fraud_prob_over_time = []
    injected_edges = 0
    success = False

    for local_t in range(attack_duration):
        snap_idx = test_idx[local_t]
        snap = attacked[snap_idx]

        # Candidate benign node with lowest current fraud score in this window.
        current_probs = predict_dynamic(model, [attacked[i] for i in test_idx[: local_t + 1]], device)[-1]
        benign_scores = current_probs[benign_candidates]
        benign_node = benign_candidates[np.argmin(benign_scores)]

        for _ in range(budget):
            sampled_amt = float(np.random.uniform(q_low, q_high))

            new_edge = torch.tensor([[benign_node], [target_node]], dtype=torch.long)
            ts_val = int(snap.time_values[0].item()) if hasattr(snap, "time_values") and snap.time_values.numel() > 0 else int(local_t)
            new_attr = torch.tensor([[sampled_amt, ts_val]], dtype=torch.float)

            if snap.edge_index.numel() == 0:
                snap.edge_index = new_edge
                snap.edge_attr = new_attr
            else:
                snap.edge_index = torch.cat([snap.edge_index, new_edge], dim=1)
                snap.edge_attr = torch.cat([snap.edge_attr, new_attr], dim=0)
            injected_edges += 1

        seq_probs = predict_dynamic(model, [attacked[i] for i in test_idx[: local_t + 1]], device)
        p_target = float(seq_probs[-1][target_node])
        fraud_prob_over_time.append(p_target)

        if p_target < 0.5:
            success = True

    attacked_metrics = evaluate_sequence_f1(model, attacked, test_idx, device)

    return {
        "target_node_index": int(target_node),
        "clean_metrics": clean_metrics,
        "attacked_metrics": attacked_metrics,
        "attack_success": bool(success),
        "injected_edges": int(injected_edges),
        "fraud_probability_over_time": fraud_prob_over_time,
        "test_indices": test_idx,
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
    plt.title("Fraud Probability vs Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fraud_probability_vs_time.png", dpi=150)
    plt.close()

    clean_f1 = result["clean_metrics"]["f1"]
    attacked_f1 = result["attacked_metrics"]["f1"]
    budgets = [0, result["injected_edges"]]
    f1s = [clean_f1, attacked_f1]

    plt.figure(figsize=(6, 4))
    plt.plot(budgets, f1s, marker="s")
    plt.xlabel("Injection Budget (Total Edges)")
    plt.ylabel("F1")
    plt.title("F1 vs Injection Budget")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "f1_vs_budget.png", dpi=150)
    plt.close()
