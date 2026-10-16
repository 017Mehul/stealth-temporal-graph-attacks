import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from slrta.data.io import split_indices
from slrta.eval.metrics import compute_binary_metrics


def _to_device_snapshots(snapshots, device):
    # Data.to() may mutate; preserve the caller's clean snapshots.
    return [copy.deepcopy(s).to(device) for s in snapshots]


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


def _labels_and_probs(model, snapshots, indices, device):
    probs = _predict_indices(model, snapshots, indices, device)
    labels, predictions = [], []
    for local_idx, snapshot_idx in enumerate(indices):
        data = snapshots[snapshot_idx]
        mask = (data.labeled_mask & (data.y >= 0)).cpu().numpy()
        if mask.any():
            labels.append(data.y.cpu().numpy()[mask].astype(int))
            predictions.append(probs[local_idx][mask])
    if not labels:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(labels), np.concatenate(predictions)


def evaluate_sequence_f1(model, snapshots, indices, device, threshold=0.5):
    y_true, y_prob = _labels_and_probs(model, snapshots, indices, device)
    return compute_binary_metrics(y_true, y_prob, threshold=threshold)


def evaluate_per_snapshot(model, snapshots, indices, device, threshold=0.5):
    probs = _predict_indices(model, snapshots, indices, device)
    results = []
    for local_idx, snapshot_idx in enumerate(indices):
        data = snapshots[snapshot_idx]
        mask = (data.labeled_mask & (data.y >= 0)).cpu().numpy()
        y_true = data.y.cpu().numpy()[mask].astype(int)
        y_prob = probs[local_idx][mask]
        metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        metrics["snapshot_index"] = int(snapshot_idx)
        results.append(metrics)
    return results


def _degree_vector(snapshot):
    edge_index = snapshot.edge_index.detach().cpu().numpy()
    n = int(snapshot.x.size(0))
    if edge_index.size == 0:
        return np.zeros(n, dtype=np.int64)
    return (
        np.bincount(edge_index[0], minlength=n)
        + np.bincount(edge_index[1], minlength=n)
    ).astype(np.int64)


def _js_divergence(p, q):
    """Jensen-Shannon divergence in bits for two nonnegative histograms."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _degree_distribution_js(clean_snapshot, candidate_snapshot):
    clean_degree = _degree_vector(clean_snapshot)
    candidate_degree = _degree_vector(candidate_snapshot)
    max_degree = int(max(clean_degree.max(initial=0), candidate_degree.max(initial=0)))
    clean_hist = np.bincount(clean_degree, minlength=max_degree + 1)
    candidate_hist = np.bincount(candidate_degree, minlength=max_degree + 1)
    return _js_divergence(clean_hist, candidate_hist)


def _edge_set(snapshot):
    edges = snapshot.edge_index.detach().cpu().numpy()
    if not edges.size:
        return set()
    return set(zip(edges[0].tolist(), edges[1].tolist()))


def _candidate_passes_constraints(
    clean_snapshots,
    attacked_snapshots,
    start_index,
    source,
    target,
    degree_increments,
    max_degree_increase,
    max_js_divergence,
):
    """Check validity and degree-distribution stealth before mutating snapshots."""
    if source == target:
        return False, 0.0, "self_loop"
    if degree_increments.get(source, 0) >= max_degree_increase:
        return False, 0.0, "source_degree_cap"
    if degree_increments.get(target, 0) >= max_degree_increase:
        return False, 0.0, "target_degree_cap"

    pair = (int(source), int(target))
    if any(pair in _edge_set(s) for s in attacked_snapshots[start_index:]):
        return False, 0.0, "duplicate_edge"

    max_observed_js = 0.0
    for idx in range(start_index, len(attacked_snapshots)):
        snap = attacked_snapshots[idx]
        active = getattr(snap, "active_mask", None)
        if active is not None:
            active = active.detach().cpu()
            if not bool(active[source]) or not bool(active[target]):
                return False, max_observed_js, "inactive_endpoint"
        candidate = copy.deepcopy(snap)
        edge = torch.tensor([[int(source)], [int(target)]], dtype=torch.long)
        candidate.edge_index = torch.cat([candidate.edge_index.cpu(), edge], dim=1)
        divergence = _degree_distribution_js(clean_snapshots[idx], candidate)
        max_observed_js = max(max_observed_js, divergence)
        if divergence > max_js_divergence:
            return False, max_observed_js, "degree_distribution_js"
    return True, max_observed_js, "accepted"


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
    for snap in snapshots:
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
        frequency = counts / max(1, int(np.unique(timestamp).size))
        tac = np.asarray([len(x) for x in time_sets], dtype=np.float32)
        base = np.column_stack([indeg, outdeg, total, avg, frequency, tac, degree, np.log1p(degree)])
        temporal = np.column_stack([degree, degree - previous_degree, np.zeros(num_nodes, dtype=np.float32)])
        raw_features = torch.tensor(np.concatenate([base, temporal], axis=1), dtype=torch.float32)
        feature_mean = getattr(snap, "feature_mean", None)
        feature_std = getattr(snap, "feature_std", None)
        if feature_mean is not None and feature_std is not None:
            raw_features = (raw_features - feature_mean.cpu()) / feature_std.cpu()
        active_mask = getattr(snap, "active_mask", None)
        if active_mask is not None:
            raw_features[~active_mask.cpu()] = 0.0
        snap.x = raw_features
        previous_degree = degree


def run_low_rate_attack(model, snapshots, cfg, device):
    attack_cfg = cfg["attack"]
    _, _, test_idx = split_indices(
        len(snapshots), cfg["training"]["train_ratio"], cfg["training"]["val_ratio"]
    )
    if not test_idx:
        raise ValueError("No test snapshots available")
    decision_threshold = float(attack_cfg.get("decision_threshold", 0.5))
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("attack.decision_threshold must be between 0 and 1")
    clean_snapshots = [copy.deepcopy(s) for s in snapshots]
    clean_metrics = evaluate_sequence_f1(model, clean_snapshots, test_idx, device, threshold=decision_threshold)
    clean_per_snapshot = evaluate_per_snapshot(model, clean_snapshots, test_idx, device, threshold=decision_threshold)
    attacked = [copy.deepcopy(s) for s in clean_snapshots]

    clean_probs = _predict_indices(model, clean_snapshots, test_idx, device)
    first_test = clean_snapshots[test_idx[0]]
    y0 = first_test.y.cpu().numpy().astype(int)
    fraud_candidates = np.where((y0 == 1) & (first_test.active_mask.cpu().numpy()))[0]
    if not len(fraud_candidates):
        raise RuntimeError("No active labeled illicit nodes in the first test snapshot.")

    # Explicit oracle benchmark: target selection uses first-test labels. This must
    # not be presented as deployable attacker knowledge.
    target_node = int(fraud_candidates[np.argmax(clean_probs[0, fraud_candidates])])

    train_idx, _, _ = split_indices(
        len(snapshots), cfg["training"]["train_ratio"], cfg["training"]["val_ratio"]
    )
    training_degrees = np.concatenate([_degree_vector(clean_snapshots[i]) for i in train_idx])
    source_degree_limit = float(np.percentile(
        training_degrees, float(attack_cfg.get("source_degree_percentile", 95.0))
    ))
    attack_duration = min(int(attack_cfg["attack_duration"]), len(test_idx))
    budget = int(attack_cfg["attack_budget_per_window"])
    max_degree_increase = int(attack_cfg.get("max_degree_increase_per_node", 3))
    max_js_divergence = float(attack_cfg.get("max_degree_distribution_js", 0.05))
    if attack_duration < 1 or budget < 0 or max_degree_increase < 1:
        raise ValueError("attack duration and degree cap must be positive; budget must be nonnegative")
    if not 0.0 <= max_js_divergence <= 1.0:
        raise ValueError("attack.max_degree_distribution_js must be between 0 and 1")

    # Amount is not a measured field in Elliptic. Use the documented neutral
    # placeholder there; only generic datasets may sample training-period amounts.
    if cfg.get("data", {}).get("dataset_type") == "elliptic":
        amount_bounds = (1.0, 1.0)
        amount_policy = "Elliptic placeholder amount 1.0; no amount-based stealth claim"
    else:
        train_amounts = [
            float(value)
            for idx in train_idx
            for value in clean_snapshots[idx].edge_attr[:, 0].cpu().numpy().tolist()
        ]
        if not train_amounts:
            train_amounts = [1.0]
        amount_bounds = tuple(np.quantile(np.asarray(train_amounts), [0.25, 0.75]).tolist())
        amount_policy = "sampled from training-period amount interquartile range"

    fraud_prob_over_time = []
    injected_edges = 0
    successful_steps = 0
    degree_increments = {}
    rejected = {}
    max_observed_js = 0.0
    accepted_constraints = True
    injected_pairs = set()

    for local_t in range(attack_duration):
        snap_idx = test_idx[local_t]
        current_probs = _predict_indices(model, attacked, [snap_idx], device)[0]
        snap = attacked[snap_idx]
        active = getattr(snap, "active_mask", torch.ones_like(snap.y, dtype=torch.bool)).cpu().numpy()
        current_degree = _degree_vector(snap)
        source_candidates = np.where(
            active
            & (np.arange(len(active)) != target_node)
            & (current_degree <= source_degree_limit)
        )[0]
        # Do not use source labels: rank eligible nodes by model score only.
        source_candidates = source_candidates[np.argsort(current_probs[source_candidates])]
        ts_val = int(snap.time_values[-1].item()) if hasattr(snap, "time_values") and snap.time_values.numel() else int(local_t)
        accepted_this_window = 0

        for source in source_candidates:
            if accepted_this_window >= budget:
                break
            if (int(source), target_node) in injected_pairs:
                rejected["duplicate_injected_edge"] = rejected.get("duplicate_injected_edge", 0) + 1
                continue
            ok, divergence, reason = _candidate_passes_constraints(
                clean_snapshots, attacked, snap_idx, int(source), target_node,
                degree_increments, max_degree_increase, max_js_divergence,
            )
            max_observed_js = max(max_observed_js, divergence)
            if not ok:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            amount = float(np.random.uniform(*amount_bounds)) if amount_bounds[1] > amount_bounds[0] else float(amount_bounds[0])
            _append_edge_to_future_snapshots(attacked, snap_idx, int(source), target_node, amount, ts_val)
            injected_pairs.add((int(source), target_node))
            degree_increments[int(source)] = degree_increments.get(int(source), 0) + 1
            degree_increments[target_node] = degree_increments.get(target_node, 0) + 1
            injected_edges += 1
            accepted_this_window += 1

        _recompute_cumulative_features(attacked, snap_idx)
        current_prob = _predict_indices(model, attacked, [snap_idx], device)[0]
        p_target = float(current_prob[target_node])
        fraud_prob_over_time.append(p_target)
        successful_steps += int(p_target < decision_threshold)

    attacked_metrics = evaluate_sequence_f1(model, attacked, test_idx, device, threshold=decision_threshold)
    attacked_per_snapshot = evaluate_per_snapshot(model, attacked, test_idx, device, threshold=decision_threshold)
    return {
        "target_node_index": int(target_node),
        "clean_metrics": clean_metrics,
        "attacked_metrics": attacked_metrics,
        "clean_per_snapshot_metrics": clean_per_snapshot,
        "attacked_per_snapshot_metrics": attacked_per_snapshot,
        "attack_success": bool(fraud_prob_over_time and fraud_prob_over_time[-1] < decision_threshold),
        "decision_threshold": decision_threshold,
        "attack_success_rate_over_steps": float(successful_steps / max(1, len(fraud_prob_over_time))),
        "injected_edges": int(injected_edges),
        "fraud_probability_over_time": fraud_prob_over_time,
        "test_indices": test_idx,
        "seed": int(cfg["project"]["seed"]),
        "attacker_knowledge": "oracle target-label selection on the first test snapshot; source selection uses model scores and structural constraints",
        "amount_policy": amount_policy,
        "stealth_constraints_enforced": bool(accepted_constraints),
        "stealth_constraints": {
            "no_self_loops": True,
            "no_duplicate_directed_edges": True,
            "active_endpoints_only": True,
            "source_degree_percentile_cap": float(attack_cfg.get("source_degree_percentile", 95.0)),
            "max_degree_increase_per_node": max_degree_increase,
            "max_degree_distribution_js": max_js_divergence,
            "observed_max_degree_distribution_js": float(max_observed_js),
            "rejected_candidates_by_reason": rejected,
        },
        "stealth_note": "Constraints enforce degree-distribution similarity and local graph validity only; they do not prove stealth against all graph statistics or a real deployed detector.",
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
