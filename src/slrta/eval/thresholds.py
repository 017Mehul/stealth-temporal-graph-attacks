import numpy as np

from slrta.eval.metrics import compute_binary_metrics


def select_f1_threshold(y_true, y_prob):
    """Select a classification threshold on validation predictions only."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    if y_true.size != y_prob.size:
        raise ValueError("y_true and y_prob must have the same number of elements")
    if y_true.size == 0:
        raise ValueError("Cannot select a threshold from empty validation predictions")
    if not np.isfinite(y_prob).all():
        raise ValueError("Validation probabilities contain NaN or infinity")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Validation labels must be binary")

    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_prob)))
    scored = [(compute_binary_metrics(y_true, y_prob, float(t))["f1"], float(t)) for t in candidates]
    best_f1 = max(item[0] for item in scored)
    # Deterministic tie-break: choose the threshold nearest 0.5.
    best_threshold = min(
        (threshold for f1, threshold in scored if np.isclose(f1, best_f1)),
        key=lambda threshold: (abs(threshold - 0.5), threshold),
    )
    return float(best_threshold)
