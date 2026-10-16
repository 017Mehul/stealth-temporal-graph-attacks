import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5):
    """Compute binary metrics with explicit behavior for empty/single-class data."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    if y_true.size != y_prob.size:
        raise ValueError("y_true and y_prob must have the same number of elements")
    if y_true.size == 0:
        return {
            "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "accuracy": 0.0, "roc_auc": None, "pr_auc": None,
            "n_samples": 0, "threshold": float(threshold),
        }
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or infinite values")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")

    y_pred = (y_prob >= threshold).astype(int)
    result = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": None,
        "pr_auc": None,
        "n_samples": int(y_true.size),
        "threshold": float(threshold),
    }
    if np.unique(y_true).size == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        result["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return result
