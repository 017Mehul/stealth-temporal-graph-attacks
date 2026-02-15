from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    # Accuracy
    try:
        metrics["accuracy"] = float((y_pred == y_true).astype(int).mean())
    except Exception:
        metrics["accuracy"] = 0.0

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.5

    return metrics
