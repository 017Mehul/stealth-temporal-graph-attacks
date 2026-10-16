import numpy as np
import pytest

from slrta.eval.metrics import compute_binary_metrics


def test_empty_metrics_are_explicit():
    result = compute_binary_metrics([], [])
    assert result["n_samples"] == 0
    assert result["roc_auc"] is None
    assert result["pr_auc"] is None


def test_single_class_metrics_do_not_claim_auc():
    result = compute_binary_metrics([0, 0], [0.1, 0.2])
    assert result["roc_auc"] is None
    assert result["pr_auc"] is None
    assert result["accuracy"] == 1.0


def test_binary_metrics_include_pr_auc():
    result = compute_binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.8])
    assert result["roc_auc"] == 1.0
    assert result["pr_auc"] == 1.0
    assert result["f1"] == 1.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_binary_metrics([0], [0.1, 0.2])
