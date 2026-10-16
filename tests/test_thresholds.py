import numpy as np
import pytest

from slrta.eval.thresholds import select_f1_threshold


def test_threshold_selection_uses_validation_probabilities_and_is_deterministic():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.8])

    threshold = select_f1_threshold(labels, probabilities)

    assert 0.0 <= threshold <= 1.0
    assert threshold == select_f1_threshold(labels, probabilities)


def test_threshold_selection_rejects_empty_validation_data():
    with pytest.raises(ValueError, match="empty validation"):
        select_f1_threshold([], [])


def test_threshold_selection_finds_perfect_partition_between_probability_values():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.8])

    threshold = select_f1_threshold(labels, probabilities)

    from slrta.eval.metrics import compute_binary_metrics

    assert compute_binary_metrics(labels, probabilities, threshold)["f1"] == 1.0


def test_threshold_selection_rejects_out_of_range_probabilities():
    with pytest.raises(ValueError, match="between 0 and 1"):
        select_f1_threshold([0, 1], [-0.1, 1.1])
