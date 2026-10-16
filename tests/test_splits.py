import pytest

from slrta.data.io import split_indices


def test_split_is_contiguous_and_chronological():
    train, val, test = split_indices(20, 0.7, 0.15)
    assert train == list(range(14))
    assert val == list(range(14, 17))
    assert test == list(range(17, 20))


@pytest.mark.parametrize("n,train,val", [(2, 0.7, 0.15), (10, 0.8, 0.2), (10, 0, 0.2), (10, 0.5, 0.6)])
def test_invalid_splits_raise(n, train, val):
    with pytest.raises(ValueError):
        split_indices(n, train, val)
