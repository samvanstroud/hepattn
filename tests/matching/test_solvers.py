import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import lap1015


@pytest.mark.parametrize("size", range(100, 2500, 50))
def test_lap1015(size):
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    _, col_idx_scipy = linear_sum_assignment(cost)
    out_early = lap1015.lap_early(cost)
    out_late = lap1015.lap_late(cost)

    assert all(col_idx_scipy == out_early)
    assert all(col_idx_scipy == out_late)

    cost = np.random.default_rng().random((size, size)) * 1e5
    _, col_idx_scipy = linear_sum_assignment(cost)
    out_early = lap1015.lap_early(cost)
    out_late = lap1015.lap_late(cost)

    # add col indices that are not in the output
    col_idx = np.arange(cost.shape[1])
    col_idx_scipy = np.concatenate([col_idx_scipy, col_idx[~np.isin(col_idx, col_idx_scipy)]])

    assert np.all(col_idx_scipy == out_early)
    assert np.all(col_idx_scipy == out_late)
