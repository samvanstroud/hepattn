import numpy as np
from scipy.optimize import linear_sum_assignment

import lap1015


def test_lap1015():
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    _, col_idx_scipy = linear_sum_assignment(cost)
    out = lap1015.lap_early(cost)
    assert all(col_idx_scipy == out)

    # larger test
    cost = np.random.rand(100, 120)  # noqa: NPY002
    _, col_idx_scipy = linear_sum_assignment(cost)
    out = lap1015.lap_late(cost)

    # add col indices that are not in the output
    col_idx = np.arange(cost.shape[1])
    col_idx_scipy = np.concatenate([col_idx_scipy, col_idx[~np.isin(col_idx, col_idx_scipy)]])

    assert np.all(col_idx_scipy == out)
