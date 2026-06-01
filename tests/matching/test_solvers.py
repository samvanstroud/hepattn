import numpy as np
import pytest
from py_lap_solver.solvers import Solvers
from scipy.optimize import linear_sum_assignment


def _total_cost(cost, col_idx):
    return cost[np.arange(len(col_idx)), col_idx].sum()


@pytest.mark.parametrize("size", range(100, 2500, 50))
def test_lap1015_matches_scipy(size):
    solver = Solvers.Lap1015Sequential
    assert solver is not None, "Lap1015Sequential not available in py-lap-solver"

    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=np.float32)
    _, col_scipy = linear_sum_assignment(cost)
    out = solver.solve_single(cost)
    assert np.array_equal(out, col_scipy)

    cost = (np.random.default_rng().random((size, size)) * 1e5).astype(np.float32)
    _, col_scipy = linear_sum_assignment(cost)
    out = solver.solve_single(cost)
    # Assignment may differ on ties, but total cost must match scipy's optimum
    assert np.isclose(_total_cost(cost, out), _total_cost(cost, col_scipy), rtol=1e-5)
    # No unassigned rows for square input
    assert (out >= 0).all()
