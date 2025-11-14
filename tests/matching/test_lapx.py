import lap
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment


@pytest.mark.parametrize("shape", [(5, 5), (8, 10), (15, 15), (50, 100), (1000, 1000)])
@pytest.mark.parametrize("random_func", ["uniform", "normal"])
def test_lapx_scipy_equivalence(shape, random_func) -> None:
    rng = np.random.default_rng()

    if random_func == "uniform":
        x = rng.random(shape)
    elif random_func == "normal":
        x = rng.normal(size=shape)

    lapx_row_indices, lapx_col_indices = lap.lapjvx(x, extend_cost=True, return_cost=False)  # type: ignore[invalid-assignment]
    scipy_row_indices, scipy_col_indices = linear_sum_assignment(x)

    assert np.all(lapx_row_indices == scipy_row_indices)
    assert np.all(lapx_col_indices == scipy_col_indices)
