import time

import numpy as np
import pytest
import torch

from hepattn.models.matcher import SOLVERS, Matcher


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 1000.0])
def test_matcher_with_target_padding(solver, scale: float):
    matcher = Matcher(default_solver=solver, adaptive_solver=False)
    rng = np.random.default_rng()

    for _ in range(10):
        n_objects = rng.integers(100, 250)
        n_valid_objects = rng.integers(n_objects // 4, n_objects // 2)
        cost = rng.random((1, n_objects, n_objects)) * scale  # Add batch dimension
        object_valid_mask = np.zeros((1, n_objects), dtype=bool)  # Add batch dimension
        object_valid_mask[0, :n_valid_objects] = True
        cost[0, :, ~object_valid_mask[0]] = 1e4  # Set padding costs

        # Convert to torch tensors
        cost_tensor = torch.from_numpy(cost).float()
        object_valid_mask = torch.from_numpy(object_valid_mask)

        # Test with full cost matrix
        idx_all = matcher(cost_tensor).squeeze(0).numpy()

        # Test with padded cost matrix
        cost[0, :, ~object_valid_mask[0]] = np.nan  # if we pass the pad mask the nans should be removed
        cost_padded = cost.copy()
        idx_padded = matcher(torch.from_numpy(cost_padded).float(), object_valid_mask).squeeze(0).numpy()

        assert np.all(idx_all[:n_valid_objects] == idx_padded[:n_valid_objects]), (
            f"Solver {solver} mismatch for n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )
        assert len(set(idx_all)) == n_objects, (
            f"Solver {solver} produced duplicate indices for n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )
        assert len(set(idx_padded)) == n_objects, (
            f"Solver {solver} produced duplicate indices for padded n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )


@pytest.mark.parametrize("solver", SOLVERS)
def test_matcher_speed(solver):
    matcher = Matcher(default_solver=solver, adaptive_solver=False)
    rng = np.random.default_rng()
    times_all = []
    times_padded = []

    for _ in range(1024):
        n_objects = rng.integers(100, 250)
        n_valid_objects = rng.integers(n_objects // 4, n_objects // 2)
        cost = rng.random((1, n_objects, n_objects)) * 100  # Add batch dimension
        object_valid_mask = np.zeros((1, n_objects), dtype=bool)  # Add batch dimension
        object_valid_mask[0, :n_valid_objects] = True
        cost[0, :, ~object_valid_mask[0]] = 1e8  # Set padding costs to infinity

        # Convert to torch tensors
        cost_tensor = torch.from_numpy(cost).float()
        object_valid_mask = torch.from_numpy(object_valid_mask)

        start_time = time.perf_counter()
        _ = matcher(cost_tensor)
        times_all.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        _ = matcher(cost_tensor, object_valid_mask)
        times_padded.append(time.perf_counter() - start_time)

    avg_time_all = np.mean(times_all)
    avg_time_padded = np.mean(times_padded)
    print(f"Solver: {solver}, Avg Time All: {avg_time_all:.6f}s, Avg Time Padded: {avg_time_padded:.6f}s")
