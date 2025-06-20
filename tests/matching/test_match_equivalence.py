import time

import numpy as np
import pytest
import torch

from hepattn.models.matcher import SOLVERS, Matcher


@pytest.mark.parametrize("solver_name", SOLVERS)
@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 1000.0])
def test_matcher_equivalence(solver_name, scale: float):
    matcher = Matcher(default_solver=solver_name, adaptive_solver=False)
    rng = np.random.default_rng()

    for _ in range(200):
        n_objects = rng.integers(100, 250)
        n_valid_objects = rng.integers(n_objects // 4, n_objects // 2)
        cost = rng.random((1, n_objects, n_objects)) * scale  # Add batch dimension
        pad_mask = np.zeros((1, n_objects), dtype=bool)  # Add batch dimension
        pad_mask[0, :n_valid_objects] = True
        cost[0, :, ~pad_mask[0]] = 1e8  # Set padding costs to infinity

        # Convert to torch tensors
        cost_tensor = torch.from_numpy(cost).float()
        pad_mask_tensor = torch.from_numpy(pad_mask)

        # Test with full cost matrix
        idx_all = matcher(cost_tensor).squeeze(0).numpy()

        # Test with padded cost matrix (simulate what happens internally)
        cost_padded = cost.copy()
        idx_padded = matcher(torch.from_numpy(cost_padded).float(), pad_mask_tensor).squeeze(0).numpy()

        assert np.all(idx_all[:n_valid_objects] == idx_padded[:n_valid_objects]), (
            f"Solver {solver_name} mismatch for n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )
        assert len(set(idx_all)) == n_objects, (
            f"Solver {solver_name} produced duplicate indices for n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )
        assert len(set(idx_padded)) == n_objects, (
            f"Solver {solver_name} produced duplicate indices for padded n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_matcher_speed(solver_name):
    matcher = Matcher(default_solver=solver_name, adaptive_solver=False)
    rng = np.random.default_rng()
    times_all = []
    times_padded = []

    for _ in range(1024):
        n_objects = rng.integers(100, 250)
        n_valid_objects = rng.integers(n_objects // 4, n_objects // 2)
        cost = rng.random((1, n_objects, n_objects)) * 100  # Add batch dimension
        pad_mask = np.zeros((1, n_objects), dtype=bool)  # Add batch dimension
        pad_mask[0, :n_valid_objects] = True
        cost[0, :, ~pad_mask[0]] = 1e8  # Set padding costs to infinity

        # Convert to torch tensors
        cost_tensor = torch.from_numpy(cost).float()
        pad_mask_tensor = torch.from_numpy(pad_mask)

        start_time = time.perf_counter()
        _ = matcher(cost_tensor)
        times_all.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        _ = matcher(cost_tensor, pad_mask_tensor)
        times_padded.append(time.perf_counter() - start_time)

    avg_time_all = np.mean(times_all)
    avg_time_padded = np.mean(times_padded)
    print(f"Solver: {solver_name}, Avg Time All: {avg_time_all:.6f}s, Avg Time Padded: {avg_time_padded:.6f}s")
