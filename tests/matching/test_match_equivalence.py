import time

import numpy as np
import pytest
import torch

from hepattn.models.matcher import SOLVERS, Matcher


def generate_dummy_cost(rng: np.random.Generator, batch_size: int, n_objects: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    cost = rng.random((batch_size, n_objects, n_objects)) * scale  # Add batch dimension
    object_valid_mask = np.zeros((batch_size, n_objects), dtype=bool)
    for i in range(batch_size):
        # Randomly set a portion of objects as valid
        n_valid_objects = rng.integers(n_objects // 4, n_objects // 2)
        object_valid_mask[i, :n_valid_objects] = True
    cost[~object_valid_mask[:, None, :].repeat(n_objects, 1)] = 1e4  # Set padding costs to large value
    return cost, object_valid_mask


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 1000.0])
def test_matcher_with_target_padding(solver, scale: float):
    matcher = Matcher(default_solver=solver, adaptive_solver=False)
    rng = np.random.default_rng()

    for _ in range(10):
        n_objects = rng.integers(100, 250)
        cost, object_valid_mask = generate_dummy_cost(rng, 1, n_objects, scale)

        # Convert to torch tensors
        cost_tensor = torch.from_numpy(cost).float()
        object_valid_mask = torch.from_numpy(object_valid_mask)
        n_valid_objects = object_valid_mask.sum()

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
@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 1000.0])
def test_matcher_with_parallel_solver(solver, scale: float):
    matcher = Matcher(default_solver=solver, adaptive_solver=False)
    matcher_parallel = Matcher(default_solver=solver, adaptive_solver=False, parallel_solver=True)
    rng = np.random.default_rng()
    batch_size = 32
    for _ in range(10):
        n_objects = rng.integers(100, 250)
        cost, object_valid_mask = generate_dummy_cost(rng, batch_size, n_objects, scale)

        # Convert to torch tensors
        cost_tensor = torch.from_numpy(cost).float()
        object_valid_mask = torch.from_numpy(object_valid_mask)

        idxs = matcher(cost_tensor, object_valid_mask).numpy()
        idxs_parallel = matcher_parallel(cost_tensor, object_valid_mask).numpy()
        object_valid_mask = object_valid_mask.numpy()

        assert idxs_parallel.shape == (batch_size, n_objects), (
            f"Parallel Solver {solver} produced incorrect shape for idxs: {idxs_parallel.shape}, expected {(batch_size, n_objects)}"
        )
        assert np.all(idxs_parallel[object_valid_mask] == idxs[object_valid_mask]), (
            f"Parallel Solver {solver} produced different indices for valid objects: {np.where(object_valid_mask)[0]}"
            f" for n_objects={n_objects}, batch_size={batch_size}, scale={scale}"
        )
        assert np.all(idxs_parallel[~object_valid_mask] == idxs[~object_valid_mask]), (
            f"Parallel Solver {solver} produced different indices for invalid objects: {np.where(~object_valid_mask)[0]}"
            f" for n_objects={n_objects}, batch_size={batch_size}, scale={scale}"
        )


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("parallel_solver", [True, False])
def test_matcher_speed(solver, parallel_solver: bool):
    matcher = Matcher(default_solver=solver, adaptive_solver=False, parallel_solver=parallel_solver)
    batch_size = 256
    rng = np.random.default_rng()
    times_all = []
    times_padded = []

    for _ in range(16):
        n_objects = rng.integers(100, 250)
        cost, object_valid_mask = generate_dummy_cost(rng, batch_size, n_objects)

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
    print(f"Solver: {solver} (parallel={parallel_solver}), Avg Time All: {avg_time_all:.6f}s, Avg Time Padded: {avg_time_padded:.6f}s")
