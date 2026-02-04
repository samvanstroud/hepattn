import numpy as np
import pytest
import torch

from hepattn.models.loss import mask_bce_cost, mask_dice_cost, mask_focal_cost
from hepattn.models.matcher import SOLVERS, Matcher, match_multiprocess, match_parallel


@pytest.mark.parametrize("size", [10, 100, 500])
@pytest.mark.parametrize("scale", [1, 1e3, 1e5])
def test_matching_indices(size, scale):
    costs = np.random.default_rng().random((1, size, size)) * scale
    costs = torch.tensor(costs, dtype=torch.float32)

    idxs = []
    for solver in SOLVERS:
        matcher = Matcher(default_solver=solver, adaptive_solver=False)
        idxs.append(matcher(costs))

    assert [np.array_equal(idxs[0], idx) for idx in idxs]


@pytest.mark.parametrize("solver", SOLVERS.keys())
@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("num_queries", [10, 50, 100])
@pytest.mark.parametrize("seq_len", [50, 100, 200])
def test_mask_recovery(solver, batch_size, num_queries, seq_len):
    torch.manual_seed(42)

    # Create a true mask and then a perfect prediction, then randomly permute the prediction
    true_mask = (torch.randn(batch_size, num_queries, seq_len) >= 0.5).float()
    pred_mask = torch.clone(true_mask[:, torch.randperm(num_queries), :])

    # compute costs
    costs_ce = mask_bce_cost(pred_mask, true_mask)
    costs_dice = mask_dice_cost(pred_mask, true_mask)
    costs_focal = mask_focal_cost(pred_mask, true_mask)

    # create a matcher
    matcher = Matcher(default_solver=solver, adaptive_solver=False)

    # check that we can exactly recover the true mask for each cost
    for costs in [costs_ce, costs_dice, costs_focal]:
        pred_idxs = matcher(costs)
        assert torch.all(pred_idxs >= 0)
        batch_idxs = torch.arange(costs.shape[0]).unsqueeze(1).expand(-1, costs.shape[-1])
        pred_mask_matched = pred_mask[batch_idxs, pred_idxs]
        assert torch.all(true_mask == pred_mask_matched)


@pytest.mark.parametrize("solver", SOLVERS.keys())
@pytest.mark.parametrize("batch_size", [8, 16])
@pytest.mark.parametrize("num_queries", [50, 100])
def test_parallel_matching_correctness(solver, batch_size, num_queries):
    """Test that parallel matching produces the same results as sequential matching."""
    torch.manual_seed(42)

    # Create random cost matrix
    costs = torch.randn(batch_size, num_queries, num_queries)

    # Sequential matcher
    matcher_sequential = Matcher(default_solver=solver, adaptive_solver=False, parallel_solver=False)

    # Parallel matcher with 2 jobs
    matcher_parallel = Matcher(default_solver=solver, adaptive_solver=False, parallel_solver=True, n_jobs=2)

    # Get results from both
    sequential_result = matcher_sequential(costs)
    parallel_result = matcher_parallel(costs)

    # Check that results are identical
    assert torch.equal(sequential_result, parallel_result), f"Results differ for solver {solver}"


@pytest.mark.parametrize("solver", SOLVERS.keys())
@pytest.mark.parametrize("batch_size", [8, 16])
@pytest.mark.parametrize("num_queries", [50, 100])
def test_multiprocess_matching_correctness(solver, batch_size, num_queries):
    """Test that multiprocess matching produces the same results as sequential matching."""
    torch.manual_seed(42)

    # Create random cost matrix
    costs = torch.randn(batch_size, num_queries, num_queries)

    # Sequential matcher
    matcher_sequential = Matcher(default_solver=solver, adaptive_solver=False, parallel_solver=False)

    # Multiprocess matcher
    matcher_multiprocess = Matcher(default_solver=solver, adaptive_solver=False, parallel_solver=True, parallel_backend="process", n_jobs=2)

    # Get results from both
    sequential_result = matcher_sequential(costs)
    multiprocess_result = matcher_multiprocess(costs)

    # Check that results are identical
    assert torch.equal(sequential_result, multiprocess_result), f"Results differ for solver {solver}"


@pytest.mark.parametrize("solver", SOLVERS.keys())
def test_query_valid_mask(solver):
    """Test that query_valid_mask properly masks out invalid queries."""
    torch.manual_seed(42)

    batch_size, num_queries = 4, 20

    # Create random cost matrix
    costs = torch.randn(batch_size, num_queries, num_queries)

    # Create query valid mask - mark some queries as invalid
    query_valid_mask = torch.ones(batch_size, num_queries, dtype=torch.bool)
    query_valid_mask[:, -5:] = False  # Last 5 queries are invalid

    matcher = Matcher(default_solver=solver, adaptive_solver=False)

    # Run with and without mask
    result_no_mask = matcher(costs)
    result_with_mask = matcher(costs, query_valid_mask=query_valid_mask)

    # Results should be different since invalid queries have high cost
    assert result_no_mask.shape == result_with_mask.shape


@pytest.mark.parametrize("solver", SOLVERS.keys())
def test_adaptive_solver_verbose(solver, capsys):
    """Test adaptive solver with verbose output."""
    torch.manual_seed(42)

    costs = torch.randn(2, 10, 10)

    # Create matcher with adaptive solver and verbose mode
    matcher = Matcher(default_solver=solver, adaptive_solver=True, adaptive_check_interval=1, verbose=True)

    # Run the matcher - this should trigger adapt_solver and print verbose output
    matcher(costs)

    # Capture printed output
    captured = capsys.readouterr()
    assert "Adaptive LAP Solver" in captured.out


def test_parallel_single_batch():
    """Test parallel matching with batch_size=1 and n_jobs=1 (edge case)."""
    torch.manual_seed(42)

    costs = torch.randn(1, 10, 10)

    # Parallel matcher with n_jobs=1 should fallback to sequential
    matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=True, n_jobs=1)

    result = matcher(costs)
    assert result.shape == (1, 10)


def test_match_parallel_low_jobs():
    """Test match_parallel function with n_jobs <= 1."""
    costs = np.random.default_rng(42).random((2, 10, 10)).astype(np.float32)
    lengths = np.array([10, 10], dtype=np.int32)

    result = match_parallel(SOLVERS["scipy"], costs, lengths, pred_dim=10, n_jobs=1)
    assert result.shape == (2, 10)


def test_match_multiprocess_unknown_solver():
    """Test match_multiprocess raises error for unknown solver."""
    costs = np.random.default_rng(42).random((2, 10, 10)).astype(np.float32)
    lengths = np.array([10, 10], dtype=np.int32)

    with pytest.raises(ValueError, match="Unknown solver"):
        match_multiprocess("nonexistent_solver", costs, lengths, pred_dim=10, n_jobs=2)


def test_matcher_invalid_solver():
    """Test Matcher raises error for invalid solver name."""
    with pytest.raises(ValueError, match="Unknown solver"):
        Matcher(default_solver="nonexistent_solver")


def test_matcher_invalid_parallel_backend():
    """Test Matcher raises error for invalid parallel_backend."""
    with pytest.raises(ValueError, match="parallel_backend must be"):
        Matcher(parallel_backend="invalid_backend")  # type: ignore[arg-type]
