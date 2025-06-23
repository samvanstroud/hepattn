import numpy as np
import pytest
import torch

from hepattn.models.loss import mask_bce_costs, mask_dice_costs, mask_focal_costs
from hepattn.models.matcher import SOLVERS, Matcher


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
    costs_ce = mask_bce_costs(pred_mask, true_mask)
    costs_dice = mask_dice_costs(pred_mask, true_mask)
    costs_focal = mask_focal_costs(pred_mask, true_mask)

    # create a matcher
    matcher = Matcher(default_solver=solver, adaptive_solver=False)

    # check that we can exactly recover the true mask for each cost
    for costs in [costs_ce, costs_dice, costs_focal]:
        pred_idxs = matcher(costs)
        assert torch.all(pred_idxs >= 0)
        batch_idxs = torch.arange(costs.shape[0]).unsqueeze(1).expand(-1, costs.shape[-1])
        pred_mask_matched = pred_mask[batch_idxs, pred_idxs]
        assert torch.all(true_mask == pred_mask_matched)
