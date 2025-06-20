import pytest
import torch

from hepattn.models.loss import mask_bce_costs, mask_dice_costs, mask_focal_costs
from hepattn.models.matcher import SOLVERS, Matcher

BATCH_SIZE = 2
SEQ_LEN = 10
NUM_QUERIES = 5


class TestMatcher:
    @pytest.fixture
    def mock_masks(self):
        # Create a true mask and then a perfect prediction, then randomly permute the prediction
        true_mask = (torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN) >= 0.5).float()
        pred_mask = torch.clone(true_mask[:, torch.randperm(NUM_QUERIES), :])
        return pred_mask, true_mask

    def test_matcher(self, mock_masks):
        pred_mask, true_mask = mock_masks

        costs_ce = mask_bce_costs(pred_mask, true_mask)
        costs_dice = mask_dice_costs(pred_mask, true_mask)
        costs_focal = mask_focal_costs(pred_mask, true_mask)

        for solver in SOLVERS:
            matcher = Matcher(default_solver=solver, adaptive_solver=False)

            for costs in [costs_ce, costs_dice, costs_focal]:
                pred_idxs = matcher(costs)

                batch_idxs = torch.arange(costs.shape[0]).unsqueeze(1).expand(-1, costs.shape[-1])

                pred_mask_matched = pred_mask[batch_idxs, pred_idxs]

                # We should be able to exactly recover the original mask
                assert torch.all(true_mask == pred_mask_matched)
