"""
Module to compute the matching cost and solve the corresponding LSAP using the Hungarian algorithm.

Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former
"""

import time

import scipy
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lap1015 import lap_early, lap_late  # type: ignore


# @torch.compile
def batch_dice_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the DICE loss for the entire batch, similar to generalized IOU for masks, over all permutations of the inputs and targets."""

    inputs = inputs.sigmoid()

    # inputs has shape (B, N, C), targets has shape (B, M, C)
    # We want to compute the DICE loss for each combination of N and M for each batch

    # Using torch.einsum to handle the batched matrix multiplication
    numerator = 2 * torch.einsum("bnc,bmc->bnm", inputs, targets)

    # Compute the denominator using sum over the last dimension (C) and broadcasting
    denominator = inputs.sum(-1).unsqueeze(2) + targets.sum(-1).unsqueeze(1)

    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss


# @torch.compile
def batch_sigmoid_ce_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the cross entropy loss over all permutations of the inputs and targets."""
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("bnc,bmc->bnm", pos, targets) + torch.einsum("bnc,bmc->bnm", neg, (1 - targets))
    return loss / inputs.shape[2]


# @torch.compile
def batch_sigmoid_focal_cost(inputs: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2) -> Tensor:
    """Compute the focal loss over all permutations of the inputs and targets for batches."""
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)
    loss = torch.einsum("bnc,bmc->bnm", focal_pos, targets) + torch.einsum("bnc,bmc->bnm", focal_neg, (1 - targets))
    return loss / inputs.shape[2]


# @torch.compile
def batch_mae_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the mean average error over all permutations of the inputs and targets for batches."""
    return (inputs[:, :, None] - targets[:, None, :]).abs().mean(-1)


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: dict,
        adaptive_lap: bool,
    ):
        """Compute the optimal assignment between the targets and the predictions of the network.

        For efficiency reasons, the targets don't include the no_object null class. Because of this, in general,
        there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
        while the others are un-matched (and thus treated as non-objects).

        Parameters
        ----------
        num_classes: int
            Number of object classes, omitting the special no_object class
        num_objects: int
            Number of object classes + 1 (the no_object class)
        loss_weights: dict
            Dictionary containing the weights for the different losses
        adaptive_lap: bool
            Attempt to use the fastest lap solver for the current problem.
            For small cost matrices, use scipy. For larger ones, use 1015 algorithm.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_objects = num_objects
        self.loss_weights = loss_weights
        assert sum(self.loss_weights.values()) != 0, "Sum of loss weights must be positive"

        self.global_step = 0
        self.adaptive_lap = adaptive_lap
        if self.adaptive_lap:
            self.lap_fn = None
        else:
            self.lap_fn = self.lap_scipy

    def get_batch_cost(self, preds, targets):
        """
        Gets the cost matrix for the whole batch

        Parameters
        ----------
        preds : dict of tensors
            The predictions from the model
        targets : dict of tensors
            The targets for the model

        Returns
        -------
        C : tensor
            The cost matrix for the whole batch, with a large value set in the input pad region
        batch_obj_lengths : tensor
            The number of inputs in each batch element
        """

        # get some useful things
        bs = len(targets["particle_valid"])
        dev = preds["masks"].device

        valid_obj_mask = targets["particle_valid"].detach()
        mask_pred = preds["masks"].detach()
        mask_tgt = targets["particle_hit_valid"].detach().to(mask_pred.dtype)

        batch_obj_lengths = torch.sum(valid_obj_mask, dim=1)

        C = None
        if "class_probs" in preds:
            obj_class_pred = preds["class_probs"].detach()  # [B, N_pred, 2]
            obj_class_tgt = valid_obj_mask.long()           # [B, N_tgt]

            # Expand predictions and targets for pairwise comparison
            obj_class_pred_exp = obj_class_pred.unsqueeze(2).expand(-1, -1, obj_class_tgt.size(1), -1)  # [B, N_pred, N_tgt, 2]
            obj_class_tgt_exp = obj_class_tgt.unsqueeze(1).unsqueeze(-1).expand(-1, obj_class_pred.size(1), -1, -1)  # [B, N_pred, N_tgt, 1]

            # Gather the predicted probability for the correct class (0 or 1)
            output = torch.gather(obj_class_pred_exp, 3, obj_class_tgt_exp).squeeze(-1)  # [B, N_pred, N_tgt]

            # Mask invalid targets
            valid_mask = valid_obj_mask.unsqueeze(1).expand_as(output)
            output = output * valid_mask

            # Compute class cost (negative probability)
            obj_class_cost = -output  # [B, N_pred, N_tgt]

            # Final weighted cost
            C = self.loss_weights["object_class_ce"] * obj_class_cost


        # add mask costs
        if self.loss_weights.get("mask_dice"):
            cost_mask_dice = batch_dice_cost(mask_pred, mask_tgt)
            if C is None:
                C = self.loss_weights["mask_dice"] * cost_mask_dice
            else:
                C += self.loss_weights["mask_dice"] * cost_mask_dice
        if self.loss_weights.get("mask_ce"):
            cost_mask_ce = batch_sigmoid_ce_cost(mask_pred, mask_tgt)
            if C is None:
                C = self.loss_weights["mask_ce"] * cost_mask_ce
            else:
                C += self.loss_weights["mask_ce"] * cost_mask_ce
        if self.loss_weights.get("mask_focal"):
            cost_mask_focal = batch_sigmoid_focal_cost(mask_pred, mask_tgt)
            if C is None:
                C = self.loss_weights["mask_focal"] * cost_mask_focal
            else:
                C += self.loss_weights["mask_focal"] * cost_mask_focal

        # set entries corresponding to invalid objects to nan (these are removed later when running LSAP)
        batch_obj_lengths = batch_obj_lengths.unsqueeze(-1)
        col_indices = torch.arange(obj_class_tgt.shape[1], device=dev).unsqueeze(0)
        null_obj_cost_mask = (col_indices < batch_obj_lengths).unsqueeze(1).expand_as(C)
        C[~null_obj_cost_mask] = torch.nan  # TODO: initialise as nan and then overwrite?

        return C, batch_obj_lengths

    @torch.no_grad()
    def forward(self, preds, targets):
        """Compute the optimal assignment between the targets and the predictions of the network."""
        batch_size, num_queries = preds["masks"].shape[:2]

        idxs = []
        self.default_idx = set(range(self.num_objects))

        # Get the full cost matrix, then run lsap on each batch element
        full_cost, batch_N = self.get_batch_cost(preds, targets)
        full_cost = full_cost.to(torch.float32).cpu().numpy()
        for batch_idx in range(batch_size):
            # get the cost matrix for this batch element
            C = full_cost[batch_idx][:, : batch_N[batch_idx]]

            # if batch_N[batch_idx] == 0:
            #     print(f"[WARN] Skipping LAP for empty batch {batch_idx}")
            #     idxs.append(torch.zeros(self.num_objects, dtype=torch.long))
            #     continue
            # else:
            #     print("Running LAP")

            # get the optimal assignment
            idx = self.lap(C)
            idxs.append(idx)

        # format indices to allow simple indexing
        idxs = torch.stack(idxs)
        idxs = (torch.arange(len(idxs)).unsqueeze(1), idxs)
        self.global_step += 1
        return idxs

    def lap_scipy(self, cost):
        src_idx, tgt_idx = scipy.optimize.linear_sum_assignment(cost)
        idx = src_idx[tgt_idx]
        idx = list(idx) + sorted(list(self.default_idx - set(idx)))
        return torch.as_tensor(idx)

    # TODO: put all lap stuff in an interface
    def lap_early(self, cost):
        return torch.as_tensor(lap_early(cost.T))

    def lap_late(self, cost):
        return torch.as_tensor(lap_late(cost.T))

    def lap(self, cost):
        if self.lap_fn is None:
            if cost.size < 200:
                self.lap_fn = self.lap_scipy
            else:
                self.lap_fn = self.lap_early

        if self.lap_fn == self.lap_early and self.global_step % 1000 == 0:
            self.lap_fn = self.get_faster_lap(cost)

        return self.lap_fn(cost)

    def get_faster_lap(self, cost):
        start = time.time()
        self.lap_early(cost)
        early_time = time.time() - start

        start = time.time()
        self.lap_late(cost)
        late_time = time.time() - start

        if early_time < late_time:
            return self.lap_early
        return self.lap_late
