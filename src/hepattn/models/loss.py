import torch
import torch.nn.functional as F

eps = 1e-6


def object_ce_loss(pred_logits, true, mask=None, weight=None):  # noqa: ARG001
    # TODO: Add support for mask?
    losses = F.binary_cross_entropy_with_logits(pred_logits, true, weight=weight)
    return losses.mean()


def object_ce_costs(pred_logits, true):
    losses = F.binary_cross_entropy_with_logits(
        pred_logits.unsqueeze(2).expand(-1, -1, true.shape[1]), true.unsqueeze(1).expand(-1, pred_logits.shape[1], -1), reduction="none"
    )
    return losses


def mask_dice_loss(pred_logits, true, mask=None, weight=None):
    pred = pred_logits.sigmoid()
    num = 2 * (pred * true)
    den = (pred.sum(-1) + true.sum(-1)).unsqueeze(-1)
    losses = 1 - (num + 1) / (den + 1)

    if weight is not None:
        losses *= weight

    if mask is not None:
        losses = losses[mask]

    return losses.mean()


def mask_dice_costs(pred_logits, true):
    pred = pred_logits.sigmoid()
    num = 2 * torch.einsum("bnc,bmc->bnm", pred, true)
    den = pred.sum(-1).unsqueeze(2) + true.sum(-1).unsqueeze(1)
    losses = 1 - (num + 1) / (den + 1)
    return losses


def mask_focal_loss(pred_logits, true, alpha=-1.0, gamma=2.0, mask=None, weight=None):
    pred = pred_logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, true, reduction="none")
    p_t = pred * true + (1 - pred) * (1 - true)
    losses = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * true + (1 - alpha) * (1 - true)
        losses = alpha_t * losses

    if weight is not None:
        losses *= weight

    if mask is not None:
        losses = losses[mask]

    return losses.mean()


def mask_focal_costs(pred_logits, true, alpha=-1.0, gamma=2.0):
    pred = pred_logits.sigmoid()
    focal_pos = ((1 - pred) ** gamma) * F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred), reduction="none")
    focal_neg = (pred**gamma) * F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred), reduction="none")
    if alpha >= 0:
        focal_pos *= alpha
        focal_neg *= 1 - alpha
    losses = torch.einsum("bnc,bmc->bnm", focal_pos, true) + torch.einsum("bnc,bmc->bnm", focal_neg, (1 - true))
    return losses


def mask_ce_loss(pred_logits, true, mask=None, weight=None):
    losses = F.binary_cross_entropy_with_logits(pred_logits, true, weight=weight, reduction="none")

    if mask is not None:
        losses = losses[mask]

    return losses.mean()


def mask_ce_costs(pred_logits, true):
    pos = F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred_logits), reduction="none")
    neg = F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred_logits), reduction="none")
    losses = torch.einsum("bnc,bmc->bnm", pos, true) + torch.einsum("bnc,bmc->bnm", neg, (1 - true))
    return losses


cost_fns = {
    "object_ce": object_ce_costs,
    "mask_ce": mask_ce_costs,
    "mask_dice": mask_dice_costs,
    "mask_focal": mask_focal_costs,
}

loss_fns = {"object_ce": object_ce_loss, "mask_ce": mask_ce_loss, "mask_dice": mask_dice_loss, "mask_focal": mask_focal_loss}
