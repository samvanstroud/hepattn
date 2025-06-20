import torch
import torch.nn.functional as F


def object_bce_loss(pred_logits, true, mask=None, weight=None):  # noqa: ARG001
    # TODO: Add support for mask?
    losses = F.binary_cross_entropy_with_logits(pred_logits, true, weight=weight)
    return losses.mean()


def object_bce_costs(pred_logits, true):
    costs = F.binary_cross_entropy_with_logits(
        pred_logits.unsqueeze(2).expand(-1, -1, true.shape[1]), true.unsqueeze(1).expand(-1, pred_logits.shape[1], -1), reduction="none"
    )
    return costs


def mask_dice_loss(pred_logits, true, mask=None, weight=None, eps=1e-6):
    pred = pred_logits.sigmoid()
    intersection = pred * true
    num_pred = pred.sum(-1).unsqueeze(-1)
    num_true = true.sum(-1).unsqueeze(-1)
    losses = 1 - (intersection + eps) / (num_pred + num_true + eps)

    if weight is not None:
        losses *= weight

    if mask is not None:
        losses = losses[mask]

    return losses.mean()


def mask_dice_costs(pred_logits, true, eps=1e-6):
    num_pred = pred_logits.sum(-1).unsqueeze(2)
    num_true = true.sum(-1).unsqueeze(1)

    # Context manager necessary to overwride global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        pred = pred_logits.sigmoid()
        intersection = torch.einsum("bnc,bmc->bnm", pred, true)
        costs = 1 - (2 * intersection + eps) / (num_pred + num_true + eps)

    return costs


def mask_iou_costs(pred_logits, true, eps=1e-6):
    num_pred = pred_logits.sum(-1).unsqueeze(2)
    num_true = true.sum(-1).unsqueeze(1)

    # Context manager necessary to overwride global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        pred = pred_logits.sigmoid()
        intersection = torch.einsum("bnc,bmc->bnm", pred, true)
        costs = 1 - (intersection + eps) / (eps + num_pred + num_true - intersection)

    return costs


def focal_loss(pred_logits, targets, balance=True, gamma=2.0, mask=None, weight=None):
    pred = pred_logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets.type_as(pred_logits), reduction="none")
    p_t = pred * targets + (1 - pred) * (1 - targets)
    losses = ce_loss * ((1 - p_t) ** gamma)

    if balance:
        alpha = 1 - targets.float().mean()
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
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

    # Context manager necessary to overwride global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        costs = torch.einsum("bnc,bmc->bnm", focal_pos, true) + torch.einsum("bnc,bmc->bnm", focal_neg, (1 - true))

    return costs


def mask_bce_loss(pred_logits, true, mask=None, weight=None):
    losses = F.binary_cross_entropy_with_logits(pred_logits, true, weight=weight, reduction="none")

    if mask is not None:
        losses = losses[mask]

    return losses.mean()


def mask_bce_costs(pred_logits, true):
    pred_logits = torch.clamp(pred_logits, -100, 100)

    pos = F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred_logits), reduction="none")
    neg = F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred_logits), reduction="none")

    # Context manager necessary to overwride global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        costs = torch.einsum("bnc,bmc->bnm", pos, true) + torch.einsum("bnc,bmc->bnm", neg, (1 - true))

    return costs


def regr_mse_loss(pred, true):
    return torch.nn.functional.mse_loss(pred, true, reduction="none")


def regr_smooth_l1_loss(pred, true):
    return torch.nn.functional.smooth_l1_loss(pred, true, reduction="none")


def regr_mse_costs(pred, true):
    return torch.nn.functional.mse_loss(pred.unsqueeze(-2), true.unsqueeze(-3), reduction="none")


def regr_smooth_l1_costs(pred, true):
    return torch.nn.functional.mse_loss(pred.unsqueeze(-2), true.unsqueeze(-3), reduction="none")


cost_fns = {
    "object_bce": object_bce_costs,
    "mask_bce": mask_bce_costs,
    "mask_dice": mask_dice_costs,
    "mask_focal": mask_focal_costs,
    "mask_iou": mask_iou_costs,
}

loss_fns = {
    "object_bce": object_bce_loss,
    "mask_bce": mask_bce_loss,
    "mask_dice": mask_dice_loss,
    "mask_focal": focal_loss,
}
