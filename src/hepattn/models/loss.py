import torch
import torch.nn.functional as F


def object_bce_loss(pred_logits, targets, sample_weight=None):
    """
    Loss function for binary object classification.

    Args:
        pred_logits: [batch_size, num_objects] - predicted logits for binary classification
        targets: [batch_size, num_objects] - ground truth class labels
        sample_weight: Optional sample weights for each element

    Returns:
        loss: Scalar tensor representing the binary cross-entropy loss
    """
    return F.binary_cross_entropy_with_logits(pred_logits, targets, weight=sample_weight)


def object_bce_cost(pred_logits, targets):
    """
    Compute batched binary object classification cost for object matching.
    Approximate the CE loss using -probs[target_class].
    Invalid objects are handled later in the matching process.

    Args:
        pred_logits: [batch_size, num_objects] - predicted logits for binary classification
        targets: [batch_size, num_objects] - ground truth class labels

    Returns:
        cost_class: [batch_size, num_objects, num_objects] - classification cost matrix
    """
    probs = pred_logits.sigmoid().unsqueeze(-1)
    targets = targets.unsqueeze(1)
    return -probs * targets - (1 - probs) * (1 - targets)


def object_ce_loss(pred_probs, true, mask=None, weight=None):  # noqa: ARG001
    losses = F.cross_entropy(pred_probs.flatten(0, 1), true.flatten(0, 1), weight=weight)
    return losses.mean()


def object_ce_cost(pred_logits, targets):
    """
    Compute batched multiclass object classification cost for object matching.
    Approximate the CE loss using -probs[target_class].
    Invalid objects are handled later in the matching process.

    Args:
        pred_logits: [batch_size, num_objects, num_classes] - predicted logits (num_classes>1)
        targets: [batch_size, num_objects] - ground truth class labels

    Returns:
        cost_class: [batch_size, num_objects, num_targets] - classification cost matrix
    """
    assert pred_logits.shape[-1] > 1
    probs = pred_logits.softmax(-1)

    batch_size, num_queries, _ = probs.shape
    num_targets = targets.size(1)

    index = targets.unsqueeze(1).expand(batch_size, num_queries, num_targets)
    return -torch.gather(probs, dim=2, index=index)


def mask_dice_loss(pred_logits, targets, object_valid_mask=None, input_pad_mask=None, norm_weight=1.0, sample_weight=None):  # noqa: ARG001
    """
    Compute the DICE loss for binary masks.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        object_valid_mask: [batch_size, num_objects] - mask indicating valid target objects
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs (not used by DICE)
        norm_weight: balance between different normalisation approach (weight = 1 completely normalises the loss per objects)
        sample_weight: Not used by DICE!

    Returns:
        loss: Scalar tensor representing the DICE loss
    """

    assert 0.0 <= norm_weight <= 1.0, f"norm_weight for mask_dice_loss must be between 0 and 1, got {norm_weight}"

    # only condition on valid object masks
    if object_valid_mask is not None:
        pred_logits = pred_logits[object_valid_mask]
        targets = targets[object_valid_mask]

    probs = pred_logits.sigmoid()
    numerator = 2 * (probs * targets).sum(-1)
    denominator = probs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = norm_weight * loss.sum() / len(probs)

    if input_pad_mask is not None:
        probs_pad = probs * input_pad_mask.unsqueeze(1)
        numerator_pad = 2 * (probs_pad * targets).sum(-1)
        denominator_pad = probs_pad.sum(-1) + targets.sum(-1)
        loss_pad = 1 - (numerator_pad + 1) / (denominator_pad + 1)
        loss += (1 - norm_weight) * loss_pad.mean()

    return loss


def mask_dice_cost(pred_logits, targets, input_pad_mask=None, sample_weight=None):
    """
    Compute DICE costs.
    Invalid objects are handled later in the matching process.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs (not used by DICE)
        sample_weight: Not used by DICE!

    Returns:
        cost: [batch_size, num_objects, num_objects] - DICE cost matrix
    """
    assert sample_weight is None
    inputs = pred_logits.sigmoid()

    # apply input padding mask
    if input_pad_mask is not None:
        inputs = inputs * input_pad_mask.unsqueeze(1)

    numerator = 2 * torch.einsum("bnc,bmc->bnm", inputs, targets)
    denominator = inputs.sum(-1).unsqueeze(2) + targets.sum(-1).unsqueeze(1)
    cost = 1 - (numerator + 1) / (denominator + 1)
    return cost


def mask_iou_cost(pred_logits, targets, input_pad_mask=None, eps=1e-6):
    # Apply input padding mask
    probs = pred_logits.sigmoid()
    if input_pad_mask is not None:
        probs = probs * input_pad_mask.unsqueeze(1)

    num_pred = probs.sum(-1).unsqueeze(2)
    num_targets = targets.sum(-1).unsqueeze(1)

    # Context manager necessary to overwrite global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        intersection = torch.einsum("bnc,bmc->bnm", probs, targets)
        cost = 1 - (intersection + eps) / (eps + num_pred + num_targets - intersection)

    return cost


def mask_focal_loss(pred_logits, targets, gamma=2.0, object_valid_mask=None, input_pad_mask=None, norm_weight=1.0, sample_weight=None):
    """
    Compute the focal loss for binary classification.

    Args:
        pred_logits: [batch_size, num_objects] - predicted logits for binary classification
        targets: [batch_size, num_objects] - ground truth class labels
        gamma: Focusing parameter for the focal loss
        object_valid_mask: [batch_size, num_objects] - mask indicating valid target objects
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        norm_weight: balance between different normalisation approach (weight = 1 completely normalises the loss per objects)
        sample_weight: Optional sample weights for each element

    Returns:
        loss: Scalar tensor representing the focal loss
    """

    assert 0.0 <= norm_weight <= 1.0, f"norm_weight for mask_focal_loss must be between 0 and 1, got {norm_weight}"

    if object_valid_mask is not None:
        pred_logits = pred_logits[object_valid_mask]
        targets = targets[object_valid_mask]
        sample_weight = sample_weight[object_valid_mask] if sample_weight is not None else None

    pred = pred_logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets.type_as(pred_logits), weight=sample_weight, reduction="none")

    p_t = pred * targets + (1 - pred) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    loss = norm_weight * loss.mean(1).sum() / len(pred)

    # Apply input padding mask
    if input_pad_mask is not None:
        ce_loss_pad = ce_loss * input_pad_mask.unsqueeze(1)
        pred_pad = pred * input_pad_mask.unsqueeze(1)

        p_t_pad = pred_pad * targets + (1 - pred_pad) * (1 - targets)
        loss_pad = ce_loss_pad * ((1 - p_t_pad) ** gamma)

        # normalise by valid elements such that each mask contributes equally
        valid_counts = input_pad_mask.sum(-1, keepdim=True)
        loss_pad = loss_pad.sum(-1) / valid_counts
        loss += (1 - norm_weight) * loss_pad.mean()

    return loss


def mask_focal_cost(pred_logits, targets, gamma=2.0, input_pad_mask=None, sample_weight=None):
    """
    Compute focal costs for binary masks.
    Invalid objects are handled later in the matching process.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        gamma: Focusing parameter for the focal loss
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Optional sample weights for each element (effectively the focal alpha)

    Returns:
        cost: [batch_size, num_objects, num_objects] - focal cost matrix
    """

    pred = pred_logits.sigmoid()
    focal_pos = ((1 - pred) ** gamma) * F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred), weight=sample_weight, reduction="none")
    focal_neg = (pred**gamma) * F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred), weight=sample_weight, reduction="none")

    # Apply input padding mask
    if input_pad_mask is not None:
        focal_pos = focal_pos * input_pad_mask.unsqueeze(1)
        focal_neg = focal_neg * input_pad_mask.unsqueeze(1)

    # Context manager necessary to overwride global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        cost = torch.einsum("bnc,bmc->bnm", focal_pos, targets) + torch.einsum("bnc,bmc->bnm", focal_neg, (1 - targets))

    return cost


def mask_bce_loss(pred_logits, targets, object_valid_mask=None, input_pad_mask=None, norm_weight=1.0, sample_weight=None):
    """
    Compute the binary cross-entropy loss for binary masks.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        object_valid_mask: [batch_size, num_objects] - mask indicating valid target objects
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        norm_weight: balance between different normalisation approach (weight = 1 completely normalises the loss per objects)
        sample_weight: Optional sample weights for each element.  Recommended to use focal instead.

    Returns:
        loss: Scalar tensor representing the binary cross-entropy loss
    """

    assert 0.0 <= norm_weight <= 1.0, f"norm_weight for mask_bce_loss must be between 0 and 1, got {norm_weight}"

    if object_valid_mask is not None:
        pred_logits = pred_logits[object_valid_mask]
        targets = targets[object_valid_mask]
        sample_weight = sample_weight[object_valid_mask] if sample_weight is not None else None

    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, weight=sample_weight, reduction="none")
    loss = norm_weight * bce_loss.mean(-1).sum() / len(pred_logits)

    # Apply input padding mask
    if input_pad_mask is not None:
        loss_pad = bce_loss * input_pad_mask.unsqueeze(1)

        # normalise by valid elements such that each mask contributes equally
        valid_counts = input_pad_mask.sum(-1, keepdim=True)
        loss_pad = loss_pad.sum(-1) / valid_counts
        loss += (1 - norm_weight) * loss_pad.mean()

    return loss


def mask_bce_cost(pred_logits, targets, input_pad_mask=None, sample_weight=None):
    """
    Compute binary cross-entropy costs for binary masks.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Optional sample weights for each element. Recommended to use focal instead.

    Returns:
        cost: [batch_size, num_objects, num_objects] - binary cross-entropy cost
    """
    pred_logits = torch.clamp(pred_logits, -100, 100)

    pos = F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred_logits), weight=sample_weight, reduction="none")
    neg = F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred_logits), weight=sample_weight, reduction="none")

    # Apply input padding mask
    if input_pad_mask is not None:
        pos = pos * input_pad_mask.unsqueeze(1)
        neg = neg * input_pad_mask.unsqueeze(1)

    # Context manager necessary to overwrite global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        cost = torch.einsum("bnc,bmc->bnm", pos, targets) + torch.einsum("bnc,bmc->bnm", neg, (1 - targets))

    return cost


def kl_div_loss(pred_logits, true, mask=None, weight=None, eps=1e-8):  # noqa: ARG001
    loss = -true * torch.log(pred_logits + eps)
    # if weight is not None:
    #     loss *= weight
    if mask is not None:
        loss = loss[mask]
    return loss.mean()


# Context manager necessary to overwride global autocast to ensure float32 cost is returned
# @torch.autocast(device_type="cuda", enabled=False)
def kl_div_cost(pred_logits, true, eps=1e-8):
    return (-true[:, None, :] * torch.log(pred_logits[:, :, None] + eps)).mean(-1)


def regr_mse_loss(pred, targets):
    return torch.nn.functional.mse_loss(pred, targets, reduction="none")


def regr_smooth_l1_loss(pred, targets):
    return torch.nn.functional.smooth_l1_loss(pred, targets, reduction="none")


def regr_mse_cost(pred, targets):
    return torch.nn.functional.mse_loss(pred.unsqueeze(-2), targets.unsqueeze(-3), reduction="none")


def regr_smooth_l1_cost(pred, targets):
    return torch.nn.functional.smooth_l1_loss(pred.unsqueeze(-2), targets.unsqueeze(-3), reduction="none")


cost_fns = {
    "object_bce": object_bce_cost,
    "object_ce": object_ce_cost,
    "mask_bce": mask_bce_cost,
    "mask_dice": mask_dice_cost,
    "mask_focal": mask_focal_cost,
    "mask_iou": mask_iou_cost,
    "kl_div": kl_div_cost,
}

loss_fns = {
    "object_bce": object_bce_loss,
    "object_ce": object_ce_loss,
    "mask_bce": mask_bce_loss,
    "mask_dice": mask_dice_loss,
    "mask_focal": mask_focal_loss,
    "kl_div": kl_div_loss,
}
