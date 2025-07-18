import torch
from torch import BoolTensor, Tensor


def build_target_masks(object_ids: Tensor, input_ids: Tensor, shuffle: bool = False) -> Tensor:
    """Get the truth masks from the object ids and input_ids.

    The difference between this function and mask_from_indices is that mask_from_indices
    expects the indices to start from zero, and here we match based on arbitrary IDs, such as barcodes.

    Args:
        object_ids (Tensor): The unique ids of the truth object labels.
        input_ids (Tensor): The ids of the per-input labels.
        shuffle (bool): Whether to shuffle the object_ids. Defaults to False.

    Returns:
        Tensor: The truth masks.
    """
    # shuffling doesn't seem to be needed here
    if shuffle:
        object_ids = object_ids[:, torch.randperm(object_ids.shape[1])]
    object_ids[object_ids == -1] = -999
    masks = input_ids.unsqueeze(-2) == object_ids.unsqueeze(-1)
    return masks


def mask_from_indices(indices: Tensor, num_masks: int | None = None) -> BoolTensor:
    """Convert a dense index tensor to a sparse bool mask.

    Indices are arbitrary and start from 0.

    Examples:
        [0, 1, 1] -> [[True, False, False], [False, True, True]]
        [0, 1, 2]] -> [[True, False, False], [False, True, False], [False, False, True]]

    Args:
        indices (Tensor): The dense indices.
        num_masks (int | None): The maximum number of masks. If None, inferred from indices.

    Returns:
        BoolTensor: The sparse mask.
    """
    assert indices.ndim == 1 or indices.ndim == 2, "indices must be 1D for single sample or 2D for batch"
    if num_masks is None:
        num_masks = indices.max() + 1
    else:
        assert num_masks > indices.max(), "num_masks must be greater than the maximum value in indices"

    indices = torch.as_tensor(indices)
    kwargs = {"dtype": torch.bool, "device": indices.device}
    if indices.ndim == 1:
        mask = torch.zeros((num_masks, indices.shape[-1]), **kwargs)
        mask[indices, torch.arange(indices.shape[-1])] = True
        mask.transpose(0, 1)[indices < 0] = False  # handle negative indices
    else:
        mask = torch.zeros((indices.shape[0], num_masks, indices.shape[-1]), **kwargs)
        mask[torch.arange(indices.shape[0]).unsqueeze(-1), indices, torch.arange(indices.shape[-1])] = True
        mask.transpose(1, 2)[indices < 0] = False  # handle negative indices

    return mask


def indices_from_mask(mask: BoolTensor, noindex: int = -1) -> Tensor:
    """Convert a sparse bool mask to a dense index tensor.

    Indices are arbitrary and start from 0.

    Examples:
        [[True, False, False], [False, True, True]] -> [0, 1, 1]

    Args:
        mask (BoolTensor): The sparse mask.
        noindex (int): The value to use for no index. Defaults to -1.

    Returns:
        Tensor: The dense indices.

    Raises:
        ValueError: If mask is not 2D for single sample or 3D for batch.
    """
    mask = torch.as_tensor(mask)
    kwargs = {"dtype": torch.long, "device": mask.device}
    if mask.ndim == 2:
        indices = torch.ones(mask.shape[-1], **kwargs) * noindex
        nonzero_idx = torch.where(mask)
        indices[nonzero_idx[1]] = nonzero_idx[0]
    elif mask.ndim == 3:
        indices = torch.ones((mask.shape[0], mask.shape[-1]), **kwargs) * noindex
        nonzero_idx = torch.where(mask)
        indices[nonzero_idx[0], nonzero_idx[2]] = nonzero_idx[1]
    else:
        raise ValueError("mask must be 2D for single sample or 3D for batch")

    # ensure indices start from 0
    valid_indices = indices[indices >= 0]
    if len(valid_indices) > 0:
        indices -= valid_indices.min()
    indices[indices < 0] = noindex

    return indices


def sanitise_mask(
    mask: BoolTensor,
    input_pad_mask: BoolTensor | None = None,
    object_class_preds: Tensor | None = None,
) -> BoolTensor:
    """Sanitise predicted masks by removing padded inputs and null class predictions.

    Args:
        mask (BoolTensor): The predicted mask.
        input_pad_mask (BoolTensor | None): The input pad mask, where a value of True represents a padded input.
            Defaults to None.
        object_class_preds (Tensor | None): Object class predictions. Defaults to None.

    Returns:
        BoolTensor: The sanitised mask.
    """
    if input_pad_mask is not None:
        mask.transpose(1, 2)[input_pad_mask] = False
    if object_class_preds is not None:
        pred_null = object_class_preds.argmax(-1) == object_class_preds.shape[-1] - 1
        mask[pred_null] = False
    return mask


def sigmoid_mask(
    mask_logits: Tensor,
    threshold: float = 0.5,
    **kwargs,
) -> BoolTensor:
    """Get a mask by thresholding the mask logits.

    Args:
        mask_logits (Tensor): The mask logits.
        threshold (float): The threshold. Defaults to 0.5.
        **kwargs: Additional keyword arguments to pass to sanitise_mask.

    Returns:
        BoolTensor: The thresholded mask.
    """
    mask = mask_logits.sigmoid() > threshold
    mask = sanitise_mask(mask, **kwargs)
    return mask


def argmax_mask(
    mask_logits: Tensor,
    weighted: bool = False,
    **kwargs,
) -> BoolTensor:
    """Get a mask by taking the argmax of the mask logits.

    Args:
        mask_logits (Tensor): The mask logits.
        weighted (bool): Weight logits according to object class confidence, as in MaskFormer.
            Defaults to False.
        **kwargs: Additional keyword arguments to pass to sanitise_mask.

    Returns:
        BoolTensor: The argmax mask.

    Raises:
        ValueError: If weighted argmax is requested but object_class_preds is not provided.
    """
    if weighted and kwargs.get("object_class_preds") is None:
        raise ValueError("weighted argmax requires object_class_preds")

    if not weighted:
        idx = mask_logits.argmax(-2)
    else:
        confidence = kwargs["object_class_preds"].max(-1)[0].unsqueeze(-1)
        assert confidence.min() >= 0.0 and confidence.max() <= 1.0, "confidence must be between 0 and 1"
        idx = (mask_logits.softmax(-2) * confidence).argmax(-2)
    mask = mask_from_indices(idx, num_masks=mask_logits.shape[-2])
    mask = sanitise_mask(mask, **kwargs)
    return mask


def mask_from_logits(
    logits: Tensor,
    mode: str,
    input_pad_mask: BoolTensor | None = None,
    object_class_preds: Tensor | None = None,
) -> BoolTensor:
    """Convert logits to masks using the specified mode.

    Args:
        logits (Tensor): The input logits.
        mode (str): The mode to use for conversion. Must be one of {"sigmoid", "argmax", "weighted_argmax"}.
        input_pad_mask (BoolTensor | None): The input pad mask. Defaults to None.
        object_class_preds (Tensor | None): Object class predictions. Defaults to None.

    Returns:
        BoolTensor: The converted mask.

    Raises:
        ValueError: If mode is not one of the supported modes.
    """
    modes = {"sigmoid", "argmax", "weighted_argmax"}
    if mode == "sigmoid":
        return sigmoid_mask(logits, input_pad_mask=input_pad_mask, object_class_preds=object_class_preds)
    if mode == "argmax":
        return argmax_mask(logits, input_pad_mask=input_pad_mask, object_class_preds=object_class_preds)
    if mode == "weighted_argmax":
        return argmax_mask(logits, weighted=True, input_pad_mask=input_pad_mask, object_class_preds=object_class_preds)
    raise ValueError(f"mode must be one of {modes}")


def mask_effs_purs(m_pred: BoolTensor, m_tgt: BoolTensor) -> tuple[Tensor, Tensor]:
    """Calculate efficiency and purity for each mask.

    Args:
        m_pred (BoolTensor): The predicted masks.
        m_tgt (BoolTensor): The target masks.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing efficiency and purity tensors.
    """
    eff = (m_pred & m_tgt).sum(-1) / m_tgt.sum(-1)
    pur = (m_pred & m_tgt).sum(-1) / m_pred.sum(-1)
    return eff, pur


def mask_eff_pur(m_pred: BoolTensor, m_tgt: BoolTensor, flat: bool = False, reduce: bool = False) -> tuple[Tensor, Tensor]:
    """Calculate efficiency and purity metrics.

    Args:
        m_pred (BoolTensor): The predicted masks.
        m_tgt (BoolTensor): The target masks.
        flat (bool): If True, calculate per assignment metric (edgewise). If False, calculate per object metric.
            Defaults to False.
        reduce (bool): If True, reduce to mean values (only applies when flat=False). Defaults to False.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing efficiency and purity values.
    """
    if flat:
        # per assignment metric (i.e. edgewise)
        eff = (m_pred & m_tgt).sum() / m_tgt.sum()
        pur = (m_pred & m_tgt).sum() / m_pred.sum()
    else:
        # per object metric (nanmean avoids invalid indices)
        eff, pur = mask_effs_purs(m_pred, m_tgt)
        if reduce:
            eff, pur = eff.nanmean(), pur.nanmean()
    return eff, pur


def reco_metrics(
    pred_mask: BoolTensor,
    tgt_mask: BoolTensor,
    pred_valid: Tensor | None = None,
    reduce: bool = False,
    min_recall: float = 1.0,
    min_purity: float = 1.0,
    min_constituents: int = 0,
) -> tuple[Tensor, Tensor]:
    """Calculate the efficiency and purity of the predicted objects.

    Args:
        pred_mask (BoolTensor): The predicted masks.
        tgt_mask (BoolTensor): The target masks.
        pred_valid (Tensor | None): Valid prediction mask. If None, inferred from pred_mask. Defaults to None.
        reduce (bool): Whether to reduce to mean values. Defaults to False.
        min_recall (float): Minimum recall threshold. Defaults to 1.0.
        min_purity (float): Minimum purity threshold. Defaults to 1.0.
        min_constituents (int): Minimum number of constituents. Defaults to 0.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing efficiency and fake rate metrics.
    """
    if pred_valid is None:
        pred_valid = pred_mask.sum(-1) > 0
    else:
        pred_valid = pred_valid.clone()
        pred_valid &= pred_mask.sum(-1) > 0

    eff, pur = mask_effs_purs(pred_mask, tgt_mask)
    pass_cuts = (eff >= min_recall) & (pur >= min_purity)

    if min_constituents > 0:
        pred_valid &= pred_mask.sum(-1) >= min_constituents

    eff = pred_valid & pass_cuts
    fake = pred_valid & ~pass_cuts

    if reduce:
        valid_tgt = tgt_mask.sum(-1) > 0
        eff = eff[valid_tgt].float().mean()
        fake = fake[pred_valid].float().mean()

    return eff, fake


def topk_attn(attn_scores: Tensor, k: int, dim: int = -1) -> BoolTensor:
    """Keep only the topk scores in each row of the attention matrix.

    Args:
        attn_scores (Tensor): The attention scores.
        k (int): The number of top scores to keep.
        dim (int): The dimension to apply top-k along. Defaults to -1.

    Returns:
        BoolTensor: A boolean mask with `True` for the top-k scores.
    """
    _, topk_indices = attn_scores.topk(k, dim=dim)
    zeros = torch.zeros_like(attn_scores, dtype=bool)
    src = torch.ones_like(topk_indices, dtype=bool).expand_as(topk_indices)
    mask = torch.scatter(zeros, dim=dim, index=topk_indices, src=src)
    return mask
