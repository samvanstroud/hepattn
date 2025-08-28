import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def sliding_window_mask_strided(
    window_size: int,
    stride: Tensor,
) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    def mask_mod(b, h, q_idx, kv_idx):
        dev = q_idx.device
        # Create tensors explicitly to avoid reuse issues
        half_val = window_size // 2
        half = torch.tensor(half_val, device=dev)

        q_center = q_idx * stride
        if torch.is_floating_point(q_center):
            q_center = torch.round(q_center)

        return (kv_idx - q_center).abs() <= half

    return mask_mod


def sliding_window_mask_strided_wrapped(window_size: int, stride, kv_len=None) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    def mask_mod(b, h, q_idx, kv_idx):
        assert stride.ndim == 1
        assert torch.is_tensor(kv_len)

        dev = q_idx.device
        # per-head stride: pick the one for this head index
        s_scalar = stride[h].reshape(())
        half = torch.tensor(window_size // 2, device=dev)

        kvl = torch.tensor([1], device=dev) if kv_len is None else kv_len.to(device=dev)
        assert kv_len.ndim == 1
        off = kvl[b].reshape(())

        q_center = q_idx * s_scalar
        if torch.is_floating_point(q_center):
            q_center = torch.round(q_center)

        diagonal = (kv_idx - q_center).abs() <= half
        wrap_left = (kv_idx - q_center + off).abs() <= half
        wrap_right = (kv_idx - q_center - off).abs() <= half
        return diagonal | wrap_left | wrap_right

    return mask_mod
