import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature
from torch.nn.functional import pad


def sliding_window_mask(window_size: int) -> _mask_mod_signature:
    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        return (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)

    return mask_mod


def sliding_window_mask_wrapped(window_size: int, q_len: Tensor) -> _mask_mod_signature:
    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        diagonal = (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)
        wrap = ((q_idx - kv_idx + q_len[0]) <= window_size // 2) | ((kv_idx - q_idx + q_len[0]) <= window_size // 2)
        return diagonal | wrap

    return mask_mod

def sliding_window_mask_strided(
    window_size: int, stride: Tensor,
) -> _mask_mod_signature:
    """
    Creates a sliding window mask with striding and wrapping support.

    Args:
        window_size: Size of the attention window (must be even)
        stride: Stride factor for sparse attention patterns
        kv_len: Sequence length tensor for wrapping

    Returns:
        A function that computes attention masks with striding and wrapping
    """
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

        diagonal   = (kv_idx - q_center).abs() <= half
        wrap  = (kv_idx - q_center).abs() <= half
        return diagonal | wrap

    return mask_mod


def sliding_window_mask_strided_wrapped(
    window_size: int, stride, kv_len=None
) -> _mask_mod_signature:
    """
    Sliding window with striding + circular wrapping.

    Args:
        window_size: even int
        stride: scalar (int/float/0-D tensor) or per-head 1-D tensor of length H
        kv_len: None (disable wrap), scalar (int/0-D tensor), or per-batch lengths
                shaped (B,), (B,1), (B,1,1), etc.
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    def mask_mod(b, h, q_idx, kv_idx):

        assert stride.ndim == 1
        assert torch.is_tensor(kv_len)

        dev = q_idx.device
        # per-head stride: pick the one for this head index
        s_scalar = stride[h].reshape(())
        half = torch.tensor(window_size // 2, device=dev)

        if kv_len is None:
            kvl = torch.tensor([1], device=dev)  # effectively disable wrap
        else:
            kvl = kv_len.to(device=dev)
        assert kv_len.ndim == 1
        off = kvl[b].reshape(())

        q_center = q_idx * s_scalar
        if torch.is_floating_point(q_center):
            q_center = torch.round(q_center)

        diagonal   = (kv_idx - q_center).abs() <= half
        wrap_left  = (kv_idx - q_center + off).abs() <= half
        wrap_right = (kv_idx - q_center - off).abs() <= half
        return diagonal | wrap_left | wrap_right

    return mask_mod