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
        
        # Ensure stride is a tensor on the right device and dtype
        if torch.is_tensor(stride):
            s = stride.to(device=dev)
        else:
            # pick a float dtype already in use if any; else default dtype
            base_float = torch.get_default_dtype()
            if torch.is_floating_point(q_idx):
                base_float = q_idx.dtype
            elif torch.is_floating_point(kv_idx):
                base_float = kv_idx.dtype
            s = torch.tensor(stride, device=dev, dtype=base_float)
        
        # Compute a common dtype for (q_idx, kv_idx, s)
        work_dtype = torch.promote_types(torch.promote_types(q_idx.dtype, kv_idx.dtype), s.dtype)
        q  = q_idx.to(dtype=work_dtype)
        kv = kv_idx.to(dtype=work_dtype)
        s  = s.to(dtype=work_dtype)

        # Create tensors explicitly to avoid reuse issues
        half_val = window_size // 2
        half = torch.tensor(half_val, device=dev, dtype=work_dtype)

        q_center = q * s
        if torch.is_floating_point(q_center):
            q_center = torch.round(q_center)

        diagonal   = (kv - q_center).abs() <= half
        wrap  = (kv - q_center).abs() <= half
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
        # q_idx/kv_idx may be 0-D under vmap; don't rely on their .shape
        dev = q_idx.device

        # ----- stride -> scalar for this (b,h,...) call -----
        # accept python numbers, list/tuple, or tensors
        if isinstance(stride, (list, tuple)):
            s = torch.as_tensor(stride, device=dev)
        elif torch.is_tensor(stride):
            s = stride.to(device=dev)
        else:
            # choose a float dtype observed in inputs if possible
            base_float = torch.get_default_dtype()
            if torch.is_floating_point(q_idx):
                base_float = q_idx.dtype
            elif torch.is_floating_point(kv_idx):
                base_float = kv_idx.dtype
            s = torch.tensor(stride, device=dev, dtype=base_float)

        # ----- choose common working dtype -----
        work_dtype = torch.promote_types(torch.promote_types(q_idx.dtype, kv_idx.dtype), s.dtype)

        q  = q_idx.to(dtype=work_dtype)
        kv = kv_idx.to(dtype=work_dtype)

        # Reduce stride to a scalar (0-D) for this head if needed
        s = s.to(dtype=work_dtype)
        if s.ndim == 0:
            s_scalar = s.reshape(())
        elif s.ndim == 1:
            # per-head stride: pick the one for this head index
            s_scalar = s[h.to(torch.long)].to(dtype=work_dtype).reshape(())
        else:
            raise ValueError("stride must be scalar or 1-D (per-head).")

        half = torch.tensor(window_size // 2, device=dev, dtype=work_dtype)

        # ----- kv_len -> scalar for this batch if provided -----
        if kv_len is None:
            off = torch.tensor(10**9, device=dev, dtype=work_dtype)  # effectively disable wrap
            off = off.reshape(())
        else:
            if isinstance(kv_len, (list, tuple)):
                kvl = torch.as_tensor(kv_len, device=dev, dtype=work_dtype)
            elif torch.is_tensor(kv_len):
                kvl = kv_len.to(device=dev, dtype=work_dtype)
            else:
                kvl = torch.tensor(int(kv_len), device=dev, dtype=work_dtype)

            if kvl.ndim == 0:
                off = kvl.reshape(())
            else:
                # Accept (B,), (B,1), (B,1,1), ... : flatten then select by batch index
                kvl_flat = kvl.reshape(-1)
                off = kvl_flat[b.to(torch.long)].reshape(())

        # ----- compute mask (all ops broadcast, return no extra dims) -----
        q_center = q * s_scalar
        if torch.is_floating_point(q_center):
            q_center = torch.round(q_center)

        diagonal   = (kv - q_center).abs() <= half
        wrap_left  = (kv - q_center + off).abs() <= half
        wrap_right = (kv - q_center - off).abs() <= half
        return diagonal | wrap_left | wrap_right

    return mask_mod
