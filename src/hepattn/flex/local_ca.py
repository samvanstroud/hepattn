import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def sliding_window_mask_strided(
    window_size: int,
    stride: Tensor,
    **kwargs,
) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    def mask_mod(b, h, q_idx, kv_idx):
        q_center = torch.round(q_idx * stride)
        return (kv_idx - q_center).abs() <= window_size // 2

    return mask_mod


def sliding_window_mask_strided_wrapped(window_size: int, stride: Tensor, kv_len: Tensor) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    def mask_mod(b, h, q_idx, kv_idx):
        # b = 0, h = 0 here
        if stride.ndim != 0:
            raise ValueError("stride must be a scalar tensor")
        if kv_len is not None and not torch.is_tensor(kv_len):
            raise TypeError("kv_len must be a Tensor")

        dev = q_idx.device

        off = kv_len.reshape(()).to(device=dev)  # offset should be equal to number of hits

        q_center = torch.round(q_idx * stride.reshape(()))

        diagonal = (kv_idx - q_center).abs() <= window_size // 2
        wrap_left = (kv_idx - q_center + off).abs() <= window_size // 2
        wrap_right = (kv_idx - q_center - off).abs() <= window_size // 2

        return diagonal | wrap_left | wrap_right

    return mask_mod
