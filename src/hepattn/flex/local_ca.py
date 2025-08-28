import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def sliding_window_mask_strided(
    window_size: int,
    stride: Tensor,
    kv_len: Tensor | None = None,
) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    def mask_mod(b, h, q_idx, kv_idx):
        dev = q_idx.device
        window = torch.tensor((window_size // 2), device=dev)
        q_center = torch.round(q_idx * stride)
        return (kv_idx - q_center).abs() <= window

    return mask_mod


def sliding_window_mask_strided_wrapped(window_size: int, stride, kv_len=None) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    def mask_mod(b, h, q_idx, kv_idx):
        # b = 0, h = 0 here
        assert stride.ndim == 0
        if kv_len is not None:
            assert torch.is_tensor(kv_len)

        dev = q_idx.device
        window = torch.tensor(window_size // 2, device=dev)

        kvl = torch.tensor([1], device=dev) if kv_len is None else kv_len.to(device=dev)
        off = kvl.reshape(())  # offset should be equal to number of hits

        q_center = torch.round(q_idx * stride.reshape(()))

        diagonal = (kv_idx - q_center).abs() <= window
        wrap_left = (kv_idx - q_center + off).abs() <= window
        wrap_right = (kv_idx - q_center - off).abs() <= window

        return diagonal | wrap_left | wrap_right

    return mask_mod
