import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature, create_block_mask

create_block_mask = torch.compile(create_block_mask, dynamic=True)


def sliding_window_mask_strided(window_size: int, stride: float, q_len: Tensor, kv_len: Tensor, dev) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    stride_val = torch.tensor(stride, device=dev)

    def mask_mod(b, h, q_idx, kv_idx):
        # b = 0, h = 0 here
        if kv_len is not None and not torch.is_tensor(kv_len):
            raise TypeError("kv_len must be a Tensor")

        q_center = torch.round(q_idx * stride_val)

        diagonal = (kv_idx - q_center).abs() <= window_size // 2

        return diagonal

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=dev)


def sliding_window_mask_strided_wrapped(window_size: int, stride: float, q_len: Tensor, kv_len: Tensor, dev) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    stride_val = torch.tensor(stride, device=dev)

    def mask_mod(b, h, q_idx, kv_idx):
        # b = 0, h = 0 here
        if kv_len is not None and not torch.is_tensor(kv_len):
            raise TypeError("kv_len must be a Tensor")

        off = kv_len.reshape(()).to(device=dev)  # offset should be equal to number of hits
        # Use scalar multiplication directly to avoid tensor creation issues
        q_center = torch.round(q_idx * stride_val)

        diagonal = (kv_idx - q_center).abs() <= window_size // 2
        wrap_left = (kv_idx - q_center + off).abs() <= window_size // 2
        wrap_right = (kv_idx - q_center - off).abs() <= window_size // 2

        return diagonal | wrap_left | wrap_right

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=dev)
