import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature, create_block_mask


@torch.compile(dynamic=True)
def sliding_window_mask_strided(window_size: int, stride: float, q_len: int, kv_len: int, dev) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    stride_val = torch.tensor(stride, device=dev)

    def mask_mod(b, h, q_idx, kv_idx):
        # b = 0, h = 0 here
        q_center = torch.round(q_idx * stride_val)
        return (kv_idx - q_center).abs() <= window_size // 2

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=dev)

@torch.compile(dynamic=True)
def sliding_window_mask_strided_wrapped(window_size: int, stride: float, q_len: int, kv_len: int, dev) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    stride_val = torch.tensor(stride, device=dev)
    kv_len_t = torch.as_tensor(kv_len, device=dev).reshape(())

    def mask_mod(b, h, q_idx, kv_idx):
        # b = 0, h = 0 here
        q_center = torch.round(q_idx * stride_val)

        diagonal = (kv_idx - q_center).abs() <= window_size // 2
        wrap_left = (kv_idx - q_center + kv_len_t).abs() <= window_size // 2
        wrap_right = (kv_idx - q_center - kv_len_t).abs() <= window_size // 2

        return diagonal | wrap_left | wrap_right

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=dev)
