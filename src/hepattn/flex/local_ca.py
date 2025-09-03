import torch
from torch.nn.attention.flex_attention import BlockMask, _mask_mod_signature, create_block_mask

# Intentionally shadow the imported function with a compiled version
create_block_mask: _mask_mod_signature = torch.compile(create_block_mask, dynamic=True)


def sliding_window_mask_strided(
    window_size: int,
    stride: float,
    q_len: int,
    kv_len: int,
    device: str,
) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    stride_val = torch.tensor(stride, device=device)

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
        # b = 0, h = 0 here
        q_center = torch.round(q_idx * stride_val)
        return (kv_idx - q_center).abs() <= window_size // 2

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)


def sliding_window_mask_strided_wrapped(
    window_size: int,
    stride: float,
    q_len: int,
    kv_len: int,
    device: str,
) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    stride_val = torch.tensor(stride, device=device)
    kv_len_t = torch.as_tensor(kv_len, device=device).reshape(())

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
        # b = 0, h = 0 here
        q_center = torch.round(q_idx * stride_val)

        diagonal = (kv_idx - q_center).abs() <= window_size // 2
        wrap_left = (kv_idx - q_center + kv_len_t).abs() <= window_size // 2
        wrap_right = (kv_idx - q_center - kv_len_t).abs() <= window_size // 2

        return diagonal | wrap_left | wrap_right

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)


def transpose_blockmask(bm: BlockMask, *, q_tokens: int, kv_tokens: int, dev: str) -> BlockMask:
    """Exact transpose of a FlexAttention BlockMask by transposing the mask function.

    Args:
        bm: forward BlockMask (built with Q_LEN=q_tokens, KV_LEN=kv_tokens)
        q_tokens: original forward Q token length
        kv_tokens: original forward KV token length
        dev: device to build the transposed mask on
    """
    orig_mod = bm.mask_mod

    # New queries are old keys; new keys are old queries.
    def mask_mod_t(b, h, q_idx, kv_idx):
        # Call the original predicate with swapped indices.
        return orig_mod(b, h, kv_idx, q_idx)

    return create_block_mask(
        mask_mod_t,
        B=None,
        H=None,
        Q_LEN=kv_tokens,  # new queries = old KV tokens
        KV_LEN=q_tokens,  # new keys    = old Q tokens
        device=dev,
    )
