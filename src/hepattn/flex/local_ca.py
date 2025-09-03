import torch
from torch.nn.attention.flex_attention import BlockMask, _mask_mod_signature, create_block_mask


def sliding_window_mask_strided(window_size: int, stride: float, q_len: int, kv_len: int, compile: bool = False, device: str | torch.device) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    stride_val = torch.tensor(stride, device=device)

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
        # b = 0, h = 0 here
        q_center = torch.round(q_idx * stride_val)
        return (kv_idx - q_center).abs() <= window_size // 2
    
    block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)
    
    if compile:
        block_mask = torch.compile(block_mask)
        
    return 


@torch.compile(dynamic=True)
def sliding_window_mask_strided_wrapped(window_size: int, stride: float, q_len: int, kv_len: int, dev) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    stride_val = torch.tensor(stride, device=dev)
    kv_len_t = torch.as_tensor(kv_len, device=dev).reshape(())

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
        # b = 0, h = 0 here
        q_center = torch.round(q_idx * stride_val)

        diagonal = (kv_idx - q_center).abs() <= window_size // 2
        wrap_left = (kv_idx - q_center + kv_len_t).abs() <= window_size // 2
        wrap_right = (kv_idx - q_center - kv_len_t).abs() <= window_size // 2

        return diagonal | wrap_left | wrap_right

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=dev)


def transpose_blockmask(bm: BlockMask, *, q_tokens: int, kv_tokens: int, device=None) -> BlockMask:
    """Exact transpose of a FlexAttention BlockMask by transposing the mask function.

    Args:
        bm: forward BlockMask (built with Q_LEN=q_tokens, KV_LEN=kv_tokens)
        q_tokens: original forward Q token length
        kv_tokens: original forward KV token length
        device: torch.device to build the transposed mask on (defaults to cpu if None)
    """
    orig_mod = bm.mask_mod
    dev = device if device is not None else getattr(bm, "device", "cpu")

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
