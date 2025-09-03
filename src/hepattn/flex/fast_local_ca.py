import torch
from torch.nn.attention.flex_attention import BlockMask


def build_strided_sliding_window_blockmask(
    *,
    window_size: int,
    stride: float,
    q_len: int,
    kv_len: int,
    device: str,
    wrap: bool,
    block_size: int = 128,
) -> BlockMask:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    q_blocks = (q_len + block_size - 1) // block_size  # How many query blocks (ceiling division)
    kv_blocks = (kv_len + block_size - 1) // block_size  # How many KV blocks (ceiling division)

    half_window = window_size // 2

    # for each query block, compute the span of the KV tokens it can attend to
    # store the indices of the KV blocks that intersect with the query block
    compute_blocks = compute_intersecting_blocks_wrap if wrap else compute_intersecting_blocks
    indices, num_blocks = compute_blocks(q_blocks, kv_blocks, stride, half_window, block_size, q_len, kv_len, device)

    # Expand dimensions for broadcasting [None, None, Q_BLOCKS, KV_BLOCKS]
    indices = indices.unsqueeze(0).unsqueeze(0) # [1,1,Q_blocks,Kmax]
    num_blocks = num_blocks.unsqueeze(0).unsqueeze(0) # [1,1,Q_blocks]


    stride_val = torch.tensor(stride, device=device)

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
        q_center = torch.round(q_idx * stride_val)
        if not wrap:
            return (kv_idx - q_center).abs() <= window_size // 2
        kv_len_t = torch.as_tensor(kv_len, device=device).reshape(())
        diagonal = (kv_idx - q_center).abs() <= window_size // 2
        wrap_left = (kv_idx - q_center + kv_len_t).abs() <= window_size // 2
        wrap_right = (kv_idx - q_center - kv_len_t).abs() <= window_size // 2
        return diagonal | wrap_left | wrap_right

    # Do NOT pass full_kv_* or any q_* unless they are *actually* full/needed.
    return BlockMask.from_kv_blocks(
        kv_num_blocks=num_blocks.to(torch.int32),
        kv_indices=indices.to(torch.int32),
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,   # optional but nice to carry for exactness
    )

def compute_intersecting_blocks(
    q_blocks: int, kv_blocks: int, stride: float, half_window: int, block_size: int, q_len: int, kv_len: int, device: str
):
    """Compute the indices of the KV blocks that intersect with the query block."""
    indices = torch.zeros((q_blocks, kv_blocks), dtype=torch.int32, device=device)
    num_blocks = torch.zeros((q_blocks,), dtype=torch.int32, device=device)

    for qb in range(q_blocks):
        q0 = qb * block_size  # First query token in the block
        q1 = min(q_len - 1, (qb + 1) * block_size - 1)  # Last query token in the block

        min_center = round(q0 * stride)  # Center of first query in the block
        max_center = round(q1 * stride)  # Center of last query in the block

        lo_tok = min_center - half_window  # Leftmost KV token this block can see
        hi_tok = max_center + half_window  # Rightmost KV token this block can see

        lo_tok_clamped = max(0, lo_tok)  # Don't go below 0
        hi_tok_clamped = min(kv_len - 1, hi_tok)  # Don't go above kv_len - 1

        if lo_tok_clamped > hi_tok_clamped:
            nb = 0  # Empty
        else:
            lo_blk = lo_tok_clamped // block_size  # Convert to block indices
            hi_blk = hi_tok_clamped // block_size
            nb = hi_blk - lo_blk + 1  # Number of KV blocks to include
            indices[qb, :nb] = torch.arange(lo_blk, hi_blk + 1, dtype=torch.int32, device=device)
        num_blocks[qb] = nb
    return indices, num_blocks


def compute_intersecting_blocks_wrap(
    q_blocks: int, kv_blocks: int, stride: float, half_window: int, block_size: int, q_len: int, kv_len: int, device: str
):
    """Compute the indices of the KV blocks that intersect with the query block."""
    indices = torch.zeros((q_blocks, kv_blocks), dtype=torch.int32, device=device)
    num_blocks = torch.zeros((q_blocks,), dtype=torch.int32, device=device)

    for qb in range(q_blocks):
        q0 = qb * block_size
        q1 = min(q_len - 1, (qb + 1) * block_size - 1)

        min_center = round(q0 * stride)
        max_center = round(q1 * stride)

        lo_tok = min_center - half_window
        hi_tok = max_center + half_window

        total_span = hi_tok - lo_tok + 1
        if total_span >= kv_len:
            # If the span covers the entire sequence, include all blocks
            nb = kv_blocks
            indices[qb, :nb] = torch.arange(0, kv_blocks, dtype=torch.int32, device=device)
            num_blocks[qb] = nb
        else:
            lo_mod = lo_tok % kv_len
            hi_mod = hi_tok % kv_len

            if lo_mod <= hi_mod:
                # Simple case: span doesn't wrap
                lo_blk = lo_mod // block_size
                hi_blk = hi_mod // block_size
                nb = hi_blk - lo_blk + 1
                indices[qb, :nb] = torch.arange(lo_blk, hi_blk + 1, dtype=torch.int32, device=device)
                num_blocks[qb] = nb
            else:
                # Wrapped case: two separate intervals
                # [0, hi_mod] and [lo_mod, kv_len-1]
                lo_blk2 = lo_mod // block_size
                hi_blk2 = (kv_len - 1) // block_size
                lo_blk1 = 0
                hi_blk1 = hi_mod // block_size

                nb1 = hi_blk1 - lo_blk1 + 1
                nb2 = hi_blk2 - lo_blk2 + 1
                nb = nb1 + nb2

                # Fill both intervals
                indices[qb, :nb1] = torch.arange(lo_blk1, hi_blk1 + 1, dtype=torch.int32, device=device)
                indices[qb, nb1:nb] = torch.arange(lo_blk2, hi_blk2 + 1, dtype=torch.int32, device=device)
                num_blocks[qb] = nb
    return indices, num_blocks
