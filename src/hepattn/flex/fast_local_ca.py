import torch
from torch.nn.attention.flex_attention import BlockMask


def _kv_blocks_nonwrap(
    q_blocks: int,
    kv_blocks: int,
    block_size: int,
    window_size: int,
    s: torch.Tensor,
    q_len: int,
    kv_len: int,
    device: str,
    dtype_float: torch.dtype,
):
    """For each query block, compute which KV blocks fall inside a sliding window (no wrap-around).
    This produces a mapping:
      - kv_num_blocks[q]: number of KV blocks visible to query block q
      - kv_indices[q, :kv_num_blocks[q]]: the actual KV block indices
    This helper over-approximates at block granularity; per-token filtering
    is refined later by `mask_mod`.
    """
    # Keep scalars as tensors (avoids graph breaks in torch.compile)
    half_f = torch.tensor(window_size // 2, device=device, dtype=dtype_float)
    kv_len_fm1 = torch.tensor(kv_len - 1, device=device, dtype=dtype_float)

    q_len_t = torch.tensor(q_len, device=device)  # stays integer
    block_size_t = torch.tensor(block_size, device=device)  # stays integer

    zero_f = torch.zeros(q_blocks, device=device, dtype=dtype_float)

    qb = torch.arange(q_blocks, device=device)  # [Q]
    q0 = qb * block_size_t  # first query token in block
    q1 = torch.minimum(q0 + (block_size_t - 1), q_len_t - 1)  # last query token in block

    # Map query-token positions to KV "centers" by proportional scaling:
    q0f = q0.to(dtype_float) * s
    q1f = q1.to(dtype_float) * s
    lo_center = torch.minimum(q0f, q1f)
    hi_center = torch.maximum(q0f, q1f)
    min_center = torch.floor(lo_center)
    max_center = torch.ceil(hi_center)

    # Convert window in tokens to an inclusive token range
    low_token = torch.maximum(min_center - half_f, zero_f)  # Leftmost KV token this block can see
    hi_token = torch.minimum(max_center + half_f, kv_len_fm1)  # Rightmost KV token this block can see

    # Convert token range to block range
    lo_blk = torch.div(low_token, block_size_t, rounding_mode="floor")
    hi_blk = torch.div(hi_token, block_size_t, rounding_mode="floor")

    # Build a [Q, K] boolean mask over KV blocks
    base = torch.arange(kv_blocks, device=device)  # [K]
    base2 = base.unsqueeze(0).expand(q_blocks, kv_blocks)  # [Q,K]
    mask = (base2 >= lo_blk.unsqueeze(1)) & (base2 <= hi_blk.unsqueeze(1))  # [Q,K]

    #  - kv_num_blocks: count of True per row
    #  - kv_indices: compacted indices per row (positions via cumsum trick)
    kv_num_blocks = mask.sum(dim=1)  # [Q]
    pos = (mask.cumsum(dim=1) - 1).masked_fill(~mask, 0)  # [Q,K]
    src = base2.masked_fill(~mask, 0)
    kv_indices = torch.zeros((q_blocks, kv_blocks), device=device, dtype=base.dtype)
    kv_indices.scatter_(dim=1, index=pos, src=src)  # compact along dim=1
    return kv_num_blocks, kv_indices


def _kv_blocks_wrap(
    q_blocks: int,
    kv_blocks: int,
    block_size: int,
    window_size: int,
    s: torch.Tensor,
    q_len: int,
    kv_len: int,
    device: str,
    dtype_float: torch.dtype,
):
    """Same as _kv_blocks_nonwrap but the sliding window can wrap around the end of the KV sequence (circular indexing).
    Handles three row types:
    1) all_rows: window covers the whole KV (mask all blocks).
    2) nonwrap_row: window doesn't cross the end (single interval).
    3) wrap_row: window crosses the end (union of two intervals).
    """
    half_f = torch.tensor(window_size // 2, device=device, dtype=dtype_float)
    block_size_t = torch.tensor(block_size, device=device)  # stays integer
    q_len_t = torch.tensor(q_len, device=device)  # stays integer

    qb = torch.arange(q_blocks, device=device)
    q0 = qb * block_size_t
    q1 = torch.minimum(q0 + (block_size_t - 1), q_len_t - 1)

    q0f = q0.to(dtype_float) * s
    q1f = q1.to(dtype_float) * s
    lo_center = torch.minimum(q0f, q1f)
    hi_center = torch.maximum(q0f, q1f)
    min_center = torch.floor(lo_center)
    max_center = torch.ceil(hi_center)

    low_token = min_center - half_f
    hi_token = max_center + half_f
    span = hi_token - low_token + 1  # window width in tokens (inclusive)

    kv_len_t = torch.tensor(kv_len, device=device, dtype=torch.int32)
    base = torch.arange(kv_blocks, device=device, dtype=torch.int64)
    base2 = base.unsqueeze(0).expand(q_blocks, kv_blocks)

    # If window covers the whole sequence, select all KV blocks
    all_rows = span >= kv_len_t

    kv_len_f = torch.tensor(kv_len, device=device, dtype=dtype_float)

    #  Mod the bounds into [0, kv_len)
    low_mod = torch.remainder(low_token, kv_len_f)
    high_mod = torch.remainder(hi_token, kv_len_f)

    # Identify whether the modulo-interval wraps around
    nonwrap_row = (~all_rows) & (low_mod <= high_mod)
    wrap_row = (~all_rows) & (low_mod > high_mod)

    # Non-wrapping interval: [low_mod, high_mod]
    low_blk_nw = torch.div(low_mod, block_size_t, rounding_mode="floor")
    hi_blk_nw = torch.div(high_mod, block_size_t, rounding_mode="floor")
    mask_nw = (base2 >= low_blk_nw.unsqueeze(1)) & (base2 <= hi_blk_nw.unsqueeze(1))

    # Wrapping interval: [0, high_mod] U [low_mod, kv_len)
    low_block2 = torch.div(low_mod, block_size_t, rounding_mode="floor")
    hi_block1 = torch.div(high_mod, block_size_t, rounding_mode="floor")
    mask_wr = (base2 <= hi_block1.unsqueeze(1)) | (base2 >= low_block2.unsqueeze(1))

    # Row-wise select the correct mask without Python branching
    mask = (all_rows.unsqueeze(1)) | (nonwrap_row.unsqueeze(1) & mask_nw) | (wrap_row.unsqueeze(1) & mask_wr)

    # Pack ragged rows (same trick as in non-wrap)
    kv_num_blocks = mask.sum(dim=1)
    pos = (mask.cumsum(dim=1) - 1).masked_fill(~mask, 0)
    src = base2.masked_fill(~mask, 0)
    kv_indices = torch.zeros((q_blocks, kv_blocks), device=device, dtype=base.dtype)
    kv_indices.scatter_(dim=1, index=pos, src=src)

    return kv_num_blocks, kv_indices


# compile the helpers
# Intentional shadowing: replace original functions with compiled versions
_kv_blocks_nonwrap = torch.compile(_kv_blocks_nonwrap, dynamic=True)  # type: ignore[invalid-assignment]
_kv_blocks_wrap = torch.compile(_kv_blocks_wrap, dynamic=True)  # type: ignore[invalid-assignment]


def build_strided_sliding_window_blockmask(
    *,
    window_size: int,
    stride: float,
    q_len: int,
    kv_len: int,
    device: str,
    wrap: bool,
    block_size: int = 128,
    dtype_float: torch.dtype = torch.float32,
) -> BlockMask:
    """Build a BlockMask for Flex Attention implementing a strided sliding window.
    High level:
      1) At *block* granularity: pick which KV blocks each Q block can see
         (fast, coarse; possibly an over-approximation).
      2) At *token* granularity: `mask_mod` filters inside those blocks so the
         final mask exactly matches a window of width `window_size` centered at
         round(q_idx * stride). If `wrap=True`, the window wraps circularly.

    Notes:
      - window_size must be even so the window is symmetric around the center.
      - `stride` controls how the window center moves as q_idx increases.
      - The compiled helpers scale by kv_len/q_len to get a safe block envelope;
        `mask_mod` does the precise per-token check using `stride`.

    Raises:
        ValueError: If window_size is odd.
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    # Number of query/KV blocks (ceil division)
    q_blocks = (q_len + block_size - 1) // block_size
    kv_blocks = (kv_len + block_size - 1) // block_size
    stride_t = torch.tensor(stride, device=device, dtype=dtype_float)

    # Compute the block-level KV visibility (coarse envelope)
    if wrap:
        kv_num_blocks, kv_indices = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride_t, q_len, kv_len, device, dtype_float)
    else:
        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride_t, q_len, kv_len, device, dtype_float)

    # Flex Attention expects [B, H, Q_blocks, ...]; we use singleton B=H=1
    kv_num_blocks = kv_num_blocks.unsqueeze(0).unsqueeze(0)  # [1,1,Q_blocks]
    kv_indices = kv_indices.unsqueeze(0).unsqueeze(0)  # [1,1,Q_blocks,kv_blocks]

    # Scalars as tensors for compiled mask_mod
    kv_len_t = torch.as_tensor(kv_len, device=device).reshape(())

    # Per-token refinement: given (q_idx, kv_idx) decide if it's inside the
    # strided window. Called by Flex Attention during block processing.
    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
        # Center of the window for this query token
        q_center = torch.round(q_idx * stride_t)
        if not wrap:
            return (kv_idx - q_center).abs() <= window_size // 2
        diagonal = (kv_idx - q_center).abs() <= window_size // 2
        wrap_left = (kv_idx - q_center + kv_len_t).abs() <= window_size // 2
        wrap_right = (kv_idx - q_center - kv_len_t).abs() <= window_size // 2
        return diagonal | wrap_left | wrap_right

    # Build the final BlockMask. seq_lengths makes sure the mask trims to
    # the exact (q_len, kv_len) even when the last block is partial.
    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(q_len, kv_len),  # make sure mask is right size (otherwise shape is num_blocks * block_size)
    )
