import torch
from torch.nn.attention.flex_attention import BlockMask

# ---------------------------
# Compiled tensor-only helpers
# ---------------------------


def _round_bankers_num_over_den(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    # Integer division and remainder
    q = torch.div(num, den, rounding_mode="floor")
    r = num - q * den
    # Compare 2*r to den to detect < 0.5, > 0.5, and == 0.5 cases
    twice_r = r << 1  # 2*r
    gt = twice_r > den
    lt = twice_r < den
    # tie -> round to even: if q is even keep q else use q+1
    q_plus_one = q + 1
    return torch.where(lt, q, torch.where(gt, q_plus_one, torch.where((q & 1) == 0, q, q_plus_one)))


def _kv_blocks_nonwrap(
    q_blocks: int,
    kv_blocks: int,
    block_size: int,
    window_size: int,
    stride: float,
    q_len: int,
    kv_len: int,
    device: str,
):
    """For each query block, compute which KV blocks fall inside a sliding window (no wrap-around).
    This produces a mapping:
      - kv_num_blocks[q]: number of KV blocks visible to query block q
      - kv_indices[q, :kv_num_blocks[q]]: the actual KV block indices
    This helper over-approximates at block granularity; per-token filtering
    is refined later by `mask_mod`.
    """
    # Keep scalars as tensors (avoids graph breaks in torch.compile)
    q_len_t = torch.tensor(q_len, device=device, dtype=torch.int64)
    kv_len_t = torch.tensor(kv_len, device=device, dtype=torch.int64)
    half_t = torch.tensor(window_size // 2, device=device, dtype=torch.int64)

    qb = torch.arange(q_blocks, device=device, dtype=torch.int64)  # [Q]
    q0 = qb * block_size  # first query token in block
    q1 = torch.minimum(q0 + (block_size - 1), torch.tensor(q_len - 1, device=device, dtype=torch.int64))  # last query token in block

    # Map query-token positions to KV "centers" by proportional scaling:
    s = torch.tensor(stride, device=device, dtype=torch.float64)
    q0f = q0.to(torch.float64) * s
    q1f = q1.to(torch.float64) * s
    lo_center = torch.minimum(q0f, q1f)
    hi_center = torch.maximum(q0f, q1f)
    min_center = torch.floor(lo_center).to(torch.int64)
    max_center = torch.ceil(hi_center).to(torch.int64)

    # Convert window in tokens to an inclusive token range [lo_tok, hi_tok]
    lo_tok = torch.maximum(min_center - half_t, torch.zeros_like(min_center))  # Leftmost KV token this block can see
    hi_tok = torch.minimum(max_center + half_t, kv_len_t - 1)  # Rightmost KV token this block can see

    # Convert token range to block range
    lo_blk = torch.div(lo_tok, block_size, rounding_mode="floor")
    hi_blk = torch.div(hi_tok, block_size, rounding_mode="floor")

    # Build a [Q, K] boolean mask over KV blocks
    base = torch.arange(kv_blocks, device=device, dtype=torch.int64)  # [K]
    base2 = base.unsqueeze(0).expand(q_blocks, kv_blocks)  # [Q,K]
    mask = (base2 >= lo_blk.unsqueeze(1)) & (base2 <= hi_blk.unsqueeze(1))  # [Q,K]

    #  - kv_num_blocks: count of True per row
    #  - kv_indices: compacted indices per row (positions via cumsum trick)
    kv_num_blocks = mask.sum(dim=1).to(torch.int32)  # [Q]
    pos = (mask.cumsum(dim=1) - 1).masked_fill(~mask, 0).to(torch.int64)  # [Q,K]
    src = base2.to(torch.int32).masked_fill(~mask, 0)
    kv_indices = torch.zeros((q_blocks, kv_blocks), dtype=torch.int32, device=device)
    kv_indices.scatter_(dim=1, index=pos, src=src)  # compact along dim=1
    return kv_num_blocks, kv_indices


def _kv_blocks_wrap(
    q_blocks: int,
    kv_blocks: int,
    block_size: int,
    window_size: int,
    stride: float,
    q_len: int,
    kv_len: int,
    device: str,
):
    """Same as _kv_blocks_nonwrap but the sliding window can wrap around the end of the KV sequence (circular indexing).
    Handles three row types:
    1) all_rows: window covers the whole KV (mask all blocks).
    2) nonwrap_row: window doesn't cross the end (single interval).
    3) wrap_row: window crosses the end (union of two intervals).
    """
    q_len_t = torch.tensor(q_len, device=device, dtype=torch.int64)
    kv_len_t = torch.tensor(kv_len, device=device, dtype=torch.int64)
    half_t = torch.tensor(window_size // 2, device=device, dtype=torch.int64)

    qb = torch.arange(q_blocks, device=device, dtype=torch.int64)
    q0 = qb * block_size
    q1 = torch.minimum(q0 + (block_size - 1), torch.tensor(q_len - 1, device=device, dtype=torch.int64))

    s = torch.tensor(stride, device=device, dtype=torch.float64)
    q0f = q0.to(torch.float64) * s
    q1f = q1.to(torch.float64) * s
    lo_center = torch.minimum(q0f, q1f)
    hi_center = torch.maximum(q0f, q1f)
    min_center = torch.floor(lo_center).to(torch.int64)
    max_center = torch.ceil(hi_center).to(torch.int64)

    lo_tok = min_center - half_t
    hi_tok = max_center + half_t
    span = hi_tok - lo_tok + 1  # window width in tokens (inclusive)

    base = torch.arange(kv_blocks, device=device, dtype=torch.int64)
    base2 = base.unsqueeze(0).expand(q_blocks, kv_blocks)

    # If window covers the whole sequence, select all KV blocks
    all_rows = span >= kv_len_t

    # Mod the bounds into [0, kv_len)
    lo_mod = torch.remainder(lo_tok, kv_len_t)
    hi_mod = torch.remainder(hi_tok, kv_len_t)

    # Identify whether the modulo-interval wraps around
    nonwrap_row = (~all_rows) & (lo_mod <= hi_mod)
    wrap_row = (~all_rows) & (lo_mod > hi_mod)

    # Non-wrapping interval: [lo_mod, hi_mod]
    lo_blk_nw = torch.div(lo_mod, block_size, rounding_mode="floor")
    hi_blk_nw = torch.div(hi_mod, block_size, rounding_mode="floor")
    mask_nw = (base2 >= lo_blk_nw.unsqueeze(1)) & (base2 <= hi_blk_nw.unsqueeze(1))

    # Wrapping interval: [0, hi_mod] U [lo_mod, kv_len)
    lo_blk2 = torch.div(lo_mod, block_size, rounding_mode="floor")
    hi_blk1 = torch.div(hi_mod, block_size, rounding_mode="floor")
    mask_wr = (base2 <= hi_blk1.unsqueeze(1)) | (base2 >= lo_blk2.unsqueeze(1))

    # Row-wise select the correct mask without Python branching
    mask = (all_rows.unsqueeze(1)) | (nonwrap_row.unsqueeze(1) & mask_nw) | (wrap_row.unsqueeze(1) & mask_wr)

    # Pack ragged rows (same trick as in non-wrap)
    kv_num_blocks = mask.sum(dim=1).to(torch.int32)
    pos = (mask.cumsum(dim=1) - 1).masked_fill(~mask, 0).to(torch.int64)
    src = base2.to(torch.int32).masked_fill(~mask, 0)
    kv_indices = torch.zeros((q_blocks, kv_blocks), dtype=torch.int32, device=device)
    kv_indices.scatter_(dim=1, index=pos, src=src)
    return kv_num_blocks, kv_indices


# compile the helpers
# Intentional shadowing: replace original functions with compiled versions
_kv_blocks_nonwrap = torch.compile(_kv_blocks_nonwrap, dynamic=True)  # type: ignore[invalid-assignment]
_kv_blocks_wrap = torch.compile(_kv_blocks_wrap, dynamic=True)  # type: ignore[invalid-assignment]


def build_strided_sliding_window_blockmask(
    *,
    window_size: int,
    stride: float,  # kept for mask_mod; not used in compiled helpers
    q_len: int,
    kv_len: int,
    device: str,
    wrap: bool,
    block_size: int = 128,
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

    # Compute the block-level KV visibility (coarse envelope)
    if wrap:
        kv_num_blocks, kv_indices = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device)
    else:
        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device)

    # Flex Attention expects [B, H, Q_blocks, ...]; we use singleton B=H=1
    kv_num_blocks = kv_num_blocks.unsqueeze(0).unsqueeze(0)  # [1,1,Q_blocks]
    kv_indices = kv_indices.unsqueeze(0).unsqueeze(0)  # [1,1,Q_blocks,kv_blocks]

    # Scalars as tensors for compiled mask_mod
    stride_val = torch.tensor(stride, device=device)
    kv_len_t = torch.as_tensor(kv_len, device=device).reshape(())

    # Per-token refinement: given (q_idx, kv_idx) decide if it's inside the
    # strided window. Called by Flex Attention during block processing.
    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
        # Center of the window for this query token
        q_center = torch.round(q_idx * stride_val)
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
