import torch
from torch.nn.attention.flex_attention import BlockMask

# ---------------------------
# Compiled tensor-only helpers
# ---------------------------


def _round_bankers_num_over_den(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    # num, den: int64 tensors (broadcastable)
    q = torch.div(num, den, rounding_mode="floor")
    r = num - q * den
    twice_r = r << 1  # 2*r
    gt = twice_r > den
    lt = twice_r < den
    # tie -> round to even
    q_plus_one = q + 1
    return torch.where(lt, q, torch.where(gt, q_plus_one, torch.where((q & 1) == 0, q, q_plus_one)))


def _kv_blocks_nonwrap(
    q_blocks: int,
    kv_blocks: int,
    block_size: int,
    window_size: int,
    q_len: int,
    kv_len: int,
    device: str,
):
    # int scalars as tensors (kept as tensors; no .item() anywhere)
    q_len_t = torch.tensor(q_len, device=device, dtype=torch.int64)
    kv_len_t = torch.tensor(kv_len, device=device, dtype=torch.int64)
    half_t = torch.tensor(window_size // 2, device=device, dtype=torch.int64)

    qb = torch.arange(q_blocks, device=device, dtype=torch.int64)  # [Q]
    q0 = qb * block_size
    q1 = torch.minimum(q0 + (block_size - 1), torch.tensor(q_len - 1, device=device, dtype=torch.int64))

    # centers = round( q * kv_len / q_len ) using banker’s rounding (matches torch.round)
    min_center = _round_bankers_num_over_den(q0 * kv_len_t, q_len_t)  # [Q]
    max_center = _round_bankers_num_over_den(q1 * kv_len_t, q_len_t)  # [Q]

    lo_tok = torch.maximum(min_center - half_t, torch.zeros_like(min_center))
    hi_tok = torch.minimum(max_center + half_t, kv_len_t - 1)

    lo_blk = torch.div(lo_tok, block_size, rounding_mode="floor")
    hi_blk = torch.div(hi_tok, block_size, rounding_mode="floor")

    base = torch.arange(kv_blocks, device=device, dtype=torch.int64)  # [K]
    base2 = base.unsqueeze(0).expand(q_blocks, kv_blocks)  # [Q,K]
    mask = (base2 >= lo_blk.unsqueeze(1)) & (base2 <= hi_blk.unsqueeze(1))  # [Q,K]

    kv_num_blocks = mask.sum(dim=1).to(torch.int32)  # [Q]
    pos = (mask.cumsum(dim=1) - 1).masked_fill(~mask, 0).to(torch.int64)  # [Q,K]
    src = base2.to(torch.int32).masked_fill(~mask, 0)
    kv_indices = torch.zeros((q_blocks, kv_blocks), dtype=torch.int32, device=device)
    kv_indices.scatter_(dim=1, index=pos, src=src)
    return kv_num_blocks, kv_indices


def _kv_blocks_wrap(
    q_blocks: int,
    kv_blocks: int,
    block_size: int,
    window_size: int,
    q_len: int,
    kv_len: int,
    device: str,
):
    q_len_t = torch.tensor(q_len, device=device, dtype=torch.int64)
    kv_len_t = torch.tensor(kv_len, device=device, dtype=torch.int64)
    half_t = torch.tensor(window_size // 2, device=device, dtype=torch.int64)

    qb = torch.arange(q_blocks, device=device, dtype=torch.int64)
    q0 = qb * block_size
    q1 = torch.minimum(q0 + (block_size - 1), torch.tensor(q_len - 1, device=device, dtype=torch.int64))

    min_center = _round_bankers_num_over_den(q0 * kv_len_t, q_len_t)
    max_center = _round_bankers_num_over_den(q1 * kv_len_t, q_len_t)

    lo_tok = min_center - half_t
    hi_tok = max_center + half_t
    span = hi_tok - lo_tok + 1

    base = torch.arange(kv_blocks, device=device, dtype=torch.int64)
    base2 = base.unsqueeze(0).expand(q_blocks, kv_blocks)

    all_rows = span >= kv_len_t
    lo_mod = torch.remainder(lo_tok, kv_len_t)
    hi_mod = torch.remainder(hi_tok, kv_len_t)
    nonwrap_row = (~all_rows) & (lo_mod <= hi_mod)
    wrap_row = (~all_rows) & (lo_mod > hi_mod)

    lo_blk_nw = torch.div(lo_mod, block_size, rounding_mode="floor")
    hi_blk_nw = torch.div(hi_mod, block_size, rounding_mode="floor")
    mask_nw = (base2 >= lo_blk_nw.unsqueeze(1)) & (base2 <= hi_blk_nw.unsqueeze(1))

    lo_blk2 = torch.div(lo_mod, block_size, rounding_mode="floor")
    hi_blk1 = torch.div(hi_mod, block_size, rounding_mode="floor")
    mask_wr = (base2 <= hi_blk1.unsqueeze(1)) | (base2 >= lo_blk2.unsqueeze(1))

    # No Python if: select row-wise using boolean broadcasting
    mask = (all_rows.unsqueeze(1)) | (nonwrap_row.unsqueeze(1) & mask_nw) | (wrap_row.unsqueeze(1) & mask_wr)

    kv_num_blocks = mask.sum(dim=1).to(torch.int32)
    pos = (mask.cumsum(dim=1) - 1).masked_fill(~mask, 0).to(torch.int64)
    src = base2.to(torch.int32).masked_fill(~mask, 0)
    kv_indices = torch.zeros((q_blocks, kv_blocks), dtype=torch.int32, device=device)
    kv_indices.scatter_(dim=1, index=pos, src=src)
    return kv_num_blocks, kv_indices


# Keep compiling the helpers (they’re now stable w.r.t. stride)
_kv_blocks_nonwrap = torch.compile(_kv_blocks_nonwrap, dynamic=True)
_kv_blocks_wrap = torch.compile(_kv_blocks_wrap, dynamic=True)


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
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    q_blocks = (q_len + block_size - 1) // block_size
    kv_blocks = (kv_len + block_size - 1) // block_size

    # Use compiled helpers (no float stride inside)
    if wrap:
        kv_num_blocks, kv_indices = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, q_len, kv_len, device)
    else:
        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, q_len, kv_len, device)

    # Expand to [B=1, H=1, Q_blocks, ...]
    kv_num_blocks = kv_num_blocks.unsqueeze(0).unsqueeze(0)  # [1,1,Q_blocks]
    kv_indices = kv_indices.unsqueeze(0).unsqueeze(0)  # [1,1,Q_blocks,kv_blocks]

    # mask_mod (unchanged semantics). Using integers here is fine; it runs inside Flex kernels.
    stride_val = torch.tensor(stride, device=device)
    kv_len_t = torch.as_tensor(kv_len, device=device).reshape(())

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
        q_center = torch.round(q_idx * stride_val)
        if not wrap:
            return (kv_idx - q_center).abs() <= window_size // 2
        diagonal = (kv_idx - q_center).abs() <= window_size // 2
        wrap_left = (kv_idx - q_center + kv_len_t).abs() <= window_size // 2
        wrap_right = (kv_idx - q_center - kv_len_t).abs() <= window_size // 2
        return diagonal | wrap_left | wrap_right

    # IMPORTANT: tell BlockMask the EXACT token lengths -> partial edge blocks handled natively
    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,  # int32
        kv_indices=kv_indices,  # int32
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(q_len, kv_len),  # <<< like create_block_mask
    )
