import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from hepattn.flex.local_ca import (
    sliding_window_mask_strided,
    sliding_window_mask_strided_wrapped,
    transpose_blockmask,
)


def _materialize_blockmask(block_mask, q_len: int, kv_len: int, device: str = "cpu") -> torch.Tensor:
    """Utility to convert a BlockMask into a dense boolean mask."""
    return create_mask(block_mask.mask_mod, 1, 1, q_len, kv_len, device)


def test_sliding_window_mask_strided_requires_even_window():
    """sliding_window_mask_strided should reject odd window sizes."""
    with pytest.raises(ValueError):
        sliding_window_mask_strided(window_size=3, q_len=8, kv_len=16, device="cpu")


def test_sliding_window_mask_strided_window_behavior():
    """Non-wrapped sliding window mask should behave like a centered window on the sequence."""
    window_size = 4
    q_len = 4
    kv_len = 16
    device = "cpu"

    block_mask = sliding_window_mask_strided(window_size, q_len, kv_len, device)
    full_mask = _materialize_blockmask(block_mask, q_len, kv_len, device)[0, 0]

    # For non-wrapped windows, queries near the start/end see windows clipped at the boundaries.
    # Check a couple of representative query positions.
    # q_idx = 0 should attend near the start of the sequence.
    attended_q0 = torch.nonzero(full_mask[0], as_tuple=False).flatten()
    assert attended_q0.min() == 0
    assert attended_q0.max() <= window_size

    # q_idx in the middle should attend a symmetric region around its projected center.
    mid_q = q_len // 2
    attended_mid = torch.nonzero(full_mask[mid_q], as_tuple=False).flatten()
    assert attended_mid.numel() > 0
    # The window width should not exceed window_size + 2 (allowing for integer rounding effects).
    assert attended_mid.max() - attended_mid.min() <= window_size + 2


def test_sliding_window_mask_strided_wrapped_and_transpose():
    """Wrapped sliding window mask should wrap around, and transpose_blockmask should be an exact transpose."""
    window_size = 4
    q_len = 8
    kv_len = 8
    device = "cpu"

    # Wrapped sliding window
    block_mask_wrapped = sliding_window_mask_strided_wrapped(window_size, q_len, kv_len, device)
    full_mask_wrapped = _materialize_blockmask(block_mask_wrapped, q_len, kv_len, device)[0, 0]

    # For wrapped windows, a query near the start should also attend to positions near the end.
    attended_q0 = torch.nonzero(full_mask_wrapped[0], as_tuple=False).flatten()
    assert (attended_q0 == 0).any()
    assert (attended_q0 >= kv_len - window_size // 2).any()

    # Now verify transpose_blockmask produces an exact transpose of the dense mask.
    transposed_block = transpose_blockmask(block_mask_wrapped, q_tokens=q_len, kv_tokens=kv_len, dev=device)
    transposed_mask = _materialize_blockmask(transposed_block, kv_len, q_len, device)[0, 0]

    assert transposed_mask.shape == full_mask_wrapped.transpose(-2, -1).shape
    assert torch.equal(transposed_mask, full_mask_wrapped.transpose(-2, -1))
