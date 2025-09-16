import importlib
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from hepattn.flex.fast_local_ca import (
    _kv_blocks_nonwrap,  # noqa: PLC2701
    _kv_blocks_wrap,  # noqa: PLC2701
    build_strided_sliding_window_blockmask,
)
from hepattn.flex.local_ca import sliding_window_mask_strided, sliding_window_mask_strided_wrapped


@pytest.fixture
def test_config():
    """Common test configuration."""
    return {"window_size": 32, "stride": 2.0, "q_len": 100, "kv_len": 1000, "device": "cpu", "block_size": 128, "dtype_float": torch.float32}


def blockmask_to_dense(block_mask, q_len, kv_len, device):
    """Convert BlockMask to dense tensor using create_mask."""
    return create_mask(block_mask.mask_mod, 1, 1, q_len, kv_len, device)


def _dense_from_blockmask(mask, q_len, kv_len, device):
    return create_mask(mask.mask_mod, 1, 1, q_len, kv_len, device)


class TestKvBlocks:
    """Test the _kv_blocks_nonwrap function."""

    def test_basic_functionality(self):
        """Test basic functionality of _kv_blocks_nonwrap."""
        q_blocks = 2
        kv_blocks = 4
        block_size = 128
        window_size = 32
        stride = torch.tensor(2.0)
        q_len = 200
        kv_len = 400
        device = "cpu"

        kv_num_blocks_nonwrap, kv_indices_nonwrap = _kv_blocks_nonwrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.float32
        )
        kv_num_blocks_wrap, kv_indices_wrap = _kv_blocks_wrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.float32
        )
        # Check output shapes
        assert kv_num_blocks_nonwrap.shape == (q_blocks,)
        assert kv_indices_nonwrap.shape == (q_blocks, kv_blocks)
        assert kv_num_blocks_wrap.shape == (q_blocks,)
        assert kv_indices_wrap.shape == (q_blocks, kv_blocks)

        # Check that all values are non-negative
        assert torch.all(kv_num_blocks_nonwrap >= 0)
        assert torch.all(kv_indices_nonwrap >= 0)
        assert torch.all(kv_num_blocks_wrap >= 0)
        assert torch.all(kv_indices_wrap >= 0)

    def test_large_window_size(self):
        """Test with window size larger than sequence length."""
        q_blocks = 2
        kv_blocks = 3
        block_size = 128
        window_size = 1000  # Much larger than kv_len
        stride = torch.tensor(1.0)
        q_len = 200
        kv_len = 300
        device = "cpu"

        kv_num_blocks_nonwrap, kv_indices_nonwrap = _kv_blocks_nonwrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.float32
        )
        kv_num_blocks_wrap, kv_indices_wrap = _kv_blocks_wrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.float32
        )
        # All blocks should be visible
        assert torch.all(kv_num_blocks_nonwrap == kv_blocks)
        assert torch.all(kv_num_blocks_wrap == kv_blocks)
        # Check that indices are valid
        assert torch.all(kv_indices_nonwrap >= 0)
        assert torch.all(kv_indices_wrap >= 0)


def test_non_wrapped_equivalence(test_config):
    """Test that fast_local_ca and local_ca produce equivalent masks for non-wrapped case."""
    # Create masks using both approaches
    fast_mask = build_strided_sliding_window_blockmask(wrap=False, **test_config)

    local_mask = sliding_window_mask_strided(
        window_size=test_config["window_size"],
        stride=test_config["stride"],
        q_len=test_config["q_len"],
        kv_len=test_config["kv_len"],
        device=test_config["device"],
    )

    # They should be identical even though they're different types
    fast_dense = fast_mask.to_dense().int()
    local_dense = local_mask.to_dense().int()
    assert torch.allclose(fast_dense, local_dense), "Fast and local CA masks should be identical for non-wrapped case"


def test_wrapped_equivalence(test_config):
    """Test that fast_local_ca and local_ca produce equivalent masks for wrapped case."""
    # Create masks using both approaches
    fast_mask = build_strided_sliding_window_blockmask(wrap=True, **test_config)

    local_mask = sliding_window_mask_strided_wrapped(
        window_size=test_config["window_size"],
        stride=test_config["stride"],
        q_len=test_config["q_len"],
        kv_len=test_config["kv_len"],
        device=test_config["device"],
    )

    # They should be identical even though they're different types
    fast_dense = fast_mask.to_dense()
    local_dense = local_mask.to_dense()
    assert torch.allclose(fast_dense, local_dense), "Fast and local CA masks should be identical for wrapped case"


class TestErrorCases:
    """Test error cases and validation."""

    def test_odd_window_size_error(self):
        """Test that odd window size raises ValueError."""
        with pytest.raises(ValueError, match="Window size must be even"):
            build_strided_sliding_window_blockmask(
                window_size=31,  # odd
                stride=2.0,
                q_len=100,
                kv_len=1000,
                device="cpu",
                wrap=False,
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_window_small_sequence(self):
        """Test with window size larger than sequence length."""
        mask = build_strided_sliding_window_blockmask(
            window_size=1000,  # much larger than sequences
            stride=1.0,
            q_len=50,
            kv_len=50,
            device="cpu",
            wrap=False,
        )

        dense_mask = blockmask_to_dense(mask, 50, 50, "cpu")
        # All tokens should be visible to all queries
        assert torch.all(dense_mask)

    def test_different_q_kv_lengths(self):
        """Test with different query and key-value lengths."""
        mask = build_strided_sliding_window_blockmask(
            window_size=32,
            stride=2.0,
            q_len=100,
            kv_len=200,  # different length
            device="cpu",
            wrap=False,
        )

        dense_mask = blockmask_to_dense(mask, 100, 200, "cpu")
        assert dense_mask.shape == (1, 1, 100, 200)

    def test_very_long_sequences(self):
        """Test with very long sequences."""
        mask = build_strided_sliding_window_blockmask(
            window_size=64,
            stride=1.5,
            q_len=2000,
            kv_len=3000,
            device="cpu",
            wrap=False,
            block_size=128,
        )

        dense_mask = blockmask_to_dense(mask, 2000, 3000, "cpu")
        assert dense_mask.shape == (1, 1, 2000, 3000)
        # Should be sparse (not all True)
        assert not torch.all(dense_mask)


class TestWrapVsNonWrap:
    """Test wrap vs non-wrap behavior."""

    def test_wrap_behavior_difference(self):
        """Test that wrap and non-wrap produce different results for edge cases."""
        # Use a case where wrapping should make a difference
        q_len, kv_len = 50, 100
        window_size = 40
        stride = 2.0

        mask_nonwrap = build_strided_sliding_window_blockmask(
            window_size=window_size,
            stride=stride,
            q_len=q_len,
            kv_len=kv_len,
            device="cpu",
            wrap=False,
        )

        mask_wrap = build_strided_sliding_window_blockmask(
            window_size=window_size,
            stride=stride,
            q_len=q_len,
            kv_len=kv_len,
            device="cpu",
            wrap=True,
        )

        dense_nonwrap = mask_nonwrap.to_dense()
        dense_wrap = mask_wrap.to_dense()

        # They should have the same shape
        assert dense_nonwrap.shape == dense_wrap.shape

        # For this configuration, wrap should allow more connections
        # (non-wrap is more restrictive)
        wrap_connections = dense_wrap.sum()
        nonwrap_connections = dense_nonwrap.sum()
        assert wrap_connections >= nonwrap_connections

    def test_wrap_equivalence_for_small_windows(self):
        """Test that wrap and non-wrap are equivalent for small windows."""
        # Small window relative to sequence length
        mask_nonwrap = build_strided_sliding_window_blockmask(
            window_size=10,
            stride=1.0,
            q_len=100,
            kv_len=100,
            device="cpu",
            wrap=False,
        )

        mask_wrap = build_strided_sliding_window_blockmask(
            window_size=10,
            stride=1.0,
            q_len=100,
            kv_len=100,
            device="cpu",
            wrap=True,
        )

        # For small windows, they should be identical
        dense_nonwrap = mask_nonwrap.to_dense()
        dense_wrap = mask_wrap.to_dense()
        assert torch.allclose(dense_nonwrap, dense_wrap)


class TestBlockMaskProperties:
    """Test BlockMask properties and methods."""

    def test_blockmask_shape_validation(self):
        """Test that BlockMask has correct shape."""
        mask = build_strided_sliding_window_blockmask(
            window_size=32,
            stride=2.0,
            q_len=100,
            kv_len=1000,
            device="cpu",
            wrap=False,
        )

        dense_mask = blockmask_to_dense(mask, 100, 1000, "cpu")
        assert dense_mask.shape == (1, 1, 100, 1000)


class TestCompiledFunctions:
    """Test that compiled functions work correctly."""

    def test_compiled_kv_blocks_nonwrap(self):
        """Test that compiled _kv_blocks_nonwrap produces correct results."""
        q_blocks = 3
        kv_blocks = 5
        block_size = 128
        window_size = 32
        stride = torch.tensor(2.0)
        q_len = 300
        kv_len = 500
        device = "cpu"

        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.float32)

        # Check output shapes
        assert kv_num_blocks.shape == (q_blocks,)
        assert kv_indices.shape == (q_blocks, kv_blocks)

        # Check that all values are valid
        assert torch.all(kv_num_blocks >= 0)
        assert torch.all(kv_num_blocks <= kv_blocks)
        assert torch.all(kv_indices >= 0)
        assert torch.all(kv_indices < kv_blocks)

    def test_compiled_kv_blocks_wrap(self):
        """Test that compiled _kv_blocks_wrap produces correct results."""
        q_blocks = 3
        kv_blocks = 5
        block_size = 128
        window_size = 32
        stride = torch.tensor(2.0)
        q_len = 300
        kv_len = 500
        device = "cpu"

        kv_num_blocks, kv_indices = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.float32)

        # Check output shapes
        assert kv_num_blocks.shape == (q_blocks,)
        assert kv_indices.shape == (q_blocks, kv_blocks)

        # Check that all values are valid
        assert torch.all(kv_num_blocks >= 0)
        assert torch.all(kv_num_blocks <= kv_blocks)
        assert torch.all(kv_indices >= 0)
        assert torch.all(kv_indices < kv_blocks)


class TestMaskModFunction:
    """Test the mask_mod function behavior."""

    def test_mask_mod_nonwrap_behavior(self):
        """Test mask_mod function for non-wrap case."""
        mask = build_strided_sliding_window_blockmask(
            window_size=32,
            stride=2.0,
            q_len=100,
            kv_len=1000,
            device="cpu",
            wrap=False,
        )

        # Test some specific mask_mod calls
        # These should match the expected sliding window behavior
        dense_mask = blockmask_to_dense(mask, 100, 1000, "cpu")

        # Check that the mask has the expected sliding window structure
        # For stride 2.0, query 0 should see keys around position 0
        # Query 1 should see keys around position 2, etc.
        for q_idx in [0, 1, 2, 10, 20]:
            if q_idx < 100:  # within bounds
                expected_center = round(q_idx * 2.0)
                window_start = max(0, expected_center - 16)
                window_end = min(1000, expected_center + 16)
                # Check that the mask is True in the expected window
                actual_window = dense_mask[0, 0, q_idx, window_start:window_end]
                assert torch.all(actual_window), f"Query {q_idx} should see keys in window [{window_start}, {window_end})"

    def test_mask_mod_wrap_behavior(self):
        """Test mask_mod function for wrap case."""
        mask = build_strided_sliding_window_blockmask(
            window_size=32,
            stride=2.0,
            q_len=100,
            kv_len=1000,
            device="cpu",
            wrap=True,
        )

        dense_mask = blockmask_to_dense(mask, 100, 1000, "cpu")

        # For wrap case, the mask should allow connections that wrap around
        # This is harder to test directly, but we can check that wrap allows
        # more connections than non-wrap for edge cases
        mask_nonwrap = build_strided_sliding_window_blockmask(
            window_size=32,
            stride=2.0,
            q_len=100,
            kv_len=1000,
            device="cpu",
            wrap=False,
        )

        dense_nonwrap = blockmask_to_dense(mask_nonwrap, 100, 1000, "cpu")

        # Wrap should allow at least as many connections as non-wrap
        assert dense_mask.sum() >= dense_nonwrap.sum()


class TestEdgeAndRoundingBehavior:
    def test_wrap_edges_allow_circular_connections(self):
        """When wrapping is enabled, include indices near kv_len-1 across boundary."""
        q_len, kv_len = 8, 32
        window_size = 8  # half=4
        stride = 4.0  # centers: 0,4,8,...

        m_wrap = build_strided_sliding_window_blockmask(window_size=window_size, stride=stride, q_len=q_len, kv_len=kv_len, device="cpu", wrap=True)
        dense = _dense_from_blockmask(m_wrap, q_len, kv_len, "cpu")

        # q_idx=0 -> center ~ 0; with wrap, indices {28,29,30,31,0,1,2,3,4} (window_size is inclusive per mask_mod)
        q0 = dense[0, 0, 0]
        expected_idxs = {28, 29, 30, 31, 0, 1, 2, 3, 4}
        assert all(q0[i].item() for i in expected_idxs), "wrap should include indices across boundary"

    def test_nonwrap_edges_trim_at_zero(self):
        """Without wrapping, negative indices must be clipped away."""
        q_len, kv_len = 8, 32
        window_size = 8  # half=4
        stride = 4.0

        m_no = build_strided_sliding_window_blockmask(window_size=window_size, stride=stride, q_len=q_len, kv_len=kv_len, device="cpu", wrap=False)
        dense = _dense_from_blockmask(m_no, q_len, kv_len, "cpu")

        q0 = dense[0, 0, 0]
        # Should include 0..4 but not kv_len-1 etc.
        assert all(q0[i].item() for i in range(5))
        assert not q0[kv_len - 1].item(), "non-wrap must not include wrapped indices"


class TestPartialBlocksAndOddBlockSizes:
    def test_partial_last_blocks_shape_and_bounds(self):
        """Ensure seq lengths respected when lengths not multiples of block_size."""
        mask = build_strided_sliding_window_blockmask(window_size=32, stride=2.0, q_len=257, kv_len=515, device="cpu", wrap=False, block_size=128)
        dense = _dense_from_blockmask(mask, 257, 515, "cpu")
        assert dense.shape == (1, 1, 257, 515)
        # Make sure last query doesn't access out-of-bounds keys
        assert dense[0, 0, 256].shape[0] == 515

    def test_small_irregular_block_size(self):
        """Stress rounding paths with tiny, non-power-of-two block size."""
        q_len, kv_len = 123, 234
        mask = build_strided_sliding_window_blockmask(window_size=12, stride=1.1, q_len=q_len, kv_len=kv_len, device="cpu", wrap=True, block_size=7)
        dense = _dense_from_blockmask(mask, q_len, kv_len, "cpu")
        assert dense.shape == (1, 1, q_len, kv_len)
        assert dense.sum() > 0  # not degenerate


class TestDtypeAndStrideVariants:
    def test_float64_dtype(self):
        """Exercise dtype_float pathway with float64."""
        mask = build_strided_sliding_window_blockmask(
            window_size=16, stride=0.75, q_len=64, kv_len=80, device="cpu", wrap=False, dtype_float=torch.float64
        )
        dense = _dense_from_blockmask(mask, 64, 80, "cpu")
        assert dense.dtype == torch.bool
        assert dense.any()

    def test_fractional_stride_less_than_one_equivalence(self):
        """For stride<1, fast mask should match reference local mask (non-wrap)."""
        params = {"window_size": 20, "stride": 0.5, "q_len": 120, "kv_len": 100, "device": "cpu"}
        fast = build_strided_sliding_window_blockmask(wrap=False, **params).to_dense().int()
        ref = sliding_window_mask_strided(**params).to_dense().int()
        assert torch.allclose(fast, ref), "fast vs local mask mismatch for stride<1"

    def test_negative_stride_stability(self):
        """Negative stride path should be stable and produce a valid mask."""
        mask = build_strided_sliding_window_blockmask(window_size=10, stride=-1.0, q_len=50, kv_len=60, device="cpu", wrap=True)
        dense = _dense_from_blockmask(mask, 50, 60, "cpu")
        assert dense.shape == (1, 1, 50, 60)
        assert dense.any(), "mask should not be empty even with negative stride"


class TestKvBlocksInternals:
    def test_kv_blocks_nonwrap_monotonic_and_in_range(self):
        """Compacted indices within range and monotonic for first kv_num_blocks."""
        q_blocks, kv_blocks = 6, 11
        block_size, window_size = 16, 24
        stride = torch.tensor(1.3)
        q_len, kv_len = 96, 176  # multiple blocks; last block partial for kv

        kv_num, kv_idx = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, "cpu", torch.float32)
        assert kv_num.shape == (q_blocks,)
        assert kv_idx.shape == (q_blocks, kv_blocks)

        for r in range(q_blocks):
            n = kv_num[r].item()
            row = kv_idx[r, :n].tolist()
            assert all(0 <= v < kv_blocks for v in row)
            assert row == sorted(row), "indices should be non-decreasing after compaction"

    def test_kv_blocks_wrap_monotonic_and_all_rows_branch(self):
        """Trigger all_rows branch (span >= kv_len) and verify compaction."""
        q_blocks, kv_blocks = 4, 8
        block_size = 32
        # Large window -> span >= kv_len for all rows in wrap mode
        window_size = 10_000
        stride = torch.tensor(2.0)
        q_len, kv_len = 100, 256

        kv_num, kv_idx = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, "cpu", torch.float32)
        assert torch.all(kv_num == kv_blocks), "all_rows path should select all blocks"
        for r in range(q_blocks):
            row = kv_idx[r, :kv_blocks].tolist()
            # With all blocks selected, we expect {0..kv_blocks-1} in order
            assert row == list(range(kv_blocks))

    def test_nonwrap_vs_wrap_block_counts_differ_when_expected(self):
        """Visible block counts per row should be >= in wrap vs non-wrap."""
        q_blocks, kv_blocks = 5, 9
        block_size, window_size = 32, 40
        stride = torch.tensor(2.0)
        q_len, kv_len = 160, 256

        num_nw, _ = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, "cpu", torch.float32)
        num_wr, _ = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, "cpu", torch.float32)
        assert torch.all(num_wr >= num_nw)


class TestEquivalenceMoreRegimes:
    def test_wrap_equivalence_with_reference_for_fractional_stride(self):
        """Cross-check fast vs reference (wrap=True) for fractional stride."""
        cfg = {"window_size": 18, "stride": 1.25, "q_len": 90, "kv_len": 140, "device": "cpu"}
        fast = build_strided_sliding_window_blockmask(wrap=True, **cfg).to_dense().int()
        ref = sliding_window_mask_strided_wrapped(**cfg).to_dense().int()
        assert torch.allclose(fast, ref)

    def test_dense_sparsity_non_trivial(self):
        """For mid-size windows, mask should be sparse and non-empty."""
        m = build_strided_sliding_window_blockmask(window_size=32, stride=1.0, q_len=300, kv_len=300, device="cpu", wrap=False)
        d = _dense_from_blockmask(m, 300, 300, "cpu")
        total = d.numel()
        on = int(d.sum())
        assert 0 < on < total, "mask should be neither empty nor fully dense"


@pytest.fixture(scope="module")
def flc_eager():
    """Reload hepattn.flex.fast_local_ca with torch.compile turned into a no-op,
    so coverage includes the original Python bodies of _kv_blocks_*.
    """

    # Make compile a no-op for reload
    def _identity_compile(fn, *_args, **_kwargs):
        return fn

    with patch("torch.compile", new=_identity_compile):
        mod = importlib.import_module("hepattn.flex.fast_local_ca")
        return importlib.reload(mod)


def test_wrap_has_mixed_rows_and_expected_blocks(flc_eager):
    """Force both wrap_row and nonwrap_row within a single _kv_blocks_wrap call,
    then verify selected block indices for each row.
    """
    q_blocks = 2
    kv_blocks = 4
    block_size = 8
    window_size = 16  # half=8 -> for row0, spans negative -> wrap_row; for row1, nonwrap
    stride = torch.tensor(1.0)  # scale by token directly
    q_len = 16
    kv_len = 32
    device = "cpu"

    kv_num, kv_idx = flc_eager._kv_blocks_wrap(  # noqa: SLF001
        q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.float32
    )

    # Row 0 should wrap: expect blocks {0,1,3} (two from start, one from end)
    n0 = int(kv_num[0])
    sel0 = kv_idx[0, :n0].tolist()
    assert sel0 == [0, 1, 3]

    # Row 1 should be non-wrap: a single interval, expect {0,1,2}
    n1 = int(kv_num[1])
    sel1 = kv_idx[1, :n1].tolist()
    assert sel1 == [0, 1, 2]


def test_nonwrap_indices_expected_interval(flc_eager):
    """Verify _kv_blocks_nonwrap returns a compact single interval of block indices."""
    q_blocks = 2
    kv_blocks = 5
    block_size = 8
    window_size = 12  # half=6
    stride = torch.tensor(1.0)
    q_len = 16
    kv_len = 40
    device = "cpu"

    kv_num, kv_idx = flc_eager._kv_blocks_nonwrap(  # noqa: SLF001
        q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.float32
    )

    # Row 0 center near ~3; token-range should map to blocks {0,1}
    n0 = int(kv_num[0])
    assert kv_idx[0, :n0].tolist() == [0, 1]

    # Row 1 center near ~11; token-range should map to blocks {0,1,2}
    n1 = int(kv_num[1])
    assert kv_idx[1, :n1].tolist() == [0, 1, 2]


def test_all_rows_branch_in_wrap(flc_eager):
    """Large window triggers 'all_rows' path in _kv_blocks_wrap."""
    q_blocks, kv_blocks = 3, 7
    block_size = 16
    window_size = 10_000  # definitely covers entire kv
    stride = torch.tensor(2.0)
    q_len, kv_len = 64, 256
    device = "cpu"

    kv_num, kv_idx = flc_eager._kv_blocks_wrap(  # noqa: SLF001
        q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.float32
    )

    assert torch.all(kv_num == kv_blocks)
    for r in range(q_blocks):
        assert kv_idx[r, :kv_blocks].tolist() == list(range(kv_blocks))


# ---------------------------------
# Extra behaviors hitting mask_mod
# ---------------------------------


def test_zero_window_size_is_center_only():
    """window_size=0 is allowed (even). Each query should only attend its center."""
    q_len, kv_len = 20, 40
    stride = 1.5
    m = build_strided_sliding_window_blockmask(window_size=0, stride=stride, q_len=q_len, kv_len=kv_len, device="cpu", wrap=False)
    d = _dense_from_blockmask(m, q_len, kv_len, "cpu")

    for q_idx in (0, 1, 5, 10, 19):
        center = round(q_idx * stride)
        center = min(max(center, 0), kv_len - 1)
        row = d[0, 0, q_idx]
        assert row[center].item() is True
        # neighbors should be False (when in bounds)
        if center - 1 >= 0:
            assert row[center - 1].item() is False
        if center + 1 < kv_len:
            assert row[center + 1].item() is False


def test_fractional_stride_wrap_equivalence_additional():
    """Another equivalence check for wrap=True with fractional stride."""
    cfg = {"window_size": 18, "stride": 1.25, "q_len": 90, "kv_len": 140, "device": "cpu"}
    fast = build_strided_sliding_window_blockmask(wrap=True, **cfg).to_dense().int()
    ref = sliding_window_mask_strided_wrapped(**cfg).to_dense().int()
    assert torch.allclose(fast, ref)


def test_rounding_half_up_behavior_fixed():
    """mask_mod uses round(q_idx * stride): for stride=1.5 and q_idx=1 -> 2."""
    q_len, kv_len = 10, 40
    stride = 1.5
    window_size = 6  # half=3

    m = build_strided_sliding_window_blockmask(window_size=window_size, stride=stride, q_len=q_len, kv_len=kv_len, device="cpu", wrap=False)
    dense = _dense_from_blockmask(m, q_len, kv_len, "cpu")

    q_idx = 1
    center = round(q_idx * stride)  # 2
    lo = max(0, center - window_size // 2)
    hi = min(kv_len - 1, center + window_size // 2)

    q1 = dense[0, 0, q_idx]
    assert all(q1[i].item() for i in range(lo, hi + 1)), "window should center at round(1.5)=2"


def test_dtype_float64_and_negative_stride():
    """Hit dtype_float path and odd stride sign; just ensure valid shape and any True."""
    m = build_strided_sliding_window_blockmask(window_size=12, stride=-0.75, q_len=64, kv_len=80, device="cpu", wrap=True, dtype_float=torch.float64)
    d = _dense_from_blockmask(m, 64, 80, "cpu")
    assert d.shape == (1, 1, 64, 80)
    assert d.any()
