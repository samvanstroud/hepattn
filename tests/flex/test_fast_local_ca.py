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
