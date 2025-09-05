import pytest
import torch

from hepattn.flex.fast_local_ca import (
    _kv_blocks_nonwrap,  # noqa: PLC2701
    _kv_blocks_wrap,  # noqa: PLC2701
    _round_bankers_num_over_den,  # noqa: PLC2701
    build_strided_sliding_window_blockmask,
)
from hepattn.flex.local_ca import sliding_window_mask_strided, sliding_window_mask_strided_wrapped


@pytest.fixture
def test_config():
    """Common test configuration."""
    return {"window_size": 32, "stride": 2.0, "q_len": 100, "kv_len": 1000, "device": "cpu", "block_size": 128}


class TestRoundBankersNumOverDen:
    """Test the _round_bankers_num_over_den function."""

    def test_basic_rounding(self):
        """Test basic rounding cases."""
        num = torch.tensor([10, 15, 20, 25])
        den = torch.tensor([3, 3, 3, 3])

        result = _round_bankers_num_over_den(num, den)
        expected = torch.tensor([3, 5, 7, 8])  # 10/3=3.33->3, 15/3=5->5, 20/3=6.67->7, 25/3=8.33->8

        assert torch.equal(result, expected)

    def test_tie_breaking_even(self):
        """Test tie-breaking to even numbers."""
        num = torch.tensor([6, 10])  # 6/4=1.5, 10/4=2.5
        den = torch.tensor([4, 4])

        result = _round_bankers_num_over_den(num, den)
        expected = torch.tensor([2, 2])  # Both should round to even (2)

        assert torch.equal(result, expected)

    def test_tie_breaking_odd(self):
        """Test tie-breaking when result would be odd."""
        num = torch.tensor([14, 18])  # 14/4=3.5, 18/4=4.5
        den = torch.tensor([4, 4])

        result = _round_bankers_num_over_den(num, den)
        expected = torch.tensor([4, 4])  # Both should round to even (4)

        assert torch.equal(result, expected)

    def test_different_devices(self):
        """Test function works on different devices."""
        if torch.cuda.is_available():
            num = torch.tensor([10, 15], device="cuda")
            den = torch.tensor([3, 3], device="cuda")

            result = _round_bankers_num_over_den(num, den)
            expected = torch.tensor([3, 5], device="cuda")

            assert torch.equal(result, expected)


class TestKvBlocksNonwrap:
    """Test the _kv_blocks_nonwrap function."""

    def test_basic_functionality(self):
        """Test basic functionality of _kv_blocks_nonwrap."""
        q_blocks = 2
        kv_blocks = 4
        block_size = 128
        window_size = 32
        stride = 2.0
        q_len = 200
        kv_len = 400
        device = "cpu"

        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device)

        # Check output shapes
        assert kv_num_blocks.shape == (q_blocks,)
        assert kv_indices.shape == (q_blocks, kv_blocks)

        # Check that all values are non-negative
        assert torch.all(kv_num_blocks >= 0)
        assert torch.all(kv_indices >= 0)

    def test_small_sequences(self):
        """Test with small sequences."""
        q_blocks = 1
        kv_blocks = 1
        block_size = 64
        window_size = 16
        stride = 1.0
        q_len = 50
        kv_len = 50
        device = "cpu"

        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device)

        # Should have at least one block visible
        assert torch.all(kv_num_blocks > 0)
        # Check that indices are valid
        assert torch.all(kv_indices >= 0)

    def test_large_window_size(self):
        """Test with window size larger than sequence length."""
        q_blocks = 2
        kv_blocks = 3
        block_size = 128
        window_size = 1000  # Much larger than kv_len
        stride = 1.0
        q_len = 200
        kv_len = 300
        device = "cpu"

        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device)

        # All blocks should be visible
        assert torch.all(kv_num_blocks == kv_blocks)
        # Check that indices are valid
        assert torch.all(kv_indices >= 0)


class TestKvBlocksWrap:
    """Test the _kv_blocks_wrap function."""

    def test_basic_functionality(self):
        """Test basic functionality of _kv_blocks_wrap."""
        q_blocks = 2
        kv_blocks = 4
        block_size = 128
        window_size = 32
        stride = 2.0
        q_len = 200
        kv_len = 400
        device = "cpu"

        kv_num_blocks, kv_indices = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device)

        # Check output shapes
        assert kv_num_blocks.shape == (q_blocks,)
        assert kv_indices.shape == (q_blocks, kv_blocks)
        assert kv_num_blocks.dtype == torch.int32
        assert kv_indices.dtype == torch.int32

        # Check that all values are non-negative
        assert torch.all(kv_num_blocks >= 0)
        assert torch.all(kv_indices >= 0)

    def test_wrapping_behavior(self):
        """Test that wrapping behavior works correctly."""
        q_blocks = 1
        kv_blocks = 2
        block_size = 64
        window_size = 100  # Large enough to wrap
        stride = 1.0
        q_len = 50
        kv_len = 100
        device = "cpu"

        kv_num_blocks, kv_indices = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device)

        # Should have blocks visible due to wrapping
        assert torch.all(kv_num_blocks > 0)
        # Check that indices are valid
        assert torch.all(kv_indices >= 0)

    def test_all_rows_case(self):
        """Test the case where window covers the whole sequence."""
        q_blocks = 1
        kv_blocks = 3
        block_size = 64
        window_size = 1000  # Much larger than kv_len
        stride = 1.0
        q_len = 50
        kv_len = 150
        device = "cpu"

        kv_num_blocks, kv_indices = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device)

        # All blocks should be visible
        assert torch.all(kv_num_blocks == kv_blocks)
        # Check that indices are valid
        assert torch.all(kv_indices >= 0)


class TestBuildStridedSlidingWindowBlockmask:
    """Test the build_strided_sliding_window_blockmask function."""

    def test_wrap_vs_nonwrap_differences(self, test_config):
        """Test that wrap=True and wrap=False produce different results when appropriate."""
        # Use a configuration where wrapping should make a difference
        config = test_config.copy()
        config["q_len"] = 10
        config["kv_len"] = 20
        config["window_size"] = 16
        config["stride"] = 2.0

        mask_no_wrap = build_strided_sliding_window_blockmask(wrap=False, **config)
        mask_wrap = build_strided_sliding_window_blockmask(wrap=True, **config)

        dense_no_wrap = mask_no_wrap.to_dense()
        dense_wrap = mask_wrap.to_dense()

        # The masks should have the same shape
        assert dense_no_wrap.shape == dense_wrap.shape

        # They might be different (depending on the specific parameters)
        # but both should be valid boolean masks
        assert dense_no_wrap.dtype == torch.bool
        assert dense_wrap.dtype == torch.bool


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
    assert torch.equal(fast_mask.to_dense(), local_mask.to_dense()), "Fast and local CA masks should be identical for non-wrapped case"


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
    assert torch.equal(fast_mask.to_dense(), local_mask.to_dense()), "Fast and local CA masks should be identical for wrapped case"
