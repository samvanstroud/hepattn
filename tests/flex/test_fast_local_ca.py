import pytest
import torch

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

        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.int32)

        # Check output shapes
        assert kv_num_blocks.shape == (q_blocks,)
        assert kv_indices.shape == (q_blocks, kv_blocks)

        # Check that all values are non-negative
        assert torch.all(kv_num_blocks >= 0)
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

        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.int32)

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

        kv_num_blocks, kv_indices = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.int32)

        # Check output shapes
        assert kv_num_blocks.shape == (q_blocks,)
        assert kv_indices.shape == (q_blocks, kv_blocks)

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

        kv_num_blocks, kv_indices = _kv_blocks_wrap(q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.int32)

        # Should have blocks visible due to wrapping
        assert torch.all(kv_num_blocks > 0)
        # Check that indices are valid
        assert torch.all(kv_indices >= 0)

        # TODO: insert expected values


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
