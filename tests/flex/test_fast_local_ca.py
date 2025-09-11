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
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.int32
        )
        kv_num_blocks_wrap, kv_indices_wrap = _kv_blocks_wrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device, torch.int32
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
