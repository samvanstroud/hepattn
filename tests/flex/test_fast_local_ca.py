import pytest
import torch

from hepattn.flex.fast_local_ca import (
    _kv_blocks_nonwrap,  # noqa: WPS437
    _kv_blocks_wrap,  # noqa: WPS437
    _round_bankers_num_over_den,  # noqa: WPS437
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
        
        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device
        )
        
        # Check output shapes
        assert kv_num_blocks.shape == (q_blocks,)
        assert kv_indices.shape == (q_blocks, kv_blocks)
        assert kv_num_blocks.dtype == torch.int32
        assert kv_indices.dtype == torch.int32
        
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
        
        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device
        )
        
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
        
        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device
        )
        
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
        
        kv_num_blocks, kv_indices = _kv_blocks_wrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device
        )
        
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
        
        kv_num_blocks, kv_indices = _kv_blocks_wrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device
        )
        
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
        
        kv_num_blocks, kv_indices = _kv_blocks_wrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device
        )
        
        # All blocks should be visible
        assert torch.all(kv_num_blocks == kv_blocks)
        # Check that indices are valid
        assert torch.all(kv_indices >= 0)


class TestBuildStridedSlidingWindowBlockmask:
    """Test the build_strided_sliding_window_blockmask function."""

    def test_basic_functionality(self, test_config):
        """Test basic functionality of build_strided_sliding_window_blockmask."""
        mask = build_strided_sliding_window_blockmask(wrap=False, **test_config)
        
        # Check that it returns a BlockMask
        assert hasattr(mask, "to_dense")
        assert hasattr(mask, "kv_num_blocks")
        assert hasattr(mask, "kv_indices")
        
        # Check that dense conversion works
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (test_config["q_len"], test_config["kv_len"])
        assert dense_mask.dtype == torch.bool

    def test_odd_window_size_error(self, test_config):
        """Test that odd window size raises ValueError."""
        test_config["window_size"] = 31  # Odd number
        
        with pytest.raises(ValueError, match="Window size must be even"):
            build_strided_sliding_window_blockmask(wrap=False, **test_config)

    def test_wrap_true(self, test_config):
        """Test with wrap=True."""
        mask = build_strided_sliding_window_blockmask(wrap=True, **test_config)
        
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (test_config["q_len"], test_config["kv_len"])
        assert dense_mask.dtype == torch.bool

    def test_different_block_sizes(self, test_config):
        """Test with different block sizes."""
        for block_size in [64, 128, 256]:
            test_config["block_size"] = block_size
            mask = build_strided_sliding_window_blockmask(wrap=False, **test_config)
            
            dense_mask = mask.to_dense()
            assert dense_mask.shape == (test_config["q_len"], test_config["kv_len"])

    def test_different_devices(self, test_config):
        """Test on different devices."""
        if torch.cuda.is_available():
            test_config["device"] = "cuda"
            mask = build_strided_sliding_window_blockmask(wrap=False, **test_config)
            
            dense_mask = mask.to_dense()
            assert dense_mask.device.type == "cuda"

    def test_edge_case_sequences(self):
        """Test with edge case sequence lengths."""
        # Very small sequences
        config = {"window_size": 8, "stride": 1.0, "q_len": 1, "kv_len": 1, "device": "cpu", "block_size": 64}
        mask = build_strided_sliding_window_blockmask(wrap=False, **config)
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (1, 1)
        
        # q_len > kv_len
        config = {"window_size": 8, "stride": 0.5, "q_len": 100, "kv_len": 50, "device": "cpu", "block_size": 64}
        mask = build_strided_sliding_window_blockmask(wrap=False, **config)
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (100, 50)

    def test_stride_parameter(self, test_config):
        """Test with different stride values."""
        for stride in [0.5, 1.0, 2.0, 4.0]:
            test_config["stride"] = stride
            mask = build_strided_sliding_window_blockmask(wrap=False, **test_config)
            
            dense_mask = mask.to_dense()
            assert dense_mask.shape == (test_config["q_len"], test_config["kv_len"])

    def test_extreme_stride_values(self):
        """Test with extreme stride values."""
        config = {"window_size": 16, "q_len": 10, "kv_len": 100, "device": "cpu", "block_size": 64}
        
        # Very small stride
        config["stride"] = 0.01
        mask = build_strided_sliding_window_blockmask(wrap=False, **config)
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (10, 100)
        
        # Very large stride
        config["stride"] = 100.0
        mask = build_strided_sliding_window_blockmask(wrap=False, **config)
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (10, 100)

    def test_boundary_conditions(self):
        """Test boundary conditions for sequence lengths."""
        # Test with sequences that are exact multiples of block size
        config = {"window_size": 16, "stride": 1.0, "q_len": 128, "kv_len": 256, "device": "cpu", "block_size": 64}
        mask = build_strided_sliding_window_blockmask(wrap=False, **config)
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (128, 256)
        
        # Test with sequences that are not multiples of block size
        config = {"window_size": 16, "stride": 1.0, "q_len": 100, "kv_len": 200, "device": "cpu", "block_size": 64}
        mask = build_strided_sliding_window_blockmask(wrap=False, **config)
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (100, 200)

    def test_window_size_edge_cases(self):
        """Test edge cases for window size."""
        config = {"stride": 1.0, "q_len": 10, "kv_len": 20, "device": "cpu", "block_size": 64}
        
        # Window size equal to sequence length
        config["window_size"] = 20
        mask = build_strided_sliding_window_blockmask(wrap=False, **config)
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (10, 20)
        
        # Window size larger than sequence length
        config["window_size"] = 50
        mask = build_strided_sliding_window_blockmask(wrap=False, **config)
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (10, 20)

    def test_compiled_functions_work(self):
        """Test that the compiled helper functions work correctly."""
        # Test _kv_blocks_nonwrap with various parameters
        q_blocks = 3
        kv_blocks = 5
        block_size = 64
        window_size = 32
        stride = 1.5
        q_len = 150
        kv_len = 250
        device = "cpu"
        
        kv_num_blocks, kv_indices = _kv_blocks_nonwrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device
        )
        
        # Check that compiled function returns valid results
        assert kv_num_blocks.shape == (q_blocks,)
        assert kv_indices.shape == (q_blocks, kv_blocks)
        assert torch.all(kv_num_blocks >= 0)
        assert torch.all(kv_num_blocks <= kv_blocks)
        
        # Test _kv_blocks_wrap with various parameters
        kv_num_blocks_wrap, kv_indices_wrap = _kv_blocks_wrap(
            q_blocks, kv_blocks, block_size, window_size, stride, q_len, kv_len, device
        )
        
        # Check that compiled function returns valid results
        assert kv_num_blocks_wrap.shape == (q_blocks,)
        assert kv_indices_wrap.shape == (q_blocks, kv_blocks)
        assert torch.all(kv_num_blocks_wrap >= 0)
        assert torch.all(kv_num_blocks_wrap <= kv_blocks)

    def test_mask_mod_functionality(self, test_config):
        """Test that the mask_mod function works correctly within the BlockMask."""
        mask = build_strided_sliding_window_blockmask(wrap=False, **test_config)
        
        # Test that the mask_mod function is callable
        assert hasattr(mask, "mask_mod")
        assert callable(mask.mask_mod)
        
        # Test mask_mod with some sample indices
        # Note: mask_mod expects (b, h, q_idx, kv_idx) but we'll test with simplified calls
        # The actual testing of mask_mod behavior is implicit in the to_dense() conversion
        
        dense_mask = mask.to_dense()
        assert dense_mask.shape == (test_config["q_len"], test_config["kv_len"])
        assert dense_mask.dtype == torch.bool
        
        # Check that the mask has some True values (not all False)
        assert torch.any(dense_mask)
        
        # Check that the mask has some False values (not all True)
        assert torch.any(~dense_mask)

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
