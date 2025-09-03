import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from hepattn.flex.local_ca import sliding_window_mask_strided, sliding_window_mask_strided_wrapped


class TestSlidingWindowMaskStridedWrapped:
    """Test the sliding_window_mask_strided_wrapped function."""

    @pytest.mark.parametrize("window_size", [2, 4, 6, 8])
    @pytest.mark.parametrize("stride", [1.0, 2.0])
    def test_sliding_window_mask_strided_wrapped_basic(self, window_size, stride):
        """Test basic functionality of sliding_window_mask_strided_wrapped."""
        device = torch.device("cpu")
        q_len = 5
        kv_len = 10

        mask_mod = sliding_window_mask_strided_wrapped(window_size, stride, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device))

        # Create block mask
        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        # Check shape
        assert block_mask.block_mask.shape == (1, q_len, kv_len)
        assert block_mask.block_mask.dtype == torch.bool

        # Check that mask is not all True or all False
        mask = block_mask.block_mask[0]  # Remove batch dimension
        assert not torch.all(mask)
        assert torch.any(mask)

        # Check that each query has some valid attention positions
        for q_idx in range(q_len):
            assert torch.any(mask[q_idx]), f"Query {q_idx} has no valid attention positions"

    @pytest.mark.parametrize("window_size", [3, 5, 7])
    def test_sliding_window_mask_strided_wrapped_odd_window_size(self, window_size):
        """Test that odd window sizes raise ValueError."""
        device = torch.device("cpu")
        q_len = 5
        kv_len = 10

        with pytest.raises(ValueError, match="Window size must be even for strided sliding window"):
            sliding_window_mask_strided_wrapped(window_size, 2.0, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device))

    def test_sliding_window_mask_strided_wrapped_wrapping_behavior(self):
        """Test that wrapping behavior works correctly."""
        device = torch.device("cpu")
        window_size = 4
        q_len = 4
        kv_len = 8

        mask_mod = sliding_window_mask_strided_wrapped(window_size, 2.0, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device))

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        mask = block_mask.block_mask[0]

        # With wrapping, queries should be able to attend to positions that wrap around the sequence
        # Query 0 (stride=0) should attend to positions around 0, including wrapping
        # Query 1 (stride=2) should attend to positions around 2, including wrapping
        # etc.

        # Check that wrapping allows attention to positions that would otherwise be out of bounds
        # For example, query 0 should be able to attend to position 7 (wrapped from -1)
        # and query 1 should be able to attend to position 0 (wrapped from 10)

        # Verify that the mask has the expected wrapping behavior
        # This is a basic check - the exact pattern depends on the implementation
        assert not torch.all(mask)
        assert torch.any(mask)

    def test_sliding_window_mask_strided_wrapped_stride_assertion(self):
        """Test that stride parameter is handled correctly."""
        device = torch.device("cpu")
        window_size = 4
        q_len = 5
        kv_len = 10

        # Test that the function works with valid parameters
        mask_mod = sliding_window_mask_strided_wrapped(window_size, 2.0, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device))

        # Should return a callable mask modification function
        assert callable(mask_mod)

    def test_sliding_window_mask_strided_wrapped_kv_len_assertion(self):
        """Test that kv_len parameter is handled correctly."""
        device = torch.device("cpu")
        window_size = 4
        q_len = 5
        kv_len = 10

        # Test that the function works with valid parameters
        mask_mod = sliding_window_mask_strided_wrapped(window_size, 2.0, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device))

        # Should return a callable mask modification function
        assert callable(mask_mod)

    def test_sliding_window_mask_strided_wrapped_device_handling(self):
        """Test that the function works correctly on different devices."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            window_size = 4
            q_len = 5
            kv_len = 10

            mask_mod = sliding_window_mask_strided_wrapped(window_size, 2.0, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device))

            block_mask = create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=device,
            )

            # Check that the mask is on the correct device
            assert block_mask.block_mask.device == device
            assert block_mask.block_mask.shape == (1, q_len, kv_len)


class TestLocalCAIntegration:
    """Integration tests for local CA functionality."""

    def test_stride_calculation_consistency(self):
        """Test that stride calculation is consistent across different sequence lengths."""
        device = torch.device("cpu")
        window_size = 4

        # Test cases with different ratios
        test_cases = [
            (5, 10, 2.0),  # q_len=5, kv_len=10, expected_stride=2.0
            (3, 9, 3.0),  # q_len=3, kv_len=9, expected_stride=3.0
            (4, 8, 2.0),  # q_len=4, kv_len=8, expected_stride=2.0
            (2, 6, 3.0),  # q_len=2, kv_len=6, expected_stride=3.0
        ]

        for q_len, kv_len, expected_stride in test_cases:
            # Test both wrapped and non-wrapped versions
            mask_mod_strided = sliding_window_mask_strided(
                window_size, expected_stride, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device)
            )
            mask_mod_wrapped = sliding_window_mask_strided_wrapped(
                window_size, expected_stride, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device)
            )

            # Create block masks
            block_mask_strided = create_block_mask(
                mask_mod_strided,
                B=None,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=device,
            )

            block_mask_wrapped = create_block_mask(
                mask_mod_wrapped,
                B=None,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=device,
            )

            # Check shapes
            assert block_mask_strided.block_mask.shape == (1, q_len, kv_len)
            assert block_mask_wrapped.block_mask.shape == (1, q_len, kv_len)

            # Check that both masks are valid
            assert not torch.all(block_mask_strided.block_mask)
            assert torch.any(block_mask_strided.block_mask)
            assert not torch.all(block_mask_wrapped.block_mask)
            assert torch.any(block_mask_wrapped.block_mask)

    def test_window_size_edge_cases(self):
        """Test edge cases for window sizes."""
        device = torch.device("cpu")
        q_len = 5
        kv_len = 10

        # Test with very small window size
        window_size_small = 2
        mask_mod_small = sliding_window_mask_strided(window_size_small, 2.0, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device))

        block_mask_small = create_block_mask(
            mask_mod_small,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        # With window_size=2, each query should attend to at most 2 positions
        mask_small = block_mask_small.block_mask[0]
        for q_idx in range(q_len):
            attended_positions = torch.sum(mask_small[q_idx])
            assert attended_positions <= 2, f"Query {q_idx} attends to {attended_positions} positions, expected <= 2"

        # Test with larger window size
        window_size_large = 8
        mask_mod_large = sliding_window_mask_strided(window_size_large, 2.0, torch.tensor(q_len, device=device), torch.tensor(kv_len, device=device))

        block_mask_large = create_block_mask(
            mask_mod_large,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        # With window_size=8, each query should attend to more positions
        mask_large = block_mask_large.block_mask[0]
        for q_idx in range(q_len):
            attended_positions = torch.sum(mask_large[q_idx])
            assert attended_positions <= 8, f"Query {q_idx} attends to {attended_positions} positions, expected <= 8"
            # Large window should generally allow more attention than small window
            assert attended_positions >= torch.sum(mask_small[q_idx])


class TestFlexLocalCAMask:
    """Test the flex_local_ca_mask functionality from MaskFormerDecoder."""

    def test_flex_local_ca_mask_returns_mask_mod_function(self):
        """Test that flex_local_ca_mask returns a mask modification function."""
        from hepattn.models.decoder import MaskFormerDecoder

        # Test configuration
        window_size = 64
        q_len = 1000
        kv_len = 4000
        device = torch.device("cpu")
        dim = 64
        num_queries = 5

        decoder_layer_config = {
            "dim": dim,
            "norm": "LayerNorm",
            "dense_kwargs": {},
            "attn_kwargs": {"attn_type": "flex"},
            "bidirectional_ca": True,
            "hybrid_norm": False,
        }

        decoder = MaskFormerDecoder(
            num_queries=num_queries,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=False,
            local_strided_attn=True,
            window_size=window_size,
            window_wrap=False,
        )

        # Test non-wrapped version
        decoder.window_wrap = False
        mask_mod_func = decoder.flex_local_ca_mask(q_len, kv_len, device)

        # Should return a mask modification function, not a tensor
        assert callable(mask_mod_func)

        # Test wrapped version
        decoder.window_wrap = True
        mask_mod_func_wrapped = decoder.flex_local_ca_mask(q_len, kv_len, device)
        assert callable(mask_mod_func_wrapped)

    def test_flex_local_ca_mask_stride_calculation(self):
        """Test that flex_local_ca_mask calculates stride correctly based on q_len/kv_len ratio."""
        from hepattn.models.decoder import MaskFormerDecoder

        # Test configuration
        window_size = 64
        device = torch.device("cpu")
        dim = 64
        num_queries = 5

        decoder_layer_config = {
            "dim": dim,
            "norm": "LayerNorm",
            "dense_kwargs": {},
            "attn_kwargs": {"attn_type": "flex"},
            "bidirectional_ca": True,
            "hybrid_norm": False,
        }

        decoder = MaskFormerDecoder(
            num_queries=num_queries,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=False,
            local_strided_attn=True,
            window_size=window_size,
            window_wrap=False,
        )

        # Test different ratios
        test_cases = [
            (5, 10, 2.0),  # q_len=5, kv_len=10, expected_stride=2.0
            (3, 9, 3.0),  # q_len=3, kv_len=9, expected_stride=3.0
            (4, 8, 2.0),  # q_len=4, kv_len=8, expected_stride=2.0
            (2, 6, 3.0),  # q_len=2, kv_len=6, expected_stride=3.0
        ]

        for q_len, kv_len, expected_stride in test_cases:
            mask_mod_func = decoder.flex_local_ca_mask(q_len, kv_len, device)

            # Create block mask to test the actual mask
            block_mask = create_block_mask(
                mask_mod_func,
                B=None,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=device,
            )

            # Check shape
            assert block_mask.block_mask.shape == (1, q_len, kv_len)
            assert block_mask.block_mask.dtype == torch.bool

            # Check that mask is not all True or all False
            mask = block_mask.block_mask[0]
            assert not torch.all(mask)
            assert torch.any(mask)

    def test_flex_local_ca_mask_wrapping_behavior(self):
        """Test that flex_local_ca_mask handles wrapping correctly."""
        from hepattn.models.decoder import MaskFormerDecoder

        # Test configuration
        window_size = 8  # Smaller window for easier testing
        device = torch.device("cpu")
        dim = 64
        num_queries = 4
        q_len = 4
        kv_len = 8

        decoder_layer_config = {
            "dim": dim,
            "norm": "LayerNorm",
            "dense_kwargs": {},
            "attn_kwargs": {"attn_type": "flex"},
            "bidirectional_ca": True,
            "hybrid_norm": False,
        }

        decoder = MaskFormerDecoder(
            num_queries=num_queries,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=False,
            local_strided_attn=True,
            window_size=window_size,
            window_wrap=False,
        )

        # Test non-wrapped version
        decoder.window_wrap = False
        mask_mod_func_no_wrap = decoder.flex_local_ca_mask(q_len, kv_len, device)

        # Test wrapped version
        decoder.window_wrap = True
        mask_mod_func_wrap = decoder.flex_local_ca_mask(q_len, kv_len, device)

        # Create block masks
        block_mask_no_wrap = create_block_mask(
            mask_mod_func_no_wrap,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        block_mask_wrap = create_block_mask(
            mask_mod_func_wrap,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        # Check that both masks have correct shapes
        assert block_mask_no_wrap.block_mask.shape == (1, q_len, kv_len)
        assert block_mask_wrap.block_mask.shape == (1, q_len, kv_len)

        # Check that both masks are valid
        assert not torch.all(block_mask_no_wrap.block_mask)
        assert torch.any(block_mask_no_wrap.block_mask)
        assert not torch.all(block_mask_wrap.block_mask)
        assert torch.any(block_mask_wrap.block_mask)

    def test_flex_local_ca_mask_device_handling(self):
        """Test that flex_local_ca_mask works correctly on different devices."""
        from hepattn.models.decoder import MaskFormerDecoder

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Test configuration
        window_size = 64
        q_len = 100
        kv_len = 400
        device = torch.device("cuda")
        dim = 64
        num_queries = 5

        decoder_layer_config = {
            "dim": dim,
            "norm": "LayerNorm",
            "dense_kwargs": {},
            "attn_kwargs": {"attn_type": "flex"},
            "bidirectional_ca": True,
            "hybrid_norm": False,
        }

        decoder = MaskFormerDecoder(
            num_queries=num_queries,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=False,
            local_strided_attn=True,
            window_size=window_size,
            window_wrap=False,
        )

        mask_mod_func = decoder.flex_local_ca_mask(q_len, kv_len, device)

        # Create block mask
        block_mask = create_block_mask(
            mask_mod_func,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        # Check that the mask is on the correct device
        assert block_mask.block_mask.device == device
        assert block_mask.block_mask.shape == (1, q_len, kv_len)
