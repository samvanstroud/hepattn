import torch
from torch.nn.attention.flex_attention import create_mask

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.utils.local_ca import auto_local_ca_mask


def test_flex_local_ca_mask_equivalence():
    """Test that flex_local_ca_mask produces equivalent masks to torch.nn.attention.flex_attention.create_mask."""
    # Test configuration
    window_size = 16
    q_len = 100
    kv_len = 1000
    device = torch.device("cpu")
    dim = 64

    # dummy query and keys embeds with correct shape[1]
    query_embed = torch.randn(1, q_len)
    key_embed = torch.randn(1, kv_len)

    # Create decoder with flex attention
    decoder_layer_config = {
        "dim": dim,
        "norm": "LayerNorm",
        "dense_kwargs": {},
        "attn_kwargs": {"attn_type": "flex"},
        "bidirectional_ca": True,
        "hybrid_norm": False,
    }

    decoder = MaskFormerDecoder(
        num_queries=q_len,
        decoder_layer_config=decoder_layer_config,
        num_decoder_layers=1,
        mask_attention=False,
        local_strided_attn=True,
        window_size=window_size,
        window_wrap=False,
    )

    # Test non-wrapped version
    decoder.window_wrap = False

    attn_mask_torch = auto_local_ca_mask(query_embed, key_embed, decoder.window_size, wrap=decoder.window_wrap)
    # Get mask from decoder's flex_local_ca_mask method using create_mask
    decoder_mask_flex = create_mask(decoder.flex_local_ca_mask(q_len, kv_len, device))
    assert torch.allclose(attn_mask_torch, decoder_mask_flex)

    # Test wrapped version
    decoder.window_wrap = True
    attn_mask_torch = auto_local_ca_mask(query_embed, key_embed, decoder.window_size, wrap=decoder.window_wrap)
    # Get mask from decoder's flex_local_ca_mask method using create_mask
    decoder_mask_flex = create_mask(decoder.flex_local_ca_mask(q_len, kv_len, device))
    assert torch.allclose(attn_mask_torch, decoder_mask_flex)


def test_flex_local_ca_mask_transpose_consistency():
    """Test that flex_local_ca_mask produces consistent transposed masks for bidirectional attention."""
    # Test configuration
    window_size = 16
    q_len = 100
    kv_len = 1000
    device = torch.device("cpu")
    dim = 64

    decoder_layer_config = {
        "dim": dim,
        "norm": "LayerNorm",
        "dense_kwargs": {},
        "attn_kwargs": {"attn_type": "flex"},
        "bidirectional_ca": True,
        "hybrid_norm": False,
    }

    decoder = MaskFormerDecoder(
        num_queries=q_len,
        decoder_layer_config=decoder_layer_config,
        num_decoder_layers=1,
        mask_attention=False,
        local_strided_attn=True,
        window_size=window_size,
        window_wrap=False,
    )

    # Get forward and transpose mask modification functions
    transpose_forward_mask_mod = create_mask(decoder.flex_local_ca_mask(q_len, kv_len, device), 1, 1, q_len, kv_len, device).transpose(-2, -1)
    transpose_mask_mod = create_mask(decoder.flex_local_ca_mask(kv_len, q_len, device), 1, 1, kv_len, q_len, device)

    assert torch.allclose(transpose_forward_mask_mod, transpose_mask_mod)
