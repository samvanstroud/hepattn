import torch
from torch import nn

from hepattn.models.norm import DyT, LayerNorm, RMSNorm, SimpleRMSNorm, get_hybrid_norm_config


def test_layernorm():
    """Test custom LayerNorm implementation."""
    dim = 128
    norm = LayerNorm(dim)
    x = torch.randn(8, 100, dim)
    output = norm(x)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_rmsnorm():
    """Test RMSNorm implementation."""
    dim = 128
    norm = RMSNorm(dim)
    x = torch.randn(8, 100, dim)
    output = norm(x)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_simple_rmsnorm():
    """Test SimpleRMSNorm implementation."""
    dim = 128
    norm = SimpleRMSNorm(dim)
    x = torch.randn(8, 100, dim)
    output = norm(x)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_dyt():
    """Test DyT normalization."""
    dim = 128
    norm = DyT(dim)
    x = torch.randn(8, 100, dim)
    output = norm(x)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_dyt_custom_alpha():
    """Test DyT with custom alpha initialization."""
    dim = 128
    norm = DyT(dim, alpha_init_value=0.8)
    x = torch.randn(8, 100, dim)
    output = norm(x)
    assert output.shape == x.shape
    assert torch.allclose(norm.alpha, torch.tensor([0.8]))


def test_get_hybrid_norm_config_first_layer():
    """Test get_hybrid_norm_config for first layer (depth=0)."""
    norm = nn.LayerNorm(128)
    depth = 0
    hybrid_norm = True
    qkv_norm = False

    attn_norm, dense_post_norm, qkv_norm_out = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

    # First layer should have norm before attention
    assert attn_norm is norm
    assert dense_post_norm is False
    assert qkv_norm_out is True  # hybrid_norm enables qkv_norm


def test_get_hybrid_norm_config_subsequent_layer():
    """Test get_hybrid_norm_config for subsequent layers (depth>0)."""
    norm = nn.LayerNorm(128)
    depth = 1
    hybrid_norm = True
    qkv_norm = False

    attn_norm, dense_post_norm, qkv_norm_out = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

    # Subsequent layers should not have norm before attention but should have post_norm
    assert attn_norm is None
    assert dense_post_norm is True
    assert qkv_norm_out is True  # hybrid_norm enables qkv_norm


def test_get_hybrid_norm_config_no_hybrid():
    """Test get_hybrid_norm_config with hybrid_norm=False."""
    norm = nn.LayerNorm(128)
    depth = 1
    hybrid_norm = False
    qkv_norm = False

    attn_norm, dense_post_norm, qkv_norm_out = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

    # Without hybrid_norm, should have standard configuration
    assert attn_norm is norm
    assert dense_post_norm is False
    assert qkv_norm_out is False


def test_get_hybrid_norm_config_qkv_norm_only():
    """Test get_hybrid_norm_config with qkv_norm=True but hybrid_norm=False."""
    norm = nn.LayerNorm(128)
    depth = 0
    hybrid_norm = False
    qkv_norm = True

    attn_norm, dense_post_norm, qkv_norm_out = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

    # qkv_norm should be enabled
    assert attn_norm is norm
    assert dense_post_norm is False
    assert qkv_norm_out is True


def test_get_hybrid_norm_config_none_norm():
    """Test get_hybrid_norm_config with None norm."""
    norm = None
    depth = 0
    hybrid_norm = False
    qkv_norm = False

    attn_norm, dense_post_norm, qkv_norm_out = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

    assert attn_norm is None
    assert dense_post_norm is False
    assert qkv_norm_out is False


def test_get_hybrid_norm_config_none_norm_with_hybrid():
    """Test get_hybrid_norm_config with None norm and hybrid_norm=True."""
    norm = None
    depth = 1
    hybrid_norm = True
    qkv_norm = False

    attn_norm, dense_post_norm, qkv_norm_out = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

    # Even with hybrid_norm, if norm is None, attn_norm should be None
    assert attn_norm is None
    assert dense_post_norm is True  # post_norm should still be True for depth > 0
    assert qkv_norm_out is True  # hybrid_norm enables qkv_norm
