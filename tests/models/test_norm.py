import pytest
import torch

from hepattn.models.norm import (
    CustomLayerNorm,
    CustomRMSNorm,
    DyT,
    FastLayerNorm,
    SimpleRMSNorm,
    get_hybrid_norm_config,
)


@pytest.fixture
def sample_input():
    """Create a sample input tensor."""
    return torch.randn(2, 10, 64)


def test_custom_layer_norm(sample_input):
    """Test CustomLayerNorm forward pass."""
    norm = CustomLayerNorm(64)
    output = norm(sample_input)

    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()
    assert output.dtype == sample_input.dtype


def test_fast_layer_norm(sample_input):
    """Test FastLayerNorm forward pass."""
    norm = FastLayerNorm(64)
    output = norm(sample_input)

    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()
    assert output.dtype == sample_input.dtype


def test_custom_rms_norm(sample_input):
    """Test CustomRMSNorm forward pass."""
    norm = CustomRMSNorm(64)
    output = norm(sample_input)

    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()


def test_simple_rms_norm(sample_input):
    """Test SimpleRMSNorm forward pass."""
    norm = SimpleRMSNorm(64)
    output = norm(sample_input)

    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()
    assert output.dtype == sample_input.dtype


def test_dyt(sample_input):
    """Test DyT forward pass."""
    norm = DyT(64)
    output = norm(sample_input)

    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()


def test_dyt_with_custom_alpha():
    """Test DyT with custom alpha initialization."""
    norm = DyT(64, alpha_init_value=0.8)
    x = torch.randn(2, 10, 64)
    output = norm(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_get_hybrid_norm_config_no_hybrid():
    """Test get_hybrid_norm_config without hybrid normalization."""
    attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm="LayerNorm", depth=0, hybrid_norm=False, qkv_norm=False)

    assert attn_norm == "LayerNorm"
    assert dense_post_norm is False
    assert qkv_norm is False


def test_get_hybrid_norm_config_with_hybrid_depth_0():
    """Test get_hybrid_norm_config with hybrid normalization at depth 0."""
    attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm="LayerNorm", depth=0, hybrid_norm=True, qkv_norm=False)

    assert attn_norm == "LayerNorm"
    assert dense_post_norm is False
    assert qkv_norm is True


def test_get_hybrid_norm_config_with_hybrid_depth_1():
    """Test get_hybrid_norm_config with hybrid normalization at depth > 0."""
    attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm="LayerNorm", depth=1, hybrid_norm=True, qkv_norm=False)

    assert attn_norm is None
    assert dense_post_norm is True
    assert qkv_norm is True


def test_get_hybrid_norm_config_with_qkv_norm():
    """Test get_hybrid_norm_config with qkv_norm enabled."""
    attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm="LayerNorm", depth=0, hybrid_norm=False, qkv_norm=True)

    assert attn_norm == "LayerNorm"
    assert dense_post_norm is False
    assert qkv_norm is True


def test_norms_with_different_dtypes():
    """Test that norms preserve dtype for float16."""
    x_fp16 = torch.randn(2, 10, 64, dtype=torch.float16)

    # Test CustomLayerNorm
    norm1 = CustomLayerNorm(64)
    out1 = norm1(x_fp16)
    assert out1.dtype == torch.float16

    # Test FastLayerNorm
    norm2 = FastLayerNorm(64)
    out2 = norm2(x_fp16)
    assert out2.dtype == torch.float16

    # Test SimpleRMSNorm
    norm3 = SimpleRMSNorm(64)
    out3 = norm3(x_fp16)
    assert out3.dtype == torch.float16


def test_norms_with_batch_size_1():
    """Test norms with batch size 1."""
    x = torch.randn(1, 10, 64)

    norms = [
        CustomLayerNorm(64),
        FastLayerNorm(64),
        CustomRMSNorm(64),
        SimpleRMSNorm(64),
        DyT(64),
    ]

    for norm in norms:
        output = norm(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
