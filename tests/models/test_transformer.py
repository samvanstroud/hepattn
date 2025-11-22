import pytest
import torch
from lightning.pytorch.cli import instantiate_class
from torch import Tensor, nn

from hepattn.models import DropPath, Encoder, EncoderLayer, LayerScale, Residual
from hepattn.models.encoder import change_attn_backends

HAS_GPU = torch.cuda.is_available()
ATTN_TYPES_GPU = {"flex", "flash", "flash-varlen"}
DEVICE = "cuda" if HAS_GPU else "cpu"


# Fixtures for common inputs
@pytest.fixture
def input_tensor():
    return torch.rand(8, 130, 128)  # (batch_size, seq_len, dim)


# Tests for DropPath
def test_droppath_no_drop(input_tensor):
    model = DropPath(drop_prob=0.0)
    model.eval()  # Ensure not training
    output = model(input_tensor)
    assert torch.equal(output, input_tensor)


def test_droppath_with_drop(input_tensor):
    model = DropPath(drop_prob=0.5)
    model.train()  # Ensure training mode
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert not torch.equal(output, input_tensor)  # Values should differ due to drop


# Tests for LayerScale
def test_layerscale(input_tensor):
    model = LayerScale(dim=input_tensor.shape[-1], init_value=0.1)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.allclose(output, input_tensor * 0.1)


# Tests for Residual
def test_residual(input_tensor):
    fn = nn.Linear(input_tensor.shape[-1], input_tensor.shape[-1])
    dim = input_tensor.shape[-1]
    model = Residual(fn=fn, norm=nn.LayerNorm(dim), layer_scale=1e-5, drop_path=0.0, dim=dim)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


def test_residual_post_norm(input_tensor):
    """Test Residual with post_norm=True."""
    fn = nn.Linear(input_tensor.shape[-1], input_tensor.shape[-1])
    dim = input_tensor.shape[-1]
    model = Residual(fn=fn, norm=nn.LayerNorm(dim), post_norm=True, dim=dim)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


def test_residual_kv_norm(input_tensor):
    """Test Residual with kv_norm=True."""
    fn = nn.Linear(input_tensor.shape[-1], input_tensor.shape[-1])
    dim = input_tensor.shape[-1]
    model = Residual(fn=fn, norm=nn.LayerNorm(dim), kv_norm=True, dim=dim)
    kv = torch.rand_like(input_tensor)
    output = model(input_tensor, kv=kv)
    assert output.shape == input_tensor.shape


def test_residual_kv_norm_without_kv_raises():
    """Test that kv_norm=True raises error when kv is not provided."""
    fn = nn.Linear(128, 128)
    model = Residual(fn=fn, norm=nn.LayerNorm(128), kv_norm=True, dim=128)
    x = torch.randn(8, 130, 128)
    with pytest.raises(ValueError, match="kv_norm is enabled but no 'kv' argument was provided"):
        model(x)


def test_residual_validation_kv_norm_without_norm():
    """Test that kv_norm=True without norm raises ValueError."""
    fn = nn.Linear(128, 128)
    with pytest.raises(ValueError, match="kv_norm is True but no norm is provided"):
        Residual(fn=fn, norm=None, kv_norm=True, dim=128)


def test_residual_validation_post_norm_without_norm():
    """Test that post_norm=True without norm raises ValueError."""
    fn = nn.Linear(128, 128)
    with pytest.raises(ValueError, match="post_norm is True but no norm is provided"):
        Residual(fn=fn, norm=None, post_norm=True, dim=128)


def test_residual_validation_kv_norm_and_post_norm():
    """Test that kv_norm and post_norm cannot both be True."""
    fn = nn.Linear(128, 128)
    with pytest.raises(ValueError, match="kv_norm and post_norm cannot both be True"):
        Residual(fn=fn, norm=nn.LayerNorm(128), kv_norm=True, post_norm=True, dim=128)


# Tests for EncoderLayer
def test_encoderlayer(input_tensor):
    dim = input_tensor.shape[-1]
    model = EncoderLayer(dim=dim, drop_path=0.0, layer_scale=1e-5)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


def test_encoderlayer_with_kwargs(input_tensor):
    dim = input_tensor.shape[-1]
    model = EncoderLayer(dim=dim, drop_path=0.1, attn_kwargs={"num_heads": 4})
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


def test_encoderlayer_hybrid_norm(input_tensor):
    """Test EncoderLayer with hybrid_norm=True."""
    dim = input_tensor.shape[-1]
    # Test first layer (depth=0) - should have norm before attention
    model_first = EncoderLayer(dim=dim, depth=0, hybrid_norm=True, norm=nn.LayerNorm(dim))
    output_first = model_first(input_tensor)
    assert output_first.shape == input_tensor.shape
    assert model_first.attn.norm is not None  # Should have norm
    assert not model_first.dense.post_norm  # Should not have post_norm

    # Test subsequent layer (depth>0) - should have post_norm for dense
    model_subsequent = EncoderLayer(dim=dim, depth=1, hybrid_norm=True, norm=nn.LayerNorm(dim))
    output_subsequent = model_subsequent(input_tensor)
    assert output_subsequent.shape == input_tensor.shape
    assert model_subsequent.attn.norm is None  # Should not have norm before attention
    assert model_subsequent.dense.post_norm  # Should have post_norm


def test_encoderlayer_qkv_norm(input_tensor):
    """Test EncoderLayer with qkv_norm=True."""
    dim = input_tensor.shape[-1]
    model = EncoderLayer(dim=dim, qkv_norm=True, norm=nn.LayerNorm(dim))
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


# Tests for Encoder
def test_encoder_forward(input_tensor):
    model = Encoder(num_layers=3, dim=input_tensor.shape[-1])
    output = model(input_tensor)
    assert isinstance(output, Tensor)
    assert output.shape == input_tensor.shape
    assert output.sum() != 0
    assert not torch.isnan(output).any()


@pytest.mark.skip(reason="Flex currently not fully implemented.")
def test_dynamic_shape_block_mask():
    model = Encoder(num_layers=3, dim=128, window_size=10, attn_kwargs={"attn_type": "flex", "torch_compile": True}).cuda()
    xs = [torch.randn(8, i, 128, device="cuda") for i in range(100, 110)]

    for x in xs:
        out = model(x)
        assert out.shape == x.shape
        assert out.sum() != 0
        assert not torch.isnan(out).any()


def test_value_residuals():
    model = Encoder(num_layers=3, dim=128, value_residual=True)
    x = torch.randn(8, 100, 128, device="cpu")
    out = model(x)
    assert out.shape == x.shape
    assert out.sum() != 0
    assert not torch.isnan(out).any()


def test_register_tokens():
    batch_size, seq_len, dim = 8, 100, 128
    num_register_tokens = 5

    # Test with register tokens - they should be removed by default
    model = Encoder(num_layers=3, dim=dim, num_register_tokens=num_register_tokens)
    x = torch.randn(batch_size, seq_len, dim)
    out = model(x)

    # Output should be same size as input (register tokens removed)
    assert out.shape == x.shape
    assert out.sum() != 0
    assert not torch.isnan(out).any()

    # Test with kv_mask - register tokens should be prepended to the mask
    kv_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Set some positions to False to test the mask
    kv_mask[:, -10:] = False

    out_with_mask = model(x, kv_mask=kv_mask)
    assert out_with_mask.shape == x.shape
    assert out_with_mask.sum() != 0
    assert not torch.isnan(out_with_mask).any()

    # Test without register tokens (should be unchanged)
    model_no_reg = Encoder(num_layers=3, dim=dim, num_register_tokens=None)
    out_no_reg = model_no_reg(x)
    assert out_no_reg.shape == x.shape

    # Test incompatibility with window attention
    with pytest.raises(AssertionError, match="Register tokens are not compatible with window attention"):
        Encoder(num_layers=3, dim=dim, num_register_tokens=num_register_tokens, window_size=10)


def test_register_tokens_with_varlen():
    batch_size, seq_len, dim = 8, 100, 128
    num_register_tokens = 5

    # Test with register tokens and varlen attention
    model = Encoder(num_layers=3, dim=dim, num_register_tokens=num_register_tokens, attn_kwargs={"attn_type": "flash-varlen"})
    x = torch.randn(batch_size, seq_len, dim)
    out = model(x)

    # Output should be same size as input (register tokens removed)
    assert out.shape == x.shape
    assert out.sum() != 0
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    ("attn_type", "attn_type_new"),
    [
        ("torch", "flash"),
        ("flash", "flex"),
        ("flex", "torch"),
        ("flash-varlen", "torch"),
        ("torch", "flash-varlen"),
    ],
)
def test_encoder_change_backends(attn_type, attn_type_new):
    if not HAS_GPU and (attn_type in ATTN_TYPES_GPU or attn_type_new in ATTN_TYPES_GPU):
        pytest.skip("Skipping GPU-specific test on CPU-only environment")
    model = Encoder(num_layers=3, dim=128, attn_type=attn_type).to(DEVICE).half()
    x_a = x_b = torch.randn(8, 128, 128, device=DEVICE).half()
    kv_mask = torch.full((8, x_a.shape[-2]), True, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        out = model(x_a, kv_mask=kv_mask if attn_type == "flash-varlen" else None)
        change_attn_backends(model, attn_type_new)
        out_new = model(x_b, kv_mask=kv_mask if attn_type_new == "flash-varlen" else None)

    assert out_new.shape == x_a.shape

    # We allow this tolerance because of fp16 precision issues
    torch.testing.assert_close(out, out_new, atol=5e-3, rtol=5e-3)


def test_encoder_with_custom_norm(input_tensor):
    """Test Encoder with custom normalization module."""
    dim = input_tensor.shape[-1]
    custom_norm = nn.LayerNorm(dim)
    model = Encoder(num_layers=3, dim=dim, norm=custom_norm)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert not torch.isnan(output).any()


def test_encoder_with_none_norm(input_tensor):
    """Test Encoder with norm=None (should default to LayerNorm)."""
    dim = input_tensor.shape[-1]
    model = Encoder(num_layers=3, dim=dim, norm=None)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert not torch.isnan(output).any()


def test_encoder_with_lightning_cli_dict(input_tensor):
    """Test Encoder with Lightning CLI dict format for norm."""
    dim = input_tensor.shape[-1]
    norm_dict = {"class_path": "torch.nn.LayerNorm", "init_args": {"normalized_shape": dim}}
    # We need to ignore type check here because we're intentionally passing a dict
    # to test the internal handling, even though type hint says Module | None
    model = Encoder(num_layers=3, dim=dim, norm=norm_dict)  # type: ignore[arg-type]
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert not torch.isnan(output).any()
    # Verify it was instantiated correctly in the layers
    assert isinstance(model.layers[0].attn.norm, nn.LayerNorm)


def test_encoderlayer_with_lightning_cli_dict():
    """Test that EncoderLayer can handle Lightning CLI dict format for norm internally."""
    dim = 128
    norm_dict = {"class_path": "torch.nn.LayerNorm", "init_args": {"normalized_shape": dim}}
    # Instantiate the norm from dict format as Lightning CLI would
    norm = instantiate_class((), norm_dict)
    model = EncoderLayer(dim=dim, norm=norm)
    x = torch.randn(8, 130, dim)
    output = model(x)
    assert output.shape == x.shape
    # Verify it's actually a LayerNorm
    assert isinstance(model.attn.norm, nn.LayerNorm)


def test_encoderlayer_with_custom_norm():
    """Test EncoderLayer with custom norm module."""
    dim = 128
    custom_norm = nn.RMSNorm(dim)
    model = EncoderLayer(dim=dim, norm=custom_norm)
    x = torch.randn(8, 130, dim)
    output = model(x)
    assert output.shape == x.shape


def test_encoderlayer_norm_none_defaults_to_layernorm():
    """Test that EncoderLayer with norm=None defaults to LayerNorm."""
    dim = 128
    model = EncoderLayer(dim=dim, norm=None)
    x = torch.randn(8, 130, dim)
    output = model(x)
    assert output.shape == x.shape
    # Check that a LayerNorm was created
    assert isinstance(model.attn.norm, nn.LayerNorm)
