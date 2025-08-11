import pytest
import torch
from torch import Tensor, nn

from hepattn.models import DropPath, Encoder, EncoderLayer, LayerScale, Residual
from hepattn.models.encoder import change_attn_backends


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
    model = Residual(fn=fn, norm="LayerNorm", layer_scale=1e-5, drop_path=0.0, dim=input_tensor.shape[-1])
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


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
@pytest.mark.gpu
def test_encoder_change_backends(attn_type, attn_type_new):
    model = Encoder(num_layers=3, dim=128, attn_type=attn_type).cuda().half()
    x_a = x_b = torch.randn(8, 128, 128, device="cuda").half()
    kv_mask = torch.full((8, x_a.shape[-2]), True, dtype=torch.bool, device="cuda")

    with torch.no_grad():
        out = model(x_a, kv_mask=kv_mask if attn_type == "flash-varlen" else None)
        change_attn_backends(model, attn_type_new)
        out_new = model(x_b, kv_mask=kv_mask if attn_type_new == "flash-varlen" else None)

    assert out_new.shape == x_a.shape

    # We allow this tolerance because of fp16 precision issues
    torch.testing.assert_close(out, out_new, atol=5e-3, rtol=5e-3)
