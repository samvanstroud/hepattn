import pytest
import torch
from torch import Tensor, nn

from hepattn.models import DropPath, Encoder, EncoderLayer, LayerScale, Residual
from hepattn.models.transformer import change_attn_backends


# Fixtures for common inputs
@pytest.fixture
def input_tensor():
    return torch.rand(8, 130, 128, device="cuda")  # (batch_size, seq_len, dim)


# Tests for DropPath
@pytest.mark.gpu
def test_droppath_no_drop(input_tensor):
    model = DropPath(drop_prob=0.0).cuda()
    model.eval()  # Ensure not training
    output = model(input_tensor)
    assert torch.equal(output, input_tensor)


@pytest.mark.gpu
def test_droppath_with_drop(input_tensor):
    model = DropPath(drop_prob=0.5).cuda()
    model.train()  # Ensure training mode
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert not torch.equal(output, input_tensor)  # Values should differ due to drop


# Tests for LayerScale
@pytest.mark.gpu
def test_layerscale(input_tensor):
    model = LayerScale(dim=input_tensor.shape[-1], init_value=0.1).cuda()
    output = model(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.allclose(output, input_tensor * 0.1)


# Tests for Residual
@pytest.mark.gpu
def test_residual(input_tensor):
    fn = nn.Linear(input_tensor.shape[-1], input_tensor.shape[-1]).cuda()
    model = Residual(fn=fn, norm="LayerNorm", layer_scale=1e-5, drop_path=0.0, dim=input_tensor.shape[-1]).cuda()
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


# Tests for EncoderLayer
@pytest.mark.gpu
def test_encoderlayer(input_tensor):
    dim = input_tensor.shape[-1]
    model = EncoderLayer(dim=dim, drop_path=0.0, layer_scale=1e-5).cuda()
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


@pytest.mark.gpu
def test_encoderlayer_with_kwargs(input_tensor):
    dim = input_tensor.shape[-1]
    model = EncoderLayer(dim=dim, drop_path=0.1, attn_kwargs={"num_heads": 4}).cuda()
    output = model(input_tensor)
    assert output.shape == input_tensor.shape


# Tests for Encoder
@pytest.mark.gpu
def test_encoder_forward(input_tensor):
    model = Encoder(num_layers=3, dim=input_tensor.shape[-1]).cuda()
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


@pytest.mark.gpu
def test_value_residuals():
    model = Encoder(num_layers=3, dim=128, value_residual=True).cuda()
    x = torch.randn(8, 100, 128, device="cuda")
    out = model(x)
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
    x = torch.randn(8, 128, 128, device="cuda").half()
    with torch.no_grad():
        out = model(x)
        # Change backend
        change_attn_backends(model, attn_type_new)
        out_new = model(x)
    assert out_new.shape == x.shape
    # We allow this tolerance because of fp16 precision issues
    torch.testing.assert_close(out, out_new, atol=5e-3, rtol=5e-3)
