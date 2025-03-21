import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_mask

from hepattn.flex.sliding_window import sliding_window_mask
from hepattn.models import Attention

torch.manual_seed(42)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex"])
def test_attention_consistency(batch_size, seq_len, dim, num_heads, bias, attn_type):
    # Generate random input tensors
    q = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device="cuda")

    # Initialize attention layers
    attention_layer = Attention(dim=dim, num_heads=num_heads, bias=bias, attn_type=attn_type).cuda().half()
    mha_layer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True).cuda().half()

    # Synchronize weights for comparison
    attention_layer.q_proj.weight.data = mha_layer.in_proj_weight[:dim, :]
    attention_layer.k_proj.weight.data = mha_layer.in_proj_weight[dim : 2 * dim, :]
    attention_layer.v_proj.weight.data = mha_layer.in_proj_weight[2 * dim :, :]
    attention_layer.out_proj.weight.data = mha_layer.out_proj.weight
    if bias:
        attention_layer.q_proj.bias.data = mha_layer.in_proj_bias[:dim]
        attention_layer.k_proj.bias.data = mha_layer.in_proj_bias[dim : 2 * dim]
        attention_layer.v_proj.bias.data = mha_layer.in_proj_bias[2 * dim :]
        attention_layer.out_proj.bias.data = mha_layer.out_proj.bias

    # Compute outputs
    custom_out = attention_layer(q, k, v)
    mha_out, _ = mha_layer(q, k, v)

    # Compare outputs
    torch.testing.assert_close(custom_out, mha_out, atol=1e-3, rtol=1e-3)


# NJT not working out of the box with flex, but can be done with a block mask
# for now just test with SDPA
def test_nested_jagged_tensor():
    attn_torch = Attention(dim=128, num_heads=8, attn_type="torch", torch_compile=False).cuda().half()
    # attn_flex = Attention(dim=128, num_heads=8, attn_type="flex", torch_compile=True).cuda().half()  # noqa: ERA001

    # Current limitation that the total sequnce length must be divisible by 128
    qs = [torch.randn(s, 128, dtype=torch.float16, device="cuda") for s in (128, 256)]
    ks = [torch.randn(s, 128, dtype=torch.float16, device="cuda") for s in (128, 256)]
    vs = [torch.randn(s, 128, dtype=torch.float16, device="cuda") for s in (128, 256)]

    nq = torch.nested.nested_tensor(qs, layout=torch.jagged, device="cuda", requires_grad=True)
    nk = torch.nested.nested_tensor(ks, layout=torch.jagged, device="cuda", requires_grad=True)
    nv = torch.nested.nested_tensor(vs, layout=torch.jagged, device="cuda", requires_grad=True)

    nt_out = attn_torch(nq, nk, nv)
    # flex_out = attn_flex(nq, nk, nv)  # noqa: ERA001

    # do the same but looping over the list
    for i, (q, k, v) in enumerate(zip(qs, ks, vs, strict=False)):
        out = attn_torch(q, k, v)
        torch.testing.assert_close(out, nt_out[i], atol=1e-3, rtol=1e-3)
        # torch.testing.assert_close(out, flex_out[i], atol=1e-3, rtol=1e-3)  # noqa: ERA001


def test_local_attention():
    window_size = 4

    # Generate random input tensors
    q = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")

    # Initialize attention layers
    attn_spda = Attention(dim=128, num_heads=8, attn_type="torch", torch_compile=False, bias=False).cuda().half()
    # attn_flex = Attention(dim=128, num_heads=8, attn_type="flex", torch_compile=True, bias=False).cuda().half()  # noqa: ERA001
    attn_flash = Attention(dim=128, num_heads=8, attn_type="flash", torch_compile=False, window_size=window_size, bias=False).cuda().half()

    # Synchronize weights for comparison
    # attn_flex.q_proj.weight.data = attn_spda.q_proj.weight  # noqa: ERA001
    # attn_flex.k_proj.weight.data = attn_spda.k_proj.weight  # noqa: ERA001
    # attn_flex.v_proj.weight.data = attn_spda.v_proj.weight  # noqa: ERA001
    # attn_flex.out_proj.weight.data = attn_spda.out_proj.weight  # noqa: ERA001
    attn_flash.q_proj.weight.data = attn_spda.q_proj.weight
    attn_flash.k_proj.weight.data = attn_spda.k_proj.weight
    attn_flash.v_proj.weight.data = attn_spda.v_proj.weight
    attn_flash.out_proj.weight.data = attn_spda.out_proj.weight

    mask_mod = sliding_window_mask(window_size)
    q_len = q.shape[-2]
    # block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len, device=q.device)  # noqa: ERA001
    mask = create_mask(mask_mod, 1, 1, q_len, q_len, device=q.device)
    # out_flex = attn_flex(q, k, v, attn_mask=block_mask)  # noqa: ERA001
    out_spda = attn_spda(q, k, v, attn_mask=mask)
    out_flash = attn_flash(q, k, v)

    # Compare outputs
    torch.testing.assert_close(out_spda, out_flash, atol=1e-3, rtol=1e-3)
    # torch.testing.assert_close(out_flex, out_flash, atol=1e-3, rtol=1e-3)  # noqa: ERA001


def test_flex_dynamic():
    # generate inputs
    xs = [torch.randn(1, i, 128, dtype=torch.float16, device="cuda") for i in range(100, 110)]

    # Initialize attention layers
    attn = Attention(dim=128, num_heads=8, attn_type="flex", torch_compile=True, bias=False).cuda().half()

    # loop over inputs
    for x in xs:
        out = attn(x, x, x)
        assert out.shape == x.shape


@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex"])
def test_cross_attention(attn_type):
    # Generate random input tensors
    q = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 256, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 256, 128, dtype=torch.float16, device="cuda")

    # Initialize attention layers
    attn = Attention(dim=128, num_heads=8, attn_type=attn_type).cuda().half()

    # Compute outputs
    out = attn(q, k, v)

    # Check output shape
    assert out.shape == q.shape
