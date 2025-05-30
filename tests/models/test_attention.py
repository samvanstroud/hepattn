import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_mask

from hepattn.flex.sliding_window import sliding_window_mask
from hepattn.models import Attention
from hepattn.models.attention import ATTN_MASK_ATTN_TYPES, VARLEN_ATTN_TYPES

torch.manual_seed(42)


# Choose primes so we don't get accidental broadcasting
# NOTE: We need enough keys/queries such that it is improbable that an entire k/q slot is masked
# if that happens then nans are produced
@pytest.mark.parametrize("batch_size", [1, 9])
@pytest.mark.parametrize("q_len", [100])
@pytest.mark.parametrize("kv_len", [None, 100, 150])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("kv_masking", [False, True])
@pytest.mark.parametrize("attn_masking", [False, True])
@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex", "flash-varlen"])
def test_attention_consistency(batch_size, dim, num_heads, bias, q_len, kv_len, attn_masking, kv_masking, attn_type):
    # Generate random input tensors
    q = torch.randn(batch_size, q_len, dim, dtype=torch.float16, device="cuda")

    # If kv_len is none, test self attenttion
    if kv_len:
        k = torch.randn(batch_size, kv_len, dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch_size, kv_len, dim, dtype=torch.float16, device="cuda")
    else:
        k = q
        v = q

    if kv_masking and attn_type in VARLEN_ATTN_TYPES:
        kv_mask = torch.randn(batch_size, k.shape[-2], dtype=torch.float16, device="cuda") >= 0.0
    else:
        kv_mask = None

    if attn_masking and attn_type in ATTN_MASK_ATTN_TYPES:
        attn_mask = torch.randn(batch_size, q.shape[-2], k.shape[-2], dtype=torch.float16, device="cuda") >= 0.0
    else:
        attn_mask = None

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
    custom_out = attention_layer(q, k, v, kv_mask=kv_mask, attn_mask=attn_mask)

    # Torch MHA expects (batch * heads, num_q, num_k) for attention mask, so have to repeat
    # It also expects the masking convention to be backwards
    torch_attn_mask = ~attn_mask.repeat_interleave(num_heads, dim=0) if attn_mask is not None else None

    torch_kv_mask = ~kv_mask if kv_mask is not None else None

    mha_out, _ = mha_layer(q, k, v, key_padding_mask=torch_kv_mask, attn_mask=torch_attn_mask)

    # Compare outputs
    torch.testing.assert_close(custom_out, mha_out, atol=1e-3, rtol=1e-3)


# NJT not working out of the box with flex, but can be done with a block mask
# for now just test with SDPA
def test_nested_jagged_tensor():
    attn_torch = Attention(dim=128, num_heads=8, attn_type="torch", torch_compile=False).cuda().half()
    # attn_flex = Attention(dim=128, num_heads=8, attn_type="flex", torch_compile=True).cuda().half()

    # Current limitation that the total sequnce length must be divisible by 128
    qs = [torch.randn(s, 128, dtype=torch.float16, device="cuda") for s in (128, 256)]
    ks = [torch.randn(s, 128, dtype=torch.float16, device="cuda") for s in (128, 256)]
    vs = [torch.randn(s, 128, dtype=torch.float16, device="cuda") for s in (128, 256)]

    nq = torch.nested.nested_tensor(qs, layout=torch.jagged, device="cuda", requires_grad=True)
    nk = torch.nested.nested_tensor(ks, layout=torch.jagged, device="cuda", requires_grad=True)
    nv = torch.nested.nested_tensor(vs, layout=torch.jagged, device="cuda", requires_grad=True)

    nt_out = attn_torch(nq, nk, nv)
    # flex_out = attn_flex(nq, nk, nv)

    # do the same but looping over the list
    for i, (q, k, v) in enumerate(zip(qs, ks, vs, strict=False)):
        out = attn_torch(q, k, v)
        torch.testing.assert_close(out, nt_out[i], atol=1e-3, rtol=1e-3)
        # torch.testing.assert_close(out, flex_out[i], atol=1e-3, rtol=1e-3)


def test_local_attention():
    window_size = 4

    # Generate random input tensors
    q = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")

    # Initialize attention layers
    attn_spda = Attention(dim=128, num_heads=8, attn_type="torch", torch_compile=False, bias=False).cuda().half()
    # attn_flex = Attention(dim=128, num_heads=8, attn_type="flex", torch_compile=True, bias=False).cuda().half()
    attn_flash = Attention(dim=128, num_heads=8, attn_type="flash", torch_compile=False, window_size=window_size, bias=False).cuda().half()

    # Synchronize weights for comparison
    # attn_flex.q_proj.weight.data = attn_spda.q_proj.weight
    # attn_flex.k_proj.weight.data = attn_spda.k_proj.weight
    # attn_flex.v_proj.weight.data = attn_spda.v_proj.weight
    # attn_flex.out_proj.weight.data = attn_spda.out_proj.weight
    attn_flash.q_proj.weight.data = attn_spda.q_proj.weight
    attn_flash.k_proj.weight.data = attn_spda.k_proj.weight
    attn_flash.v_proj.weight.data = attn_spda.v_proj.weight
    attn_flash.out_proj.weight.data = attn_spda.out_proj.weight

    mask_mod = sliding_window_mask(window_size)
    q_len = q.shape[-2]
    # block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len, device=q.device)
    mask = create_mask(mask_mod, 1, None, q_len, q_len, device=q.device)
    # out_flex = attn_flex(q, k, v, attn_mask=block_mask)
    # Squeeze operation is required as for SPDA attention we assume mask is the same accross heads
    # TODO: Standardise this accross the different backends, both for whether it should brodcast the
    # shape over heads and whether it should assume masks are true for valid slots or not
    out_spda = attn_spda(q, k, v, attn_mask=mask.squeeze(1))
    out_flash = attn_flash(q, k, v)

    # Compare outputs
    torch.testing.assert_close(out_spda, out_flash, atol=1e-3, rtol=1e-3)
    # torch.testing.assert_close(out_flex, out_flash, atol=1e-3, rtol=1e-3)


def test_flex_dynamic():
    # generate inputs
    xs = [torch.randn(1, i, 128, dtype=torch.float16, device="cuda") for i in range(100, 110)]

    # Initialize attention layers
    attn = Attention(dim=128, num_heads=8, attn_type="flex", torch_compile=True, bias=False).cuda().half()

    # loop over inputs
    for x in xs:
        out = attn(x, x, x)
        assert out.shape == x.shape


@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex", "flash-varlen"])
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
