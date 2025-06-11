import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_mask

from hepattn.flex.sliding_window import sliding_window_mask
from hepattn.models import Attention
from hepattn.models.attention import ATTN_MASK_ATTN_TYPES, VARLEN_ATTN_TYPES

torch.manual_seed(42)


def copy_attention_weights(src: Attention, dst: Attention):
    dst.in_proj_weight.data.copy_(src.in_proj_weight.data)
    if src.in_proj_bias is not None:
        dst.in_proj_bias.data.copy_(src.in_proj_bias.data)
    dst.out_proj.weight.data.copy_(src.out_proj.weight.data)
    if src.out_proj.bias is not None:
        dst.out_proj.bias.data.copy_(src.out_proj.bias.data)


# Choose primes so we don't get accidental broadcasting
# NOTE: We need enough keys/queries such that it is improbable that an entire k/q slot is masked
# if that happens then nans are produced
@pytest.mark.parametrize("batch_size", [1, 9])
@pytest.mark.parametrize("q_len", [128])
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
    self_attn = False
    # If kv_len is none, test self attenttion
    if kv_len:
        kv = torch.randn(batch_size, kv_len, dim, dtype=torch.float16, device="cuda")
    else:
        kv = q
        self_attn = True

    if not self_attn and kv_masking and attn_type in VARLEN_ATTN_TYPES:
        kv_mask = torch.randn(batch_size, kv.shape[-2], dtype=torch.float16, device="cuda") >= 0.0
    else:
        kv_mask = None

    if attn_masking and attn_type in ATTN_MASK_ATTN_TYPES:
        attn_mask = torch.randn(batch_size, q.shape[-2], kv.shape[-2], dtype=torch.float16, device="cuda") >= 0.0
    else:
        attn_mask = None

    # Initialize attention layers
    attention_layer = Attention(dim=dim, num_heads=num_heads, bias=bias, attn_type=attn_type).cuda().half()
    mha_layer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True).cuda().half()

    # Synchronize weights for comparison
    copy_attention_weights(attention_layer, mha_layer)

    # Compute outputs
    custom_out = attention_layer(q, kv if not self_attn else None, kv_mask=kv_mask, attn_mask=attn_mask)

    # Torch MHA expects (batch * heads, num_q, num_k) for attention mask, so have to repeat
    # It also expects the masking convention to be backwards
    torch_attn_mask = ~attn_mask.repeat_interleave(num_heads, dim=0) if attn_mask is not None else None

    torch_kv_mask = ~kv_mask if kv_mask is not None else None

    mha_out, _ = mha_layer(q, kv, kv, key_padding_mask=torch_kv_mask, attn_mask=torch_attn_mask)

    # Compare outputs
    torch.testing.assert_close(custom_out, mha_out, atol=1e-3, rtol=1e-3)


# NJT not working out of the box with flex, but can be done with a block mask
# for now just test with SDPA
def test_nested_jagged_tensor():
    attn_torch = Attention(dim=128, num_heads=8, attn_type="torch", torch_compile=False).cuda().half()
    # attn_flex = Attention(dim=128, num_heads=8, attn_type="flex", torch_compile=True).cuda().half()

    # Current limitation that the total sequnce length must be divisible by 128
    qs = [torch.randn(s, 128, dtype=torch.float16, device="cuda") for s in (128, 256)]
    kvs = [torch.randn(s, 128, dtype=torch.float16, device="cuda") for s in (128, 256)]

    nq = torch.nested.nested_tensor(qs, layout=torch.jagged, device="cuda", requires_grad=True)
    nkv = torch.nested.nested_tensor(kvs, layout=torch.jagged, device="cuda", requires_grad=True)
    nt_out = attn_torch(nq, nkv)
    # flex_out = attn_flex(nq, nkv)

    # do the same but looping over the list
    for i, (q, kv) in enumerate(zip(qs, kvs, strict=False)):
        out = attn_torch(q, kv)
        torch.testing.assert_close(out, nt_out[i], atol=1e-3, rtol=1e-3)
        # torch.testing.assert_close(out, flex_out[i], atol=1e-3, rtol=1e-3)


def test_local_attention():
    window_size = 4

    # Generate random input tensors
    q = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")
    kv = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")

    # Initialize attention layers
    attn_spda = Attention(dim=128, num_heads=8, attn_type="torch", torch_compile=False, bias=False).cuda().half()
    # attn_flex = Attention(dim=128, num_heads=8, attn_type="flex", torch_compile=True, bias=False).cuda().half()
    attn_flash = Attention(dim=128, num_heads=8, attn_type="flash", torch_compile=False, window_size=window_size, bias=False).cuda().half()

    # Synchronize weights for comparison
    # copy_attention_weights(attn_spda, attn_flex)
    copy_attention_weights(attn_spda, attn_flash)

    mask_mod = sliding_window_mask(window_size)
    q_len = q.shape[-2]
    # block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len, device=q.device)
    mask = create_mask(mask_mod, 1, None, q_len, q_len, device=q.device)
    # out_flex = attn_flex(q, kv, attn_mask=block_mask)
    # Squeeze operation is required as for SPDA attention we assume mask is the same accross heads
    # TODO: Standardise this accross the different backends, both for whether it should brodcast the
    # shape over heads and whether it should assume masks are true for valid slots or not
    out_spda = attn_spda(q, kv, attn_mask=mask.squeeze(1))
    out_flash = attn_flash(q, kv)

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
        out = attn(x, x)
        assert out.shape == x.shape


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex", "flash-varlen"])
def test_self_attention(batch_size, seq_len, dim, num_heads, bias, attn_type):
    # Generate random input tensor
    qkv = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device="cuda")

    # Initialize attention layers
    attn = Attention(dim=dim, num_heads=num_heads, bias=bias, attn_type=attn_type).cuda().half()
    self_attn = Attention(dim=dim, num_heads=num_heads, bias=bias, attn_type=attn_type).cuda().half()

    # Synchronize weights for comparison
    copy_attention_weights(attn, self_attn)

    # Compute outputs
    out = attn(qkv, qkv)
    self_out = self_attn(qkv)

    # Check that outputs are consistent
    torch.testing.assert_close(out, self_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex", "flash-varlen"])
def test_cross_attention(attn_type):
    # Generate random input tensors
    q = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")
    kv = torch.randn(1, 256, 128, dtype=torch.float16, device="cuda")

    # Initialize attention layers
    attn = Attention(dim=128, num_heads=8, attn_type=attn_type).cuda().half()

    # Compute outputs
    out = attn(q, kv)

    # Check output shape
    assert out.shape == q.shape


@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex", "flash-varlen"])
@pytest.mark.parametrize("attn_type_new", ["torch", "flash", "flex", "flash-varlen"])
def test_attention_change_backend(attn_type, attn_type_new):
    # Generate random input tensors
    q = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda")
    kv = torch.randn(1, 256, 128, dtype=torch.float16, device="cuda")

    # Initialize attention layers
    attn = Attention(dim=128, num_heads=8, attn_type=attn_type).cuda().half()

    # Compute outputs
    out = attn(q, kv)

    # Change backend
    attn.set_backend(attn_type_new)

    assert attn.attn_type == attn_type_new, f"Expected {attn_type_new}, got {attn.attn_type}"

    out_new = attn(q, kv)

    # Check that the outputs are consistent
    torch.testing.assert_close(out, out_new, atol=1e-3, rtol=1e-3)
