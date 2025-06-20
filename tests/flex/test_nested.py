# https://github.com/pytorch-labs/attention-gym/blob/bf70245667249dc1051bfd6182ab2671fd056245/examples/flex_attn.ipynb#L986
import random
from functools import lru_cache

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):  # noqa: N803
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)
torch.manual_seed(0)


random.seed(0)
torch.manual_seed(0)

batch_size = 16
n_heads = 16
D = 64


def prepare_qkv_values(tensor):
    return tensor._values.detach().requires_grad_()  # noqa: SLF001


def build_seq_idx(tensor: torch.Tensor):
    offsets = tensor.offsets()
    total_length = tensor.offsets()[-1].item()
    # Create a range tensor from 0 to total_length
    range_tensor = torch.arange(total_length, device="cuda", dtype=torch.int32)

    # Use searchsorted to find the index for each position
    seq_idx = torch.searchsorted(offsets, range_tensor, right=True) - 1

    return seq_idx


def create_njt_wrapper(orig_mask_mod, offsets, seq_idx):
    """Generic Wrapper that converts Dense mask_mod functions to NJT mask_mod functions"""

    def njt_score_mod(b, h, q_idx, kv_idx):
        q_nested = q_idx - offsets[seq_idx[q_idx]]
        kv_nested = kv_idx - offsets[seq_idx[kv_idx]]
        is_same_sequence = seq_idx[q_idx] == seq_idx[kv_idx]
        return orig_mask_mod(b, h, q_nested, kv_nested) & is_same_sequence

    return njt_score_mod


# Dense Score Mod
def causal_mask(b, h, q_idx, kv_idx):  # noqa: ARG001
    return q_idx >= kv_idx


@pytest.mark.gpu
def test_flex_nested():
    # Current limitation that the total combined sequence length must be divisible by 128
    sentence_lengths = [random.randint(1, 1024) for _ in range(batch_size - 1)]  # noqa: S311
    total = sum(sentence_lengths)
    sentence_lengths.append(128 - total % 128)
    total = sum(sentence_lengths)

    ragged_tensors = [torch.randn(n, n_heads, D, device="cuda") for n in sentence_lengths]
    query = torch.nested.nested_tensor(ragged_tensors, layout=torch.jagged, requires_grad=True)
    key = torch.nested.nested_tensor(ragged_tensors, layout=torch.jagged, requires_grad=True)
    value = torch.nested.nested_tensor(ragged_tensors, layout=torch.jagged, requires_grad=True)

    # Build the seq_idx lookup table for
    offsets = query.offsets()
    seq_idx = build_seq_idx(query)

    causal_score_mod_njt = create_njt_wrapper(causal_mask, offsets, seq_idx)

    query_values = prepare_qkv_values(query)
    key_values = prepare_qkv_values(key)
    value_values = prepare_qkv_values(value)

    block_mask = create_block_mask_cached(causal_score_mod_njt, 1, 1, total, total, device=query_values.device)
    out_flex = flex_attention(
        query_values.view(1, -1, n_heads, D).transpose(1, 2),
        key_values.view(1, -1, n_heads, D).transpose(1, 2),
        value_values.view(1, -1, n_heads, D).transpose(1, 2),
        block_mask=block_mask,
    )
    out_sdpa = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=True,
    )

    sdpa_outs = []
    flex_outs = []

    grad_out = torch.randn_like(out_sdpa)

    sdpa_outs.append(out_sdpa)
    out_sdpa.backward(grad_out)
    sdpa_outs += [query.grad, key.grad, value.grad]

    flex_outs.append(out_flex)
    out_flex.backward(grad_out._values.unsqueeze(0))  # noqa: SLF001
    flex_outs += [query_values.grad, key_values.grad, value_values.grad]

    for flex, sdpa in zip(flex_outs, sdpa_outs, strict=False):
        flex = flex.squeeze(0)
        torch.testing.assert_close(flex, sdpa._values, atol=1e-2, rtol=1e-2)  # noqa: SLF001

    print("Correctness check passed ✅")

    print(block_mask)
