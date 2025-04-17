import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

data_type = torch.float16
flex_attention = torch.compile(flex_attention, dynamic=False)
B: int = 16
H: int = 16
S: int = 8192
D: int = 64
device = "cuda"
qkv = [torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True) for _ in range(3)]


def causal_fa2():
    return F.scaled_dot_product_attention(*qkv, is_causal=False)


def sdpa_mask():
    return F.scaled_dot_product_attention(*qkv)


def flex_attention_call():
    return flex_attention(*qkv)


assert torch.allclose(causal_fa2(), flex_attention_call(), atol=1e-3)
assert torch.allclose(sdpa_mask(), flex_attention_call(), atol=1e-3)
