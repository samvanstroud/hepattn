import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

data_type = torch.float16
B: int = 16
H: int = 16
S: int = 8192
D: int = 64
device = "cuda"


@pytest.fixture
def qkv():
    return [torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True) for _ in range(3)]


def flex_attention_call(qkv):
    flexattn = torch.compile(flex_attention, dynamic=False)
    return flexattn(*qkv)


@pytest.mark.gpu
def test_flex_basic(qkv):
    assert torch.allclose(F.scaled_dot_product_attention(*qkv), flex_attention_call(qkv), atol=1e-3)
