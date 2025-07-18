import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.LayerNorm):
    """Slightly faster LayerNorm by seting elementwise_affine=False."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, elementwise_affine=False)

    def forward(self, x):
        dtype = x.dtype
        return super().forward(x).to(dtype)


class RMSNorm(nn.Module):
    """RNMSNorm from https://arxiv.org/abs/1910.07467."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.weight


class SimpleRMSNorm(nn.Module):
    """From X-transformers."""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale


class DyT(nn.Module):
    """2503.10622."""

    def __init__(self, dim, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
