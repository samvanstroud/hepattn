import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch import BoolTensor, Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, _score_mod_signature, flex_attention

from hepattn.models.norm import LayerNorm

ATTN_TYPES = {
    "torch": F.scaled_dot_product_attention,
    "flex": flex_attention,
    "flash": flash_attn_func,
}


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        attn_type: str = "torch",
        torch_compile: bool = False,
        window_size: int | None = None,
        value_residual: bool = False,
        qkv_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "num_heads must divide dim."
        assert attn_type in ATTN_TYPES, f"Invalid attention type: {attn_type}"
        assert window_size is None or attn_type == "flash", "Window size can only be specified for flash attention"

        self.dim = dim
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.attn = ATTN_TYPES[attn_type]
        self.window_size = None
        self.value_residual = value_residual
        self.qkv_norm = qkv_norm

        if attn_type == "flash":
            self.window_size = (window_size // 2, window_size // 2) if window_size is not None else (-1, -1)
        if torch_compile or attn_type == "flex":
            self.attn = torch.compile(self.attn, dynamic=True)

        self.q_proj = nn.Linear(dim, self.dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, dim, bias=bias)

        if self.value_residual:
            self.value_residual_mix = nn.Sequential(nn.Linear(dim, num_heads), nn.Sigmoid())

        if self.qkv_norm:
            self.q_norm = LayerNorm(dim)
            self.k_norm = LayerNorm(dim)
            self.v_norm = LayerNorm(dim)

    def separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        x = x.unflatten(-1, (num_heads, -1))  # B S D -> B S H Dh
        if self.attn_type != "flash":
            x = x.transpose(-3, -2)  # B S H Dh -> B H S Dh
        return x

    def recombine_heads(self, x: Tensor) -> Tensor:
        if self.attn_type != "flash":
            x = x.transpose(-3, -2)  # B H S Dh -> B S H Dh
        return x.flatten(-2)  # B S H Dh -> B S D

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        attn_mask: BlockMask | BoolTensor | None = None,
        score_mod: _score_mod_signature | None = None,
        initial_values: dict | None = None,
    ) -> Tensor:
        """
        Multi-head attention forward pass.

        Parameters
        ----------
        q : Tensor
            Queries tensor of shape (B, S, D).
        k : Tensor, optional
            Keys tensor of shape (B, S, D). If None, defaults to q.
        v : Tensor, optional
            Values tensor of shape (B, S, D). If None, defaults to q.
        attn_mask : BlockMask | BoolTensor, optional
            Attention mask to apply. If None, no mask is applied.
        score_mod : _score_mod_signature, optional
            Score modifier function for flex attention. If None, no score modifier is applied.
        initial_values : dict, optional
            Initial values for value residual connection.
        """
        # Default to self-attention
        k = k if k is not None else q
        v = v if v is not None else q

        # Mix for value residual
        mix = None
        if self.value_residual:
            mix = self.value_residual_mix(q)
            mix = mix.unsqueeze(-1)
            if self.attn_type != "flash":
                mix = mix.transpose(-2, -3)

        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Normalize queries, keys, and values
        if self.qkv_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)

        # Separate into heads
        q = self.separate_heads(q, self.num_heads)
        k = self.separate_heads(k, self.num_heads)
        v = self.separate_heads(v, self.num_heads)

        # Residual connection with initial values
        if self.value_residual:
            if not initial_values:
                initial_values["v"] = v
            else:
                v = v * mix + initial_values["v"] * (1.0 - mix)

        # Fused attention
        if self.attn_type == "flex":
            out = self.attn(q, k, v, block_mask=attn_mask, score_mod=score_mod)
        elif self.attn_type == "torch":
            # Have to expand the attention mask so that it is broadcasted over the head dimension
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(-3)
            out = self.attn(q, k, v, attn_mask=attn_mask)
        elif self.attn_type == "flash":
            out = self.attn(q, k, v, window_size=self.window_size)
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

        # Get output
        out = self.recombine_heads(out)
        return self.out_proj(out)
