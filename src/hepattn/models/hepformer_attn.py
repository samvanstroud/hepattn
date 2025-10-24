from abc import ABC

import torch

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

from torch import BoolTensor, Tensor, nn


def merge_masks(
    q_mask: BoolTensor | None,
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
    q_shape,
    k_shape,
) -> BoolTensor:
    """Create a full attention mask which incorporates the padding information.

    Using pytorch transformer convention:
        False: Real node
        True:  Zero padded

    Parameters
    ----------
    q_mask : BoolTensor | None
        Mask for the queries, of shape (batch, q_len).
    kv_mask : BoolTensor | None
        Mask for the keys and values, of shape (batch, kv_len).
    attn_mask : BoolTensor | None
        Full attention mask, of shape (batch, q_len, kv_len).
    q_shape : Size
        Shape of the queries tensor, (batch, q_len, dim).
    k_shape : Size
        Shape of the keys tensor, (batch, kv_len, dim).
    """
    # Create the full mask which combines the attention and padding masks
    mask = None

    # if both masks exist, combine them
    if q_mask is not None and kv_mask is not None:
        mask = q_mask.unsqueeze(-1) | kv_mask.unsqueeze(-2)

    # if only one mask exists, expand it to the other dimension
    if q_mask is None and kv_mask is not None:
        mask = kv_mask.unsqueeze(-2).expand(-1, q_shape[-2], -1)
    if kv_mask is None and q_mask is not None:
        mask = q_mask.unsqueeze(-1).expand(-1, -1, k_shape[-2])

    # include the attention mask
    if attn_mask is not None:
        mask = attn_mask if mask is None else attn_mask | mask

    return mask


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def torch_meff_attn(q: Tensor, k: Tensor, v: Tensor, mask: BoolTensor, dropout: float) -> Tensor:
    # masking can lead to nans, see
    # - https://github.com/pytorch/pytorch/issues/110213
    # - https://github.com/pytorch/pytorch/issues/103749
    # TODO: change mask convention
    if mask is not None:
        mask = mask.contiguous()
        mask = ~mask
        # mask = (1.0 - mask.float()) * torch.finfo(q.dtype).min

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)


def torch_flash_attn(q: Tensor, k: Tensor, v: Tensor, mask: BoolTensor, dropout: float) -> Tensor:
    assert mask is None, "Flash attention does not support attention masks"
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)


def flash_attn_wrap(q: Tensor, k: Tensor, v: Tensor, dropout: float, window_size: tuple[int]) -> Tensor:
    return flash_attn_func(q, k, v, dropout_p=dropout, window_size=window_size)  # type: ignore


attn_types = {"torch-meff": torch_meff_attn, "torch-flash": torch_flash_attn, "flash": flash_attn_wrap}


class Attention(nn.Module, ABC):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        attn_type: str = "torch-meff",
        n_kv_heads: int | None = None,
        window_size: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert self.n_kv_heads is not None
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.bias = bias

        self.attn_type = attn_type
        self.attn_func = attn_types[self.attn_type]
        self.backend = self._flash_backend if self.attn_type == "flash" else self._torch_backend
        if window_size is None:
            self.window_size = (-1, -1)
        else:
            assert attn_type == "flash"
            assert window_size % 2 == 0
            self.window_size = (window_size // 2, window_size // 2)

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=self.bias)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=self.bias)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=self.bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=self.bias)
        self.add_zero_attn = True

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
    ) -> Tensor:
        # combine masks
        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape)

        # input projections
        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        # add a dummy token to attend to - avoids nan when all tokens are padded
        if self.add_zero_attn:
            batch = q.shape[0]
            zero_attn_shape = (batch, 1, self.dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = nn.functional.pad(attn_mask, (0, 1), value=False)
            if kv_mask is not None:
                kv_mask = nn.functional.pad(kv_mask, (0, 1), value=False)

        # run attention
        output = self.backend(q, k, v, attn_mask)

        # return output projection
        return self.wo(output)

    def _torch_backend(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: BoolTensor | None = None):
        batch, q_len, _ = q.shape
        _, kv_len, _ = k.shape
        q = q.view(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # repeat keys and values to match number of query heads
        if self.repeats > 1:
            k, v = repeat_kv(k, v, self.repeats, dim=-2)

        # reshape mask
        if attn_mask is not None:
            attn_mask = attn_mask.view(batch, 1, q_len, kv_len).expand(-1, self.n_heads, -1, -1)

        # run attention
        output = self.attn_func(q, k, v, mask=attn_mask, dropout=self.dropout)  # type: ignore

        # reshape output
        output = output.transpose(1, 2).contiguous().view(batch, -1, self.dim)

        # return output
        return output

    def _flash_backend(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: BoolTensor | None = None):
        assert attn_mask is None

        batch, q_len, _ = q.shape
        _, kv_len, _ = k.shape
        q_p = q.view(batch, q_len, self.n_heads, self.head_dim)
        k_p = k.view(batch, kv_len, self.n_kv_heads, self.head_dim)
        v_p = v.view(batch, kv_len, self.n_kv_heads, self.head_dim)

        # repeat keys and values to match number of query heads
        if self.repeats > 1:
            k_p, v_p = repeat_kv(k_p, v_p, self.repeats, dim=-2)

        # run attention
        output = self.attn_func(q_p, k_p, v_p, dropout=self.dropout, window_size=self.window_size)  # type: ignore

        # reshape output
        output = output.view_as(q)

        # return output
        return output


class SelfAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim=dim, **kwargs)
        self.norm = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.norm(x)
        return self.attention(x, x, x, **kwargs)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim=dim, **kwargs)
        self.norm_q = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.norm_kv = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, q: Tensor, kv: Tensor, **kwargs) -> Tensor:
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        return self.attention(q, kv, kv, **kwargs)


class GLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, activation: str = "SiLU", bias: bool = True, gated: bool = True):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim * 2

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.gate = None
        if gated:
            self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = getattr(nn, activation)()

    def forward(self, x) -> Tensor:
        x = self.norm(x)
        out = self.activation(self.w1(x))
        if self.gate:
            out = out * self.gate(x)
        return self.w2(out)