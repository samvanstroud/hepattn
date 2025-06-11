from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from hepattn.flex import relative_position, relative_position_wrapped
from hepattn.flex.sliding_window import sliding_window_mask, sliding_window_mask_wrapped
from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.norm import LayerNorm

create_block_mask = torch.compile(create_block_mask, dynamic=True)

SCORE_MODS = {
    "relative_position": relative_position,
    "relative_position_wrapped": relative_position_wrapped,
}


class DropPath(nn.Module):
    """Randomly drop layers: https://arxiv.org/abs/1603.09382"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(mask_shape, dtype=x.dtype, device=x.device)
        return x * mask.floor().div(keep_prob)


class LayerScale(nn.Module):
    """Learn residual strength: https://arxiv.org/abs/2103.17239"""

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Residual(nn.Module):
    """Neatly wrap x = x + drop(scale * fn(norm(x)))"""

    def __init__(
        self,
        fn: nn.Module,
        dim: int,
        norm: str | None,
        post_norm: bool = False,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
    ) -> None:
        """Parameters
        ----------
        dim : int
            The dimension of the input and output.
        fn : nn.Module
            The module to wrap. Must be non-resizing.
        norm : str, optional
            The normalization layer.
        post_norm : bool, optional
            Instead of standard pre-norm, apply norm before the residual (post-norm for the previous op).
        layer_scale : float | None, optional
            The initial value for the layer_scale. If None, then no layer_scale is applied.
        drop_path : float, optional
            The drop path rate.
        """
        super().__init__()
        self.fn = fn
        self.ls = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.dp = DropPath(drop_path) if drop_path else nn.Identity()
        self.post_norm = post_norm

        if isinstance(norm, str):
            self.norm = getattr(nn, norm)(dim, elementwise_affine=False)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            # TODO: Find whatever is passing type args instead of str
            self.norm = LayerNorm(dim)
            print(f"Got unrecognised norm layer {norm}, defaulting to {self.norm}")

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.post_norm:
            x = self.norm(x)
            return x + self.dp(self.ls(self.fn(x, **kwargs)))
        return x + self.dp(self.ls(self.fn(self.norm(x), **kwargs)))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 0,
        norm: str | None = None,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        value_residual: bool = False,
        hybrid_norm: bool = False,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
    ) -> None:
        """Encoder layer: self-attention -> feed-forward.

        Parameters
        ----------
        dim : int
            Dimension of the embeddings.
        depth : int
            The depth of the layer.
        norm : str, optional
            The normalization layer.
        drop_path : float, optional
            Drop path rate.
        layer_scale : float | None, optional
            Initial layer_scale value.
        value_residual : bool, optional
            Whether to apply a residual connection to initial values.
        hybrid_norm : bool, optional
            Whether to use HybridNorm from 2503.04598.
        dense_kwargs : dict | None, optional
            Keyword arguments for dense layer.
        attn_kwargs : dict | None, optional
            Keyword arguments for self-attention layer.
        """
        super().__init__()

        attn_kwargs = attn_kwargs or {}
        dense_kwargs = dense_kwargs or {}
        norm = norm or "LayerNorm"

        # handle hybridnorm
        qkv_norm = hybrid_norm
        if depth == 0:
            hybrid_norm = False
        attn_norm = norm if not hybrid_norm else None
        dense_post_norm = not hybrid_norm

        self.dim = dim
        self.value_residual = value_residual
        residual = partial(Residual, dim=dim, layer_scale=layer_scale, drop_path=drop_path)
        self.attn = residual(Attention(self.dim, qkv_norm=qkv_norm, **attn_kwargs), norm=attn_norm)
        self.dense = residual(Dense(self.dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.dense(self.attn(x, **kwargs))


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        attn_type: str = "torch",
        window_size: int | None = None,
        window_wrap: bool = False,
        score_mod: str | None = None,
        value_residual: bool = False,
        **layer_kwargs,
    ) -> None:
        """Transformer encoder.

        Parameters
        ----------
        num_layers : int
            Number of layers.
        dim : int
            Dimension of the embeddings at each layer.
        window_size : int | None, optional
            The window size for the sliding window.
        value_residual : bool, optional
            Whether to use value residual.
        kwargs : dict
            Keyword arguments for EncoderLayer.
        """
        super().__init__()

        assert not window_wrap or window_size, "Window size must be set if window wrap is True."
        assert attn_type != "flex" or score_mod is None, "Score mod is only supported with flex attention."
        layer_kwargs = layer_kwargs or {}

        self.num_layers = num_layers
        self.dim = dim
        self.attn_type = attn_type
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.score_mod = SCORE_MODS[score_mod] if score_mod else None
        self.value_residual = value_residual

        # handle masking
        self.mask_mod = None
        self.q_len = None

        # handle attention
        attn_kwargs = layer_kwargs.get("attn_kwargs", None) or {}
        attn_kwargs["attn_type"] = attn_type
        if value_residual:
            attn_kwargs["value_residual"] = True
        attn_kwargs["window_size"] = window_size
        layer_kwargs["attn_kwargs"] = attn_kwargs

        self.layers = torch.nn.ModuleList([EncoderLayer(dim=dim, depth=i, **layer_kwargs) for i in range(num_layers)])

    def forward(self, x: Tensor, x_sort_value: Tensor | None = None, **kwargs) -> Tensor:
        # If value to sort on is provided, use it to sort the tokens
        # We don't need to use the stable sort assuming that the sort values are unique
        if x_sort_value is not None:
            x_sort_idx = torch.argsort(x_sort_value, axis=-1)
            x = torch.gather(x, -2, x_sort_idx.unsqueeze(-1).expand_as(x))

        # Initialise sliding window mask
        if self.mask_mod is None and self.attn_type != "flash" and self.window_size:
            self.q_len = torch.tensor([1], device=x.device)
            self.mask_mod = (
                sliding_window_mask(self.window_size) if not self.window_wrap else sliding_window_mask_wrapped(self.window_size, self.q_len)
            )

        # Handle masking
        attn_mask = None
        q_len = x.shape[-2]
        if self.attn_type == "torch" and self.mask_mod:
            attn_mask = create_mask(self.mask_mod, 1, 1, q_len, q_len, device=x.device)
        elif self.attn_type == "flex" and self.mask_mod:
            self.q_len[0] = q_len
            attn_mask = create_block_mask(self.mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len, device=x.device)

        # Add wrapping for flash attention with sliding window
        if self.attn_type == "flash" and self.window_wrap:
            x = torch.cat([x[:, -self.window_size // 2 :], x, x[:, : self.window_size // 2]], dim=1)

        # Apply layers
        initial_values = {} if self.value_residual else None
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, score_mod=self.score_mod, initial_values=initial_values, **kwargs)

        # Remove wrapping for flash attention with sliding window
        if self.attn_type == "flash" and self.window_wrap:
            x = x[:, self.window_size // 2 : -self.window_size // 2]

        # If we sorted the tokens, undo the sorting
        if x_sort_value is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, axis=-1)
            x = torch.gather(x, -2, x_unsort_idx.unsqueeze(-1).expand_as(x))

        return x
