from functools import partial

import torch
from torch import BoolTensor, Tensor, nn

from hepattn.models import Attention, Dense, LayerNorm
from hepattn.models.flex_local_mask import sliding_window_mask


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
        norm: nn.Module = LayerNorm,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        dim: int = 0,
    ) -> None:
        """Parameters
        ----------
        fn : nn.Module
            The module to wrap. Must be non-resizing.
        norm : str, optional
            The normalization layer.
        layer_scale : float | None, optional
            The initial value for the layer_scale. If None, then no layer_scale is applied.
        drop_path : float, optional
            The drop path rate.
        dim : int
            The dimension of the input and output.
        """
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)
        self.ls = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.dp = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x + self.dp(self.ls(self.fn(self.norm(x), *args, **kwargs)))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: nn.Module = LayerNorm,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        window_size: int | None = None,
    ) -> None:
        """Encoder layer: self-attention -> feed-forward.

        Parameters
        ----------
        dim : int
            Dimension of the embeddings.
        norm : str, optional
            The normalization layer.
        drop_path : float, optional
            Drop path rate.
        layer_scale : float | None, optional
            Initial layer_scale value.
        dense_kwargs : dict | None, optional
            Keyword arguments for dense layer.
        attn_kwargs : dict | None, optional
            Keyword arguments for self-attention layer.
        window_size : int | None, optional
            The window size for the sliding window.
        """
        super().__init__()

        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}

        self.dim = dim
        residual = partial(Residual, dim=dim, norm=norm, layer_scale=layer_scale, drop_path=drop_path)
        self.attn = residual(Attention(self.dim, **attn_kwargs))
        self.dense = residual(Dense(self.dim, **dense_kwargs))

        self.mask_mod = None
        if window_size is not None:
            self.mask_mod = sliding_window_mask(10)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.dense(self.attn(x, x, x, mask_mod=self.mask_mod, **kwargs))


class Encoder(nn.Module):
    def __init__(self, num_layers: int, dim: int, **kwargs) -> None:
        """Transformer encoder.

        Parameters
        ----------
        num_layers : int
            Number of layers.
        dim : int
            Dimension of the embeddings at each layer.
        kwargs : dict
            Keyword arguments for EncoderLayer.
        """
        super().__init__()

        self.num_layers = num_layers
        self.dim = dim

        self.layers = torch.nn.ModuleList([EncoderLayer(dim=dim, **kwargs) for _ in range(num_layers)])

    def forward(self, x: Tensor, mask: BoolTensor | None = None, **kwargs) -> Tensor:
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)

        for layer in self.layers:
            x = layer(x, **kwargs)

        return x, mask
