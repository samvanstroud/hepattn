"""
Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former
"""

from functools import partial

import torch
from torch import Tensor, nn

from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.norm import LayerNorm
from hepattn.models.transformer import Residual


class MaskFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        mask_attention: bool = True,
        bidirectional_ca: bool = True,
    ) -> None:
        super().__init__()

        self.mask_attention = mask_attention
        self.bidirectional_ca = bidirectional_ca

        attn_kwargs = attn_kwargs or {}
        dense_kwargs = dense_kwargs or {}

        residual = partial(Residual, dim=dim, norm=LayerNorm)
        self.q_ca = residual(Attention(dim, **attn_kwargs))
        self.q_sa = residual(Attention(dim, **attn_kwargs))
        self.q_dense = residual(Dense(dim, **dense_kwargs))
        if self.bidirectional_ca:
            self.kv_ca = residual(Attention(dim, **attn_kwargs))
            self.kv_dense = residual(Dense(dim, **dense_kwargs))

    def forward(self, q: Tensor, kv: Tensor, attn_mask: Tensor | None = None, kv_mask: Tensor | None = None) -> Tensor:
        assert kv_mask is None, "KV mask is not yet supported"

        # q are object queries, kv are hit embeddings
        # if we want to do mask attention
        if self.mask_attention:
            assert attn_mask is not None, "attn mask must be provided for mask attention"
            attn_mask = attn_mask.detach()
            # If a BoolTensor is provided, positions with `True` are not allowed to attend while `False` values will be unchanged.
            # if the attn mask is completely invalid for a given query, allow it to attend everywhere
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        else:
            attn_mask = None

        # update queries
        q = self.q_ca(q, k=kv, v=kv, attn_mask=attn_mask)  # needs input pad mask
        q = self.q_sa(q)
        q = self.q_dense(q)

        # update inputs
        if self.bidirectional_ca:
            if attn_mask is not None:
                # Index from the back so we are batch shape agnostic
                attn_mask = attn_mask.transpose(-2, -1)
            kv = self.kv_ca(kv, k=q, v=q, attn_mask=attn_mask)  # needs input pad mask
            kv = self.kv_dense(kv)

        return q, kv
