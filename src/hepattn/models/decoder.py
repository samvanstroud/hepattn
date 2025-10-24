from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import ModuleList

from hepattn.models.dense import Dense
from hepattn.models.hepformer_loss import HEPFormerLoss
from hepattn.models.hepformer_attn import GLU, CrossAttention, SelfAttention


class MaskDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        md_config: Mapping,
        mask_net: nn.Module,
        num_objects: int,
        class_net: nn.Module | None = None,
        aux_loss: bool = False,
        version: str = "v2",
        bidir: bool | None = None,  # backwards compatability
        V2: bool | None = None,  # backwards compatability
    ):
        super().__init__()
        self.aux_loss = aux_loss

        self.inital_q = nn.Parameter(torch.empty((num_objects, embed_dim)))
        nn.init.normal_(self.inital_q)
        decoder_layer = MaskDecoderLayer

        self.layers = nn.ModuleList([decoder_layer(embed_dim, **md_config) for _ in range(num_layers)])

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.class_net = class_net
        self.mask_net = mask_net

    def get_preds(self, queries: Tensor, mask_tokens: Tensor):
        # get mask predictions from queries and mask tokens
        pred_masks = get_masks(mask_tokens, queries, self.mask_net)  # [..., mask.squeeze(0)] when padding is enabled
        # get class predictions from queries

        if self.class_net is None:
            return {"masks": pred_masks}

        class_logits = self.class_net(queries)
        if class_logits.shape[-1] == 1:
            class_probs = class_logits.sigmoid()
            class_probs = torch.cat([1 - class_probs, class_probs], dim=-1)
        else:
            class_probs = class_logits.softmax(-1)

        return {"class_logits": class_logits, "class_probs": class_probs, "masks": pred_masks}

    def forward(self, x: Tensor, mask: Tensor = None):
        # apply norm
        q = self.norm1(self.inital_q.expand(x.shape[0], -1, -1))
        x = self.norm2(x)

        intermediate_outputs: list | None = [] if self.aux_loss else None
        for layer in self.layers:
            if self.aux_loss:
                assert intermediate_outputs is not None
                intermediate_outputs.append({"queries": q, **self.get_preds(q, x)})
            q, x = layer(q, x, mask_net=self.mask_net)

        preds = {"queries": q, "x": x, **self.get_preds(q, x)}
        if self.aux_loss:
            preds["intermediate_outputs"] = intermediate_outputs
        return preds


def get_masks(x: Tensor, q: Tensor, mask_net: nn.Module):
    mask_tokens = mask_net(q)
    pred_masks = torch.einsum("bqe,ble->bql", mask_tokens, x)
    return pred_masks


class MaskDecoderLayer(nn.Module):
    """Using the updated V2 transformer implementation"""

    def __init__(self, dim: int, n_heads: int, mask_attention: bool, bidirectional_ca: bool, topk: int | None = None, local_ca: int | None = None) -> None:
        super().__init__()
        if topk and not mask_attention:
            raise ValueError("Topk only works with mask attention")
        if mask_attention and local_ca:
            raise ValueError("Cannot use both mask attention and local ca")

        self.mask_attention = mask_attention
        self.bidirectional_ca = bidirectional_ca
        self.topk = topk
        self.local_ca = local_ca

        self.q_ca = CrossAttention(dim=dim, n_heads=n_heads)
        self.q_sa = SelfAttention(dim=dim, n_heads=n_heads)
        self.q_dense = GLU(dim)
        if bidirectional_ca:
            self.kv_ca = CrossAttention(dim=dim, n_heads=n_heads)
            self.kv_dense = GLU(dim)

    def forward(self, q: Tensor, kv: Tensor, mask_net: nn.Module, kv_mask: Tensor | None = None) -> Tensor:
        attn_mask = None

        # if we want to do mask attention
        if self.mask_attention:
            scores = get_masks(kv, q, mask_net).detach()

            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = scores.sigmoid() < 0.1

            # if the attn mask is completely invalid for a given query, allow it to attend everywhere
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        # update queries with cross attention from nodes
        q = q + self.q_ca(q, kv, kv_mask=kv_mask, attn_mask=attn_mask)

        # update queries with self attention
        q = q + self.q_sa(q)

        # dense update
        q = q + self.q_dense(q)

        # update nodes with cross attention from queries and dense layer
        # TODO: test also with self attention
        if self.bidirectional_ca:
            if attn_mask is not None:
                attn_mask = attn_mask.transpose(1, 2)
            kv = kv + self.kv_ca(kv, q, q_mask=kv_mask, attn_mask=attn_mask)
            kv = kv + self.kv_dense(kv)

        return q, kv