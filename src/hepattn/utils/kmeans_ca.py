import torch
from torch import Tensor, nn


class KMeansCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        update: str = "mean",
        value_proj: bool = False,
        mask_attn: bool = False,
        respect_attn_mask: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert update in {"sum", "mean"}
        self.dim = dim
        self.update = update
        self.respect_attn_mask = mask_attn or respect_attn_mask
        self.eps = eps
        self.v_proj = nn.Linear(dim, dim, bias=False) if value_proj else None

    def forward(
        self,
        q: Tensor,  # (B, N, D)
        k: Tensor | None = None,  # (B, M, D)
        v: Tensor | None = None,  # (B, M, D)
        attn_mask: Tensor | None = None,  # (B, N, M) bool
        q_mask: Tensor | None = None,  # (B, N) bool
        kv_mask: Tensor | None = None,  # (B, M) bool
    ) -> Tensor:
        if v is None:
            raise ValueError("KMeansCrossAttention requires v (values).")

        neg_inf = float("-inf")
        if k is None:
            raise ValueError("KMeansCrossAttention requires k (keys).")
        logits = q @ k.transpose(-2, -1)

        if q_mask is not None:
            logits = logits.masked_fill(~q_mask.unsqueeze(-1), neg_inf)
        if self.respect_attn_mask and attn_mask is not None:
            logits = logits.masked_fill(~attn_mask, neg_inf)
        if kv_mask is not None:
            logits = logits.masked_fill(~kv_mask.unsqueeze(-2), neg_inf)

        # One pass gives both argmax indices and max values
        max_val, idx = logits.max(dim=-2)  # max over N -> (B, M), (B, M)
        valid = torch.isfinite(max_val)  # (B, M) tokens with any allowed query

        vv = v if self.v_proj is None else self.v_proj(v)  # (B, M, D)
        vv = vv * valid.unsqueeze(-1).to(vv.dtype)

        B, M, D = vv.shape
        N = logits.shape[-2]

        out = vv.new_zeros((B, N, D))
        out.scatter_add_(1, idx.unsqueeze(-1).expand(B, M, D), vv)

        if self.update == "mean":
            counts = vv.new_zeros((B, N))
            counts.scatter_add_(1, idx, valid.to(vv.dtype))
            out = out / (counts.clamp_min(1.0).unsqueeze(-1) + self.eps)

        return out
