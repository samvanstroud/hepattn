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
        logits: Tensor | dict | None = None,  # (B, N, M) or sparse dict
        **kwargs,
    ) -> Tensor:
        if v is None:
            raise ValueError("KMeansCrossAttention requires v (values).")

        neg_inf = float("-inf")
        sparse_logits = isinstance(logits, dict)

        if logits is None:
            if k is None:
                raise ValueError("Provide either logits or k.")
            logits = q @ k.transpose(-2, -1)

        if not sparse_logits:
            if q_mask is not None:
                logits = logits.masked_fill(~q_mask.unsqueeze(-1), neg_inf)
            if self.respect_attn_mask and attn_mask is not None:
                logits = logits.masked_fill(~attn_mask, neg_inf)
            if kv_mask is not None:
                logits = logits.masked_fill(~kv_mask.unsqueeze(-2), neg_inf)

            # One pass gives both argmax indices and max values
            max_val, idx = logits.max(dim=-2)  # max over N -> (B, M), (B, M)
            valid = torch.isfinite(max_val)  # (B, M) tokens with any allowed query
        else:
            values = logits["values"]
            indices = logits["indices"]
            kv_len = v.shape[1]
            if "kv_len" in logits:
                provided = logits["kv_len"]
                if isinstance(provided, torch.Tensor):
                    provided = int(provided.item())
                if provided != kv_len:
                    raise ValueError(f"Sparse logits kv_len ({provided}) does not match v length ({kv_len}).")

            if indices.dim() == 2:
                indices = indices.unsqueeze(0).expand(values.shape[0], -1, -1)

            if q_mask is not None:
                values = values.masked_fill(~q_mask.unsqueeze(-1), neg_inf)
            if self.respect_attn_mask and attn_mask is not None:
                values = values.masked_fill(~attn_mask.gather(2, indices), neg_inf)
            if kv_mask is not None:
                kv_mask_values = kv_mask.gather(1, indices.reshape(indices.shape[0], -1)).reshape_as(values)
                values = values.masked_fill(~kv_mask_values, neg_inf)

            bsz, num_queries, window = values.shape
            flat_keys = indices.reshape(bsz, -1)
            flat_logits = values.reshape(bsz, -1)

            max_val = flat_logits.new_full((bsz, kv_len), neg_inf)
            max_val.scatter_reduce_(1, flat_keys, flat_logits, reduce="amax", include_self=True)
            valid = torch.isfinite(max_val)

            q_idx = torch.arange(num_queries, device=values.device).unsqueeze(1).expand(num_queries, window).reshape(1, -1)
            q_idx = q_idx.expand(bsz, -1)
            is_max = flat_logits == max_val.gather(1, flat_keys)
            q_idx = torch.where(is_max, q_idx, torch.full_like(q_idx, num_queries))

            idx = flat_keys.new_full((bsz, kv_len), num_queries)
            idx.scatter_reduce_(1, flat_keys, q_idx, reduce="amin", include_self=True)
            idx = torch.where(valid, idx, torch.zeros_like(idx))

        vv = v if self.v_proj is None else self.v_proj(v)  # (B, M, D)
        vv = vv * valid.unsqueeze(-1).to(vv.dtype)

        B, M, D = vv.shape
        N = num_queries if sparse_logits else logits.shape[-2]

        out = vv.new_zeros((B, N, D))
        out.scatter_add_(1, idx.unsqueeze(-1).expand(B, M, D), vv)

        if self.update == "mean":
            counts = vv.new_zeros((B, N))
            counts.scatter_add_(1, idx, valid.to(vv.dtype))
            out = out / (counts.clamp_min(1.0).unsqueeze(-1) + self.eps)

        return out
