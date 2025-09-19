import math

import torch
from torch import nn


class QuerySource(nn.Module):
    """Base interface: produce queries from the current features dict `x`."""

    def __init__(self, num_queries: int, dim: int):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim

    def forward(self, x: dict) -> dict:
        raise NotImplementedError


class FixedQuerySource(QuerySource):
    def __init__(self, num_queries: int, dim: int, init: str = "", hit_particle_ratio: float = 0):
        super().__init__(num_queries, dim)
        if init == "randn":
            self.register_buffer("bank", torch.randn(num_queries, dim))
        elif init == "zero":
            self.register_buffer("bank", torch.zeros(num_queries, dim))
        else:
            self.bank = nn.Parameter(torch.randn(num_queries, dim))

    def forward(self, x: dict) -> dict:
        batch_size = x["key_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]

        if self.hit_particle_ratio:
            self.num_queries = math.ceil(num_constituents / self.hit_particle_ratio)

        # Bank assumed shape: (N_bank, d); ordered by phi ascending
        n_bank = self.bank.shape[-2]
        k = min(self.num_queries, n_bank)
        device = self.bank.device

        # === Even-in-phi index selection (bin centers) ===
        # centers = (i + 0.5) * (N_bank / k), i = 0..k-1
        step = n_bank / float(k)
        centers = (torch.arange(k, device=device) + 0.5) * step
        idx = centers.floor().long().clamp_(0, n_bank - 1)

        # Guard: if rounding ever yields duplicates (can happen for tiny N/k),
        # fill remaining slots with near-uniform starts, then dedup & trim.
        if torch.unique_consecutive(idx).numel() < k:
            idx = torch.unique_consecutive(idx)
            missing = k - idx.numel()
            if missing > 0:
                extras = (torch.arange(missing, device=device) * step).floor().long().clamp_(0, n_bank - 1)
                idx = torch.unique(torch.cat([idx, extras], dim=0))[:k]
            # Ensure sorted for stable selection
            idx, _ = torch.sort(idx)

        # Select those queries from the bank
        # If you want to preserve your previous behavior controlled by self.query_init,
        # put this selection behind that flag. Otherwise, always use even-phi sampling.
        bank_sel = self.bank.index_select(0, idx)  # (k, d)

        # Expand to batch
        q = bank_sel.unsqueeze(0).expand(batch_size, -1, -1)

        # Valid mask for the chosen queries
        q_valid = torch.ones((batch_size, k), dtype=torch.bool, device=x["key_embed"].device)

        return {"query_embed": q, "query_valid": q_valid}


class ModulatedQuerySource(QuerySource):
    def __init__(self, num_queries: int, dim: int, ratio: int = 4, hidden: int = 512, init: str = ""):
        super().__init__(num_queries, dim)
        self.ratio = ratio  # r
        n_basic = num_queries * ratio  # n = m*r

        if init == "randn":
            self.register_buffer("basic_bank", torch.randn(n_basic, dim))
        elif init == "zero":
            self.register_buffer("basic_bank", torch.zeros(n_basic, dim))
        else:
            self.basic_bank = nn.Parameter(torch.randn(n_basic, dim))

        # A(F) -> W^D logits (Eq. 5 in DQ-Det), shape (B, m*r) -> softmax per group r
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, num_queries * ratio))

    def forward(self, x: dict) -> dict:
        B, m, r, D = x["key_embed"].shape[0], self.num_queries, self.ratio, self.dim

        # 1) global feature A(F): simplest = mean over keys (use your own pooling if you prefer)
        g = x["key_embed"].mean(dim=1)  # (B, D)

        # 2) per-image weights
        wd = self.mlp(g).view(B, m, r).softmax(dim=-1)  # (B, m, r)

        # 3) group the basic queries and mix
        qb = (self.basic_bank if hasattr(self, "basic_bank") else self.basic_bank).view(m, r, D).unsqueeze(0).expand(B, -1, -1, -1)  # (B, m, r, D)
        qm = torch.einsum("bmr,bmrd->bmd", wd, qb)  # (B, m, D)

        return {
            "query_embed": qm,
            "dq_weights": wd,  # optional: for logging
            "basic_queries": qb.view(B, m * r, D),  # optional: for β-branch
        }
