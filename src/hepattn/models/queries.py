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


# class FixedQuerySource(QuerySource):
#     def __init__(self, num_queries: int, dim: int, init: str = "", hit_particle_ratio: float = 0, phi_shift=0):
#         super().__init__(num_queries, dim)
#         if init == "randn":
#             self.register_buffer("bank", torch.randn(num_queries, dim))
#         elif init == "zero":
#             self.register_buffer("bank", torch.zeros(num_queries, dim))
#         else:
#             self.bank = nn.Parameter(torch.randn(num_queries, dim))
#         self.hit_particle_ratio = hit_particle_ratio
#         self.num_queries = num_queries
#         self.phi_shift = phi_shift

#     def forward(self, x: dict) -> dict:
#         batch_size = x["key_embed"].shape[0]
#         num_constituents = x["key_embed"].shape[-2]
#         device = self.bank.device

#         phi_vals = 2 * torch.pi * (torch.arange(self.num_queries, device=device) / self.num_queries - self.phi_shift)

#         if self.hit_particle_ratio:
#             num_queries = math.ceil(num_constituents / self.hit_particle_ratio)

#             # Bank assumed shape: (self.num_queries, d); ordered by phi ascending
#             k = min(num_queries, self.num_queries)

#             # === Even-in-phi index selection (bin centers) ===
#             # centers = (i + 0.5) * (self.num_queries / k), i = 0..k-1
#             step = self.num_queries / float(k)
#             centers = (torch.arange(k, device=device) + 0.5) * step
#             idx = centers.floor().long().clamp_(0, self.num_queries - 1)

#             # Guard: if rounding ever yields duplicates (can happen for tiny N/k),
#             # fill remaining slots with near-uniform starts, then dedup & trim.
#             if torch.unique_consecutive(idx).numel() < k:
#                 idx = torch.unique_consecutive(idx)
#                 missing = k - idx.numel()
#                 if missing > 0:
#                     extras = (torch.arange(missing, device=device) * step).floor().long().clamp_(0, self.num_queries - 1)
#                     idx = torch.unique(torch.cat([idx, extras], dim=0))[:k]
#                 # Ensure sorted for stable selection
#                 idx, _ = torch.sort(idx)

#             # Select those queries from the bank
#             # If you want to preserve your previous behavior controlled by self.query_init,
#             # put this selection behind that flag. Otherwise, always use even-phi sampling.
#             bank_sel = self.bank.index_select(0, idx)  # (k, d)
#             phi_vals = phi_vals.index_select(0, idx)

#             # Expand to batch
#             q = bank_sel.unsqueeze(0).expand(batch_size, -1, -1)

#         else:
#             q = self.bank.unsqueeze(0).expand(batch_size, -1, -1)
#             k = self.num_queries

#         # Valid mask for the chosen queries
#         q_valid = torch.full((batch_size, k), True, device=device)

#         return {"query_embed": q, "query_valid": q_valid, "query_phi": phi_vals}


class FixedQuerySource(QuerySource):
    """
    Produces k dynamic queries from M base queries by:
      - method="pick": evenly-spaced picks of base queries (original behavior)
      - method="window": circular, localized weighted averages (evenly spaced centers)
      - method="window_adaptive": same as 'window' but centers are equal-mass quantiles
        derived from event-specific hits' phi positions (x['hits_phi']).

    For window modes, choose window='hann' (compact support) or 'gaussian' (soft tails).
    Bandwidth in index units is ~ bw_scale * (M / k).
    """

    def __init__(
        self,
        num_queries: int,
        dim: int,
        init: str = "",
        hit_particle_ratio: float = 0.0,
        phi_shift: float = 0.0,
        method: str = "pick",  # "pick", "window", "window_adaptive"
        window: str = "hann",  # used in window modes
        bw_scale: float = 0.6,  # window bandwidth scale (index units)
        eps: float = 1e-8,
    ):
        super().__init__(num_queries, dim)
        if init == "randn":
            self.register_buffer("bank", torch.randn(num_queries, dim))
        elif init == "zero":
            self.register_buffer("bank", torch.zeros(num_queries, dim))
        else:
            self.bank = nn.Parameter(torch.randn(num_queries, dim))

        self.hit_particle_ratio = float(hit_particle_ratio)
        self.num_queries = int(num_queries)  # M
        self.phi_shift = float(phi_shift)
        self.method = method.lower()
        self.window = window.lower()
        self.bw_scale = float(bw_scale)
        self.eps = float(eps)

    def get_base_queries(self, x: dict, subset_stride: int = 1) -> dict:
        """
        Build a training-only branch input that uses the raw base bank as queries.

        Args:
            x: features dict expected to include key "key_embed" with shape (B, N, d_key)
            subset_stride: take every `subset_stride`-th base query (>=1). Use 1 to take all M.

        Returns:
            dict with keys:
              - "query_embed": (B, M_sel, D)
              - "query_valid": (B, M_sel) all True
              - "query_phi": (B, M_sel) φ positions for selected bases with phi_shift applied
        """
        B = x["key_embed"].shape[0]
        device = self.bank.device
        M = self.num_queries

        if subset_stride is None or subset_stride < 1:
            subset_stride = 1

        if subset_stride == 1:
            idx = torch.arange(M, device=device)
        else:
            idx = torch.arange(0, M, subset_stride, device=device)

        bank_sel = self.bank.index_select(0, idx)  # (M_sel, D)
        q_base = bank_sel.unsqueeze(0).expand(B, -1, -1)  # (B, M_sel, D)
        q_valid = torch.full((B, idx.numel()), True, device=device)

        base_phi = 2 * torch.pi * (idx.float() / float(M) - self.phi_shift)  # (M_sel,)
        base_phi = base_phi.unsqueeze(0).expand(B, -1)  # (B, M_sel)

        return {"query_embed": q_base, "query_valid": q_valid, "query_phi": base_phi}

    # ---------------- window helpers ----------------
    @staticmethod
    def _circular_index_distance(centers, indices, M: int):
        """
        centers: (k,) float indices in [0, M)
        indices: (M,) float indices 0..M-1
        returns: (k, M) minimal wrap-around distance in index units
        """
        diff = (centers[:, None] - indices[None, :]).abs()
        return torch.minimum(diff, M - diff)

    def _window_weights(self, dists, s):
        """dists: (k, M) circular distances (index units).
        s: bandwidth radius (index units).
        returns: (k, M) nonnegative weights."""
        s = max(float(s), 1e-6)
        if self.window == "hann":
            w = torch.clamp(1.0 + torch.cos(math.pi * dists / s), min=0.0)
            w = 0.5 * w
            w = torch.where(dists <= s, w, torch.zeros_like(w))
        elif self.window == "gaussian":
            sigma = s / math.sqrt(2.0 * math.log(2.0))  # FWHM ≈ s
            w = torch.exp(-0.5 * (dists / sigma) ** 2)
        else:
            raise ValueError(f"Unsupported window type: {self.window}")
        return w

    # ---------------- equal-mass centers from hits φ ----------------
    @staticmethod
    def _equal_mass_centers_from_hits(hits_phi: torch.Tensor, k: int, device=None):
        """
        hits_phi: (H,) tensor of hit angles in radians. Can be any range; will be wrapped to [0, 2π).
        k: number of centers to produce
        Returns: (k,) centers in angle (radians) on [0, 2π)
        Strategy:
          - wrap to [0, 2π), sort
          - cut at largest gap to linearize
          - take quantiles at (j+0.5)/k
        """
        if hits_phi is None or hits_phi.numel() == 0 or k <= 0:
            return None  # caller should fall back

        two_pi = 2.0 * math.pi
        hp = torch.remainder(hits_phi.flatten(), two_pi)  # [0, 2π)
        H = hp.numel()
        if H == 0:
            return None

        hp, _ = torch.sort(hp)  # ascending
        # gaps including wrap-around
        diffs = torch.diff(hp, prepend=hp[:1])
        diffs[0] = (hp[0] + two_pi) - hp[-1]  # wrap gap
        # find largest gap, cut after that point
        g = torch.argmax(diffs)  # index of start of largest gap
        # rotate so segment starts after the largest gap
        hp_lin = torch.roll(hp, shifts=-int(g) - 1)
        # now hp_lin spans a contiguous arc of length 2π without big wrap
        # build empirical CDF (uniform weights)
        # quantile targets
        t = (torch.arange(k, device=hp_lin.device, dtype=hp_lin.dtype) + 0.5) / float(k)  # (k,)
        # CDF at samples: u_i = (i+1)/H ; inverse via searchsorted + linear interp
        # positions of knots in CDF:
        u = (torch.arange(H, device=hp_lin.device, dtype=hp_lin.dtype) + 1.0) / float(H)  # (H,)

        # for each t_j, find left index i where u[i-1] < t <= u[i]
        idx = torch.searchsorted(u, t, right=False).clamp_(min=0, max=H - 1)  # (k,)
        idx0 = (idx - 1).clamp_(0, H - 1)
        u0 = torch.where(idx > 0, u[idx0], torch.zeros_like(u[idx]))
        u1 = u[idx]
        phi0 = hp_lin[idx0]
        phi1 = hp_lin[idx]
        # avoid divide-by-zero if duplicates
        alpha = torch.where((u1 - u0) > 0, (t - u0) / (u1 - u0), torch.zeros_like(t))
        centers_lin = phi0 + alpha * (phi1 - phi0)  # (k,)
        # map back to original circle (undo the rotation)
        centers = torch.remainder(centers_lin + hp[(g + 1) % H], two_pi)
        return centers

    def forward(self, x: dict) -> dict:
        """
        Expects:
          x["key_embed"]: (B, N, d_key)
          (optional) x["hits_phi"]: (B, H) or (H,) hit angles in radians for adaptive centers
        """
        B = x["key_embed"].shape[0]
        N_const = x["key_embed"].shape[-2]
        device = self.bank.device
        M = self.num_queries

        # Decide k (number of dynamic queries this event)
        if self.hit_particle_ratio:
            k = min(math.ceil(N_const / self.hit_particle_ratio), M)
        else:
            k = M

        # Base φ for indexing/reporting
        base_phi = 2 * torch.pi * (torch.arange(M, device=device) / M - self.phi_shift)

        # Common bandwidth in index units
        step = M / float(max(k, 1))
        s = max(1e-6, min(self.bw_scale * (M / max(k, 1)), M / 2.0))

        # Prepare outputs
        q_out = []
        phi_out = []

        if self.method == "pick":
            # Evenly spaced picks, same for all batch items
            centers_idx = (torch.arange(k, device=device, dtype=torch.float32) + 0.5) * step
            idx = centers_idx.floor().long().clamp_(0, M - 1)
            bank_sel = self.bank.index_select(0, idx)  # (k, d)
            q = bank_sel.unsqueeze(0).expand(B, -1, -1)
            phi_vals = base_phi.index_select(0, idx)
            q_valid = torch.full((B, k), True, device=device)
            return {"query_embed": q, "query_valid": q_valid, "query_phi": phi_vals}

        elif self.method in ("window", "window_adaptive"):
            indices = torch.arange(M, device=device, dtype=torch.float32)
            two_pi = 2.0 * math.pi

            # Get per-batch centers (adaptive uses hits; non-adaptive uses even spacing)
            for b in range(B):
                if self.method == "window_adaptive":
                    hits_phi = x.get("hits_phi", None)
                    if hits_phi is not None:
                        # support both (B,H) and (H,) inputs
                        hp_b = hits_phi[b] if hits_phi.dim() == 2 else hits_phi
                        centers_phi = self._equal_mass_centers_from_hits(hp_b.to(device), k, device=device)
                    else:
                        centers_phi = None
                else:
                    centers_phi = None

                if centers_phi is None:
                    # fallback: evenly spaced centers (in index units -> phi)
                    centers_idx = (torch.arange(k, device=device, dtype=torch.float32) + 0.5) * step
                    centers_phi = two_pi * (centers_idx / M)  # no phi_shift here; shift applied only for reporting if needed
                # Map φ centers -> fractional index centers in [0,M)
                centers_idx = (centers_phi / two_pi) * M  # float indices
                centers_idx = torch.remainder(centers_idx, M)

                # Build weights and queries for this batch item
                dists = self._circular_index_distance(centers_idx, indices, M)  # (k, M)
                W = self._window_weights(dists, s)  # (k, M)
                W = W / (W.sum(dim=1, keepdim=True) + self.eps)

                q_dyn = W @ self.bank  # (k, d)
                q_out.append(q_dyn.unsqueeze(0))
                # Report φ centers with phi_shift applied
                phi_vals = 2 * torch.pi * (centers_idx / M - self.phi_shift)
                phi_out.append(phi_vals.unsqueeze(0))

            q = torch.cat(q_out, dim=0)  # (B, k, d)
            phi_vals = torch.cat(phi_out, dim=0)  # (B, k)
            q_valid = torch.full((B, k), True, device=device)
            return {"query_embed": q, "query_valid": q_valid, "query_phi": phi_vals}

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'pick', 'window', or 'window_adaptive'.")


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
