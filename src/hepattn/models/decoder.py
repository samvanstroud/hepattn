"""Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former.
"""

from functools import partial

import torch
from torch import Tensor, nn
import math

from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.encoder import Residual
from hepattn.models.posenc import pos_enc_symmetric
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask
from hepattn.utils.local_ca import auto_local_ca_mask
from hepattn.utils.model_utils import unmerge_inputs


class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        mask_attention: bool = True,
        use_query_masks: bool = False,
        posenc: dict[str, float] | None = None,
        local_strided_attn: bool = False,
        window_size: int = 512,
        window_wrap: bool = True,
        align_phi: bool = False,
        align_phi_contiguos: bool = False,
        combine_and: bool = False,
        combine_or: bool = False,
    ):
        """MaskFormer decoder that handles multiple decoder layers and task integration.

        Args:
            num_queries: The number of object-level queries.
            decoder_layer_config: Configuration dictionary used to initialize each MaskFormerDecoderLayer.
            num_decoder_layers: The number of decoder layers to stack.
            mask_attention: If True, attention masks will be used to control which input constituents are attended to.
            use_query_masks: If True, predicted query masks will be used to control which queries are valid.
            posenc: Optional module for positional encoding.
            local_strided_attn: If True, uses local strided window attention.
            window_size: The size of the window for local strided window attention.
            window_wrap: If True, wraps the window for local strided window attention.
        """
        super().__init__()

        # Ensure mask_attention is passed to decoder layers
        decoder_layer_config = decoder_layer_config.copy()
        decoder_layer_config["mask_attention"] = mask_attention

        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.dim = decoder_layer_config["dim"]
        self.tasks: list | None = None  # Will be set by MaskFormer
        self.num_queries = num_queries
        self.mask_attention = mask_attention
        self.use_query_masks = use_query_masks
        self.posenc = posenc
        self.local_strided_attn = local_strided_attn
        self.attn_type = decoder_layer_config.get("attn_type", "torch")
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.initial_queries = nn.Parameter(torch.randn(self.num_queries, decoder_layer_config["dim"]))
        self.align_phi = align_phi
        self.align_phi_contiguos = align_phi_contiguos
        self.combine_and = combine_and
        self.combine_or = combine_or

        if self.local_strided_attn:
            assert self.attn_type == "torch", f"Invalid attention type when local_strided_attn is True: {self.attn_type}, must be 'torch'"
        # assert not (self.local_strided_attn and self.mask_attention), "local_strided_attn and mask_attention cannot both be True"

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        """Forward pass through decoder layers.

        Args:
            x: Dictionary containing embeddings and masks.
            input_names: List of input names for constructing attention masks.

        Returns:
            Tuple containing updated embeddings and outputs from each decoder layer and final outputs.
        """
        batch_size = x["key_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]

        # Generate the queries that represent objects
        x["query_embed"] = self.initial_queries.expand(batch_size, -1, -1)
        x["query_valid"] = torch.full((batch_size, self.num_queries), True, device=x["query_embed"].device)

        if self.posenc:
            x["query_posenc"], x["key_posenc"] = self.generate_positional_encodings(x)

        attn_mask = None
        if self.local_strided_attn:
            assert x["query_embed"].shape[0] == 1, "Local strided attention only supports batch size 1"
            decoder_mask = auto_local_ca_mask(x["query_embed"], x["key_embed"], self.window_size, wrap=self.window_wrap)

        outputs: dict[str, dict] = {}
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            if self.posenc:
                x["query_embed"], x["key_embed"] = self.add_positional_encodings(x)

            attn_masks: dict[str, torch.Tensor] = {}
            query_mask = None

            assert self.tasks is not None
            for task in self.tasks:
                if not task.has_intermediate_loss:
                    continue

                # Get the outputs of the task given the current embeddings
                task_outputs = task(x)

                # Update x with task outputs for downstream use
                if isinstance(task, IncidenceRegressionTask):
                    x["incidence"] = task_outputs[task.outputs[0]].detach()
                if isinstance(task, ObjectClassificationTask):
                    x["class_probs"] = task_outputs[task.outputs[0]].detach()

                outputs[f"layer_{layer_index}"][task.name] = task_outputs

                # Collect attention masks from different tasks
                task_attn_masks = task.attn_mask(task_outputs)
                for input_name, attn_mask in task_attn_masks.items():
                    if input_name in attn_masks:
                        attn_masks[input_name] |= attn_mask
                    else:
                        attn_masks[input_name] = attn_mask

                # Collect query masks
                if self.use_query_masks:
                    task_query_mask = task.query_mask(task_outputs)
                    if task_query_mask is not None:
                        query_mask = task_query_mask if query_mask is None else query_mask | task_query_mask
                        x["query_mask"] = query_mask

            # Construct the full attention mask for MaskAttention decoder
            if attn_masks and self.mask_attention:
                attn_mask = torch.full((batch_size, self.num_queries, num_constituents), False, device=x["key_embed"].device)
                for input_name, task_attn_mask in attn_masks.items():
                    attn_mask[x[f"key_is_{input_name}"].unsqueeze(1).expand_as(attn_mask)] = task_attn_mask.flatten()
            if self.local_strided_attn:
                if self.combine_and:
                    attn_mask = attn_mask & decoder_mask
                elif self.combine_or:
                    attn_mask = attn_mask | decoder_mask
                else:
                    attn_mask = decoder_mask

            if attn_mask is not None:
                outputs[f"layer_{layer_index}"]["attn_mask"] = attn_mask

            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"],
                x["key_embed"],
                attn_mask=attn_mask,
                q_mask=x.get("query_mask"),
                kv_mask=x.get("key_valid"),
            )

            # update the individual input constituent representations
            x = unmerge_inputs(x, input_names)

        return x, outputs

    def add_positional_encodings(self, x: dict):
        x["query_embed"] = x["query_embed"] + x["query_posenc"]
        x["key_embed"] = x["key_embed"] + x["key_posenc"]
        return x["query_embed"], x["key_embed"]

    def align_queries_to_keys(self, key_phi, Q):
        device = key_phi.device
        K = key_phi.numel()

        # --- 1) Quantile bin edges so each bin has ~K/Q keys
        qs = torch.linspace(0.0, 1.0, Q + 1, device=device)
        edges = torch.quantile(key_phi.float(), qs.clamp(0, 1))

        # Handle potential duplicate edges (flat regions)
        eps = torch.finfo(key_phi.dtype).eps
        edges = torch.maximum(edges, edges.clone().roll(1))
        edges[0] -= eps
        edges[-1] += eps

        # --- 2) Assign each key to a bin
        bin_ids = torch.bucketize(key_phi, edges) - 1    # in [0, Q-1]
        bin_ids.clamp_(0, Q - 1)

        # --- 3) Per-bin query φ as **median** of member keys (fallback = bin midpoint if empty)
        query_phi = torch.empty(Q, device=device, dtype=key_phi.dtype)
        for b in range(Q):
            mask = (bin_ids == b)
            if mask.any():
                query_phi[b] = key_phi[mask].median() + torch.pi
            else:
                # empty bin: place at the middle of its edges
                query_phi[b] = 0.5 * (edges[b] + edges[b + 1]) + torch.pi

        return query_phi

    def generate_positional_encodings(self, x: dict):
        if self.align_phi:
            key_phi = x["key_phi"]
            x["query_phi"] = self.align_queries_to_keys(key_phi, self.num_queries)
        # elif self.align_phi_contiguos:
        #     key_phi = x["key_phi"]
        #     x["query_phi"] = hybrid_queries_group_then_fill(key_phi, self.num_queries, target_per_group=3, tau=None, return_assign=False)
        else:
            x["query_phi"] = 2 * torch.pi * torch.arange(self.num_queries, device=x["query_embed"].device) / self.num_queries
        query_posenc = pos_enc_symmetric(x["query_phi"], self.dim, self.posenc["alpha"], self.posenc["base"])
        key_posenc = pos_enc_symmetric(x["key_phi"], self.dim, self.posenc["alpha"], self.posenc["base"])
        return query_posenc, key_posenc


# def ang_diff(a, b):
#     # signed shortest diff a-b in (-pi, pi]
#     return torch.atan2(torch.sin(a-b), torch.cos(a-b))

# def circ_mean(phi):
#     # do trig in fp32 (autocast-safe), cast back
#     x = phi.to(torch.float32)
#     s = torch.sin(x).mean()
#     c = torch.cos(x).mean()
#     return torch.atan2(s, c).to(phi.dtype)

# @torch.no_grad()
# def hybrid_queries_group_then_fill(phi, Q, target_per_group=3, tau=None, return_assign=False):
#     """
#     1) Make contiguous groups of keys on the circle (sort -> cut at largest gap -> sweep).
#     2) If number of data-driven groups G < Q, fill remaining (Q-G) queries with
#        evenly spaced angles across [0, 2π).
#     3) Return query phis sorted by increasing angle. (Keys remain assigned only to the data-driven groups.)

#     Args:
#       phi: 1-D tensor of key angles (radians), device/dtype preserved for outputs
#       Q:   total number of queries desired
#       target_per_group: approx size for each data-driven group
#       tau: optional max within-group spread (radians); if set, groups won’t exceed this much spread (greedy)
#       return_assign: if True, also returns per-key group ids for data-driven groups (synthetic fill slots get id = -1)

#     Returns:
#       centers: (Q,) tensor of query angles in increasing order
#       assign (optional): (K,) long tensor, mapping each key -> query id in [0..Q-1] (or -1 for synthetic slots)
#     """
#     device, dtype = phi.device, phi.dtype
#     K = int(phi.numel())
#     if K == 0:
#         centers = torch.linspace(0, 2*math.pi, steps=Q, device=device, dtype=dtype, endpoint=False)
#         return (centers, torch.empty(0, device=device, dtype=torch.long)) if return_assign else centers

#     Q = int(Q)
#     assert Q >= 1

#     # ---- A) Sort and cut after largest gap to linearize the circle
    # phi_sorted, idx_sorted = torch.sort(phi)
    # gaps = torch.cat([phi_sorted[1:] - phi_sorted[:-1],
    #                   phi_sorted[:1] + 2*torch.pi - phi_sorted[-1:]], dim=0)
    # cut = gaps.argmax().item()
    # phi_lin = torch.roll(phi_sorted, shifts=-(cut + 1), dims=0)
    # idx_lin = torch.roll(idx_sorted,  shifts=-(cut + 1), dims=0)

    # # ---- B) Sweep to make contiguous groups near target_per_group, respecting tau if given
    # groups = []
    # start = 0
    # while start < K:
    #     # start with ~target size
    #     end = min(K, start + int(target_per_group))
    #     if end <= start:
    #         end = start + 1
    #     # greedily extend while within spread tolerance (if tau set)
    #     if tau is not None and tau > 0:
    #         while end < K and torch.abs(ang_diff(phi_lin[end-1], phi_lin[start])) <= tau:
    #             end += 1
    #     groups.append((start, end))
    #     start = end

    # # centers for data-driven groups
    # centers_data = []
    # for (a, b) in groups:
    #     seg = phi_lin[a:b]
    #     centers_data.append(circ_mean(seg))
    # centers_data = torch.stack(centers_data, dim=0) if groups else torch.empty(0, device=device, dtype=dtype)

    # # ---- C) If we have fewer groups than Q, fill remaining uniformly across [0, 2π)
    # G = int(centers_data.numel())
    # if G >= Q:
    #     centers_all = centers_data[:Q]  # if you ever want to trim
    #     fill_ids = torch.empty(0, dtype=torch.long, device=device)
    # else:
    #     M = Q - G
    #     # Evenly spaced fill grid; choose a phase that is as far from data-centers as possible
    #     grid = torch.linspace(0, 2*math.pi, steps=M, device=device, dtype=dtype, endpoint=False)

    #     if G == 0:
    #         fill = grid  # no data centers; just uniform
    #     else:
    #         # choose offset among M candidates to maximize minimum distance to data centers
    #         # offsets: 0, Δ, 2Δ, ..., (M-1)Δ where Δ=2π/M
    #         delta = 2*math.pi / M
    #         offsets = torch.arange(M, device=device, dtype=dtype) * delta
    #         best_idx = 0
    #         best_score = -1.0
    #         for i in range(M):
    #             cand = (grid + offsets[i]).remainder(2*math.pi)
    #             # distance to nearest data center (on circle)
    #             d = torch.min(torch.abs(ang_diff(cand[:, None], centers_data[None, :])), dim=1).values
    #             score = d.min()  # maximize worst-case separation
    #             if score > best_score:
    #                 best_score = score
    #                 best_idx = i
    #         fill = (grid + offsets[best_idx]).remainder(2*math.pi)

    #     centers_all = torch.cat([centers_data.remainder(2*math.pi), fill], dim=0)

    # # ---- D) Sort centers by increasing angle in [0, 2π), then map back to (-π, π] if you prefer
    # two_pi = 2 * math.pi
    # c_norm = torch.remainder(centers_all, two_pi)
    # perm = torch.argsort(c_norm)
    # centers_sorted = centers_all[perm]

    # return centers_sorted


class MaskFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        depth: int = 0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        mask_attention: bool = True,
        bidirectional_ca: bool = True,
        hybrid_norm: bool = False,
    ) -> None:
        """Initialize a MaskFormer decoder layer.

        Args:
            dim: Embedding dimension.
            norm: Normalization type.
            depth: Layer depth index.
            dense_kwargs: Optional arguments for Dense layers.
            attn_kwargs: Optional arguments for Attention layers.
            mask_attention: If True, enables mask attention.
            bidirectional_ca: If True, enables bidirectional cross-attention.
            hybrid_norm: If True, enables hybrid normalization.
        """
        super().__init__()

        self.dim = dim
        self.mask_attention = mask_attention
        self.bidirectional_ca = bidirectional_ca

        # handle hybridnorm
        qkv_norm = hybrid_norm
        if depth == 0:
            hybrid_norm = False
        attn_norm = norm if not hybrid_norm else None
        dense_post_norm = not hybrid_norm

        attn_kwargs = attn_kwargs or {}
        dense_kwargs = dense_kwargs or {}

        residual = partial(Residual, dim=dim, norm=norm)
        self.q_ca = residual(Attention(dim, qkv_norm=qkv_norm, **attn_kwargs), norm=attn_norm)
        self.q_sa = residual(Attention(dim, qkv_norm=qkv_norm, **attn_kwargs), norm=attn_norm)
        self.q_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

        if self.bidirectional_ca:
            self.kv_ca = residual(Attention(dim, qkv_norm=qkv_norm, **attn_kwargs), norm=attn_norm)
            self.kv_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

    def forward(self, q: Tensor, kv: Tensor, attn_mask: Tensor | None = None, q_mask: Tensor | None = None, kv_mask: Tensor | None = None) -> Tensor:
        """Forward pass for the decoder layer.

        Args:
            q: Query embeddings.
            kv: Key/value embeddings.
            attn_mask: Optional attention mask.
            q_mask: Optional query mask.
            kv_mask: Optional key/value mask.

        Returns:
            Tuple of updated query and key/value embeddings.
        """
        if self.mask_attention:
            assert attn_mask is not None, "attn_mask must be provided for mask attention"
            attn_mask = attn_mask.detach()
            # True values indicate a slot will be included in the attention computation, while False will be ignored.
            # If the attn mask is completely invalid for a given query, allow it to attend everywhere
            attn_mask = torch.where(torch.all(~attn_mask, dim=-1, keepdim=True), True, attn_mask)
        else:
            attn_mask = None

        # Update query/object embeddings with the key/constituent embeddings
        q = self.q_ca(q, kv=kv, attn_mask=attn_mask, q_mask=q_mask, kv_mask=kv_mask)
        q = self.q_sa(q, q_mask=q_mask)
        q = self.q_dense(q)

        # Update key/constituent embeddings with the query/object embeddings
        if self.bidirectional_ca:
            if attn_mask is not None:
                # Index from the back so we are batch shape agnostic
                attn_mask = attn_mask.transpose(-2, -1)

            kv = self.kv_ca(kv, kv=q, attn_mask=attn_mask, q_mask=kv_mask, kv_mask=q_mask)
            kv = self.kv_dense(kv)

        return q, kv

    def set_backend(self, attn_type: str) -> None:
        """Set the backend for the attention layers.

        Args:
            attn_type: Attention implementation type to use.
        """
        self.q_ca.fn.set_backend(attn_type)
        self.q_sa.fn.set_backend(attn_type)

        if self.bidirectional_ca:
            self.kv_ca.fn.set_backend(attn_type)
