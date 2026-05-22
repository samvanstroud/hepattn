"""Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former.
"""

from functools import partial

import torch
from torch import Tensor, nn

from hepattn.flex.local_ca import flex_local_ca_mask
from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.encoder import Residual
from hepattn.models.norm import get_hybrid_norm_config
from hepattn.models.posenc import pos_enc_symmetric
from hepattn.utils.local_ca import auto_local_ca_mask
from hepattn.utils.model_utils import unmerge_inputs


class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        kv_input_name: str = "key",
        secondary_kv_input_name: str | None = None,
        mask_attention: bool = True,
        use_query_masks: bool = False,
        posenc: dict[str, float] | None = None,
        local_strided_attn: bool = False,
        window_size: int = 512,
        window_wrap: bool = True,
        block_size: int = 128,
        unified_decoding: bool = False,
        phi_shift: float = 0.0,
        unmask_all_false: bool = True,
        dynamic_queries: bool = False,
        dynamic_query_source: str | None = None,
        debug: bool = False,
        intermediate_tasks: bool = True,
    ):
        """MaskFormer decoder that handles multiple decoder layers and task integration.

        Args:
            num_queries: The number of object-level queries.
            decoder_layer_config: Configuration dictionary used to initialize each MaskFormerDecoderLayer.
            num_decoder_layers: The number of decoder layers to stack.
            kv_input_name: Name of the constituent set used as decoder keys/values.
            secondary_kv_input_name: Optional second constituent set for an additional cross-attention stage.
            mask_attention: If True, attention masks will be used to control which input constituents are attended to.
            use_query_masks: If True, predicted query masks will be used to control which queries are valid.
            posenc: Optional module for positional encoding.
            local_strided_attn: If True, uses local strided window attention.
            window_size: The size of the window for local strided window attention.
            window_wrap: If True, wraps the window for local strided window attention.
            block_size: The size of the block for fast local CA.
            unified_decoding: If True, inputs remain merged for task processing instead of being unmerged after each layer.
            phi_shift: The shift in the phi angle for positional encoding.
            unmask_all_false: If True, queries with all-false attention masks will be unmasked to attend everywhere.
            dynamic_queries: If True, queries are initialized dynamically.
            dynamic_query_source: Name of the input type to use as the source for dynamic query initialization.
            debug: If True, stores non-flex attention masks in decoder outputs for debugging.
            intermediate_tasks: If True, run task heads at every decoder layer instead of only the last.
        """
        super().__init__()

        decoder_layer_config = decoder_layer_config.copy()
        decoder_layer_config["secondary_cross_attention"] = secondary_kv_input_name is not None
        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.dim = decoder_layer_config["dim"]
        self.tasks: list | None = None  # Will be set by MaskFormer
        self.encoder_tasks: list | None = None  # Will be set by MaskFormer
        self._num_queries = num_queries
        self.kv_input_name = kv_input_name
        self.secondary_kv_input_name = secondary_kv_input_name
        self.mask_attention = mask_attention
        self.use_query_masks = use_query_masks
        self.posenc = posenc
        self.local_strided_attn = local_strided_attn
        self.attn_type = decoder_layer_config.get("attn_kwargs", {}).get("attn_type", "torch")
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.unified_decoding = unified_decoding
        self.dynamic_queries = dynamic_queries
        self.dynamic_query_source = dynamic_query_source
        self.debug = debug
        self.intermediate_tasks = intermediate_tasks

        # Only initialize learned queries if not using dynamic queries
        if not dynamic_queries:
            self.initial_queries = nn.Parameter(torch.randn(self._num_queries, decoder_layer_config["dim"]))

        self.block_size = block_size
        self.phi_shift = phi_shift
        self.unmask_all_false = unmask_all_false

        if self.local_strided_attn:
            assert self.attn_type in {"torch", "flex"}, (
                f"Invalid attention type when local_strided_attn is True: {self.attn_type}, must be 'torch' or 'flex'"
            )
        assert not (self.local_strided_attn and self.mask_attention), "local_strided_attn and mask_attention cannot both be True"
        if self.local_strided_attn and self.secondary_kv_input_name is not None:
            raise ValueError("local_strided_attn does not support secondary_kv_input_name")

    @staticmethod
    def _decoder_kv_keys(kv_input_name: str) -> tuple[str, str]:
        return f"{kv_input_name}_embed", f"{kv_input_name}_valid"

    def _key_posenc(self, x: dict[str, Tensor], kv_input_name: str, kv_select: Tensor | None = None) -> Tensor:
        key_phi_key = f"{kv_input_name}_phi"
        key_phi = x[key_phi_key] if key_phi_key in x else x["key_phi"]
        if kv_select is not None:
            key_phi = key_phi[:, kv_select]
        return pos_enc_symmetric(key_phi, self.dim, self.posenc["alpha"], self.posenc["base"])

    def _compact_decoder_kv_stream(
        self,
        x: dict[str, Tensor],
        kv_input_name: str,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Compact the decoder KV stream to real tokens only.

        For local-strided attention the window geometry should be defined over actual tokens,
        not padded slots. When a validity mask is present, compact the selected KV stream to
        valid tokens only and drop the KV mask for downstream attention.
        """
        kv_embed_key, kv_valid_key = self._decoder_kv_keys(kv_input_name)
        kv_embed = x[kv_embed_key]
        kv_valid = x.get(kv_valid_key)

        if not self.local_strided_attn or kv_valid is None:
            return kv_embed, kv_valid, None

        assert kv_embed.shape[0] == 1, "Local strided attention only supports batch size 1"
        kv_select = kv_valid[0].bool()

        if kv_select.all():
            return kv_embed, None, None

        if kv_select.sum() == 0:
            raise ValueError(f"Decoder KV stream '{kv_input_name}' has no valid tokens after masking.")

        compact_kv_embed = kv_embed[:, kv_select, :]
        return compact_kv_embed, None, kv_select

    def _build_attn_mask(
        self,
        x: dict[str, Tensor],
        attn_masks: dict[str, torch.Tensor],
        kv_input_name: str,
        num_constituents: int,
    ) -> torch.Tensor | None:
        if not attn_masks or not self.mask_attention:
            return None

        if self.unified_decoding:
            if kv_input_name == "key":
                if len(attn_masks) > 1:
                    raise ValueError(f"In merged input mode, expected only one attention mask, got {len(attn_masks)}")
                attn_mask = next(iter(attn_masks.values()))
            else:
                attn_mask = attn_masks.get(kv_input_name)
                if attn_mask is None:
                    return None

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(-1).expand(-1, -1, num_constituents)
        else:
            attn_mask = torch.full((x["key_embed"].shape[0], self.num_queries(x), num_constituents), False, device=x["key_embed"].device)
            if kv_input_name == "key":
                for input_name, task_attn_mask in attn_masks.items():
                    attn_mask[x[f"key_is_{input_name}"].unsqueeze(1).expand_as(attn_mask)] = task_attn_mask.flatten()
            else:
                task_attn_mask = attn_masks.get(kv_input_name)
                if task_attn_mask is None:
                    return None
                attn_mask = task_attn_mask
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(-1).expand(-1, -1, num_constituents)

        attn_mask = attn_mask.detach()
        if self.unmask_all_false:
            attn_mask = torch.where(torch.all(~attn_mask, dim=-1, keepdim=True), True, attn_mask)
        return attn_mask

    def _update_decoder_kv(
        self,
        x: dict[str, Tensor],
        input_names: list[str],
        kv_input_name: str,
        decoder_kv_embed: Tensor,
        decoder_kv_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        kv_embed_key, kv_valid_key = self._decoder_kv_keys(kv_input_name)
        x[kv_embed_key] = decoder_kv_embed

        existing_kv_valid = x.get(kv_valid_key)
        if decoder_kv_mask is not None:
            x[kv_valid_key] = decoder_kv_mask
        elif existing_kv_valid is not None and existing_kv_valid.shape[-1] != decoder_kv_embed.shape[1]:
            x[kv_valid_key] = torch.ones(
                decoder_kv_embed.shape[:2],
                dtype=torch.bool,
                device=decoder_kv_embed.device,
            )

        if kv_input_name == "key":
            x["key_embed"] = decoder_kv_embed
            if any(f"key_is_{input_name}" in x for input_name in input_names):
                x = unmerge_inputs(x, input_names)
        return x

    def num_queries(self, x) -> int:
        if self.dynamic_queries:
            return x["query_embed"].shape[1]
        return self._num_queries

    def initialize_dynamic_queries(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Initialize queries dynamically using the `query_init` task.

        This selects hit embeddings whose predicted first-hit probability passes the task threshold,
        then keeps the top-k by probability where k is `self._num_queries`.

        Notes:
            - Only supports batch_size == 1.
            - Expects `x` to contain `source_embed` and `source_valid`.

        Returns:
            Tuple of:
                - query_embed: (1, N_queries, dim) query embeddings
                - query_valid: (1, N_queries) validity mask (always created for torch.compile compatibility)

        Raises:
            ValueError: If `decoder.dynamic_queries` is False, decoder tasks are unset, the `query_init` task
                is missing, required source tensors are missing, or batch size is not 1.
        """
        if not self.dynamic_queries:
            raise ValueError("initialize_dynamic_queries called but decoder.dynamic_queries is False")

        if self.encoder_tasks is None:
            raise ValueError("dynamic_queries=True requires decoder.encoder_tasks to be set")

        # Find query_init task (must be in encoder_tasks)
        query_init_task = next((t for t in self.encoder_tasks if t.name == "query_init"), None)
        if query_init_task is None:
            raise ValueError("dynamic_queries=True requires a task named 'query_init' in encoder_tasks.")

        source_embed_key = f"{self.dynamic_query_source}_embed"
        source_valid_key = f"{self.dynamic_query_source}_valid"
        if source_embed_key not in x or source_valid_key not in x:
            raise ValueError(f"dynamic_queries=True requires '{source_embed_key}' and '{source_valid_key}' in decoder input dict")

        # Get predictions
        outputs = query_init_task.forward(x)
        preds = query_init_task.predict(outputs)

        # For the source field, tasks typically return predictions with keys matching task.predict_outputs
        # We look for the "prob" prediction for the query source.
        # TODO: This still makes some assumptions about the task output structure.
        prob_key = next((k for k in preds if k.endswith("_prob")), None)
        if prob_key is None:
            raise ValueError(f"query_init task '{query_init_task.name}' did not return a probability prediction.")

        source_prob = preds[prob_key]  # (B, N_source)
        if source_prob.shape[0] != 1:
            raise ValueError(f"dynamic_queries only supports batch_size=1, got {source_prob.shape[0]}")

        # Filter: probability >= threshold AND valid, then select top-k by probability
        valid_mask = (source_prob[0] >= query_init_task.threshold) & x[source_valid_key][0]
        selected_indices = torch.where(valid_mask)[0]

        if selected_indices.numel() == 0:
            selected_indices = source_prob[0].topk(self._num_queries).indices

        # If more candidates than needed, keep top-k by probability
        if selected_indices.numel() > self._num_queries:
            probs = source_prob[0, selected_indices]
            top_k_idx = probs.topk(self._num_queries).indices
            selected_indices = selected_indices[top_k_idx]

        # Sort to preserve original spatial ordering
        selected_indices = selected_indices.sort().values
        num_selected = selected_indices.numel()
        device = x[source_embed_key].device

        # get selected embeddings
        selected_constituent_embeds = x[source_embed_key][0, selected_indices].detach()

        # Pad to fixed num_queries size if fewer were selected to match the full target set
        if num_selected < self._num_queries:
            num_padding = self._num_queries - num_selected
            # Pad with zero embeddings
            null_padding = torch.zeros(num_padding, self.dim, device=device, dtype=selected_constituent_embeds.dtype)
            query_embed = torch.cat([selected_constituent_embeds, null_padding], dim=0).unsqueeze(0)  # (1, num_queries, dim)
            # Validity: selected queries are valid, padding is not
            query_valid = torch.cat([
                torch.ones(num_selected, dtype=torch.bool, device=device),
                torch.zeros(num_padding, dtype=torch.bool, device=device),
            ]).unsqueeze(0)  # (1, num_queries)
        else:
            query_embed = selected_constituent_embeds.unsqueeze(0)  # (1, N_selected, dim)
            # All queries are valid - still create mask for consistent control flow with torch.compile
            query_valid = torch.ones(1, num_selected, dtype=torch.bool, device=device)

        return query_embed, query_valid

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        """Forward pass through decoder layers.

        Args:
            x: Dictionary containing embeddings and masks.
            input_names: List of input names for constructing attention masks.

        Returns:
            Tuple containing:
                - x: Updated embedding dictionary.
                - outputs: Layer-wise task outputs, including an "encoder" key with metadata.

        Raises:
            ValueError: If in merged input mode and multiple attention masks are provided.
        """
        kv_embed_key, kv_valid_key = self._decoder_kv_keys(self.kv_input_name)
        batch_size = x["key_embed"].shape[0]
        decoder_kv_embed, decoder_kv_mask, decoder_kv_select = self._compact_decoder_kv_stream(x, self.kv_input_name)
        num_constituents = decoder_kv_embed.shape[-2]

        # Generate or use pre-initialized queries
        outputs: dict[str, dict] = {"encoder": {}}
        if not self.dynamic_queries:
            # Static learned queries (backward compatible)
            x["query_embed"] = self.initial_queries.expand(batch_size, -1, -1)
        else:
            # Initialize dynamic queries
            x["query_embed"], query_valid = self.initialize_dynamic_queries(x)
            # Always set query_mask for consistent control flow with torch.compile
            x["query_mask"] = query_valid
            outputs["encoder"]["query_mask"] = query_valid

        if self.posenc:
            x["query_posenc"] = self.generate_positional_encodings(x)

        attn_mask = None
        if self.local_strided_attn:
            assert x["query_embed"].shape[0] == 1, "Local strided attention only supports batch size 1"
            q_len = x["query_embed"].shape[1]
            query_mask = x.get("query_mask")
            query_valid_mask = query_mask[0] if query_mask is not None else None
            kv_len = decoder_kv_embed.shape[1]
            if self.attn_type == "torch":
                if query_valid_mask is None:
                    attn_mask = auto_local_ca_mask(
                        x["query_embed"],
                        decoder_kv_embed,
                        self.window_size,
                        wrap=self.window_wrap,
                    )
                else:
                    device = x["query_embed"].device
                    dtype_float = x["query_embed"].dtype
                    stride_q_len = torch.clamp(query_valid_mask.sum(dtype=dtype_float), min=1.0)
                    stride = torch.as_tensor(kv_len, device=device, dtype=dtype_float) / stride_q_len

                    q_idx = torch.arange(q_len, device=device, dtype=dtype_float)
                    kv_idx = torch.arange(kv_len, device=device, dtype=dtype_float)
                    q_center = torch.round(q_idx * stride)

                    half_window = self.window_size // 2
                    diagonal = (kv_idx.unsqueeze(0) - q_center.unsqueeze(1)).abs() <= half_window

                    if self.window_wrap:
                        kv_len_t = torch.as_tensor(kv_len, device=device, dtype=dtype_float)
                        wrap_left = (kv_idx.unsqueeze(0) - q_center.unsqueeze(1) + kv_len_t).abs() <= half_window
                        wrap_right = (kv_idx.unsqueeze(0) - q_center.unsqueeze(1) - kv_len_t).abs() <= half_window
                        diagonal = diagonal | wrap_left | wrap_right

                    attn_mask = (query_valid_mask.unsqueeze(-1) & diagonal).unsqueeze(0)
            elif self.attn_type == "flex":
                device = x["query_embed"].device
                dtype_float = x["query_embed"].dtype
                if query_valid_mask is None:
                    stride_q_len: int | Tensor = q_len
                else:
                    stride_q_len = torch.clamp(query_valid_mask.sum(dtype=dtype_float), min=1.0)
                attn_mask = flex_local_ca_mask(
                    q_len=q_len,
                    kv_len=kv_len,
                    device=device,
                    dtype_float=dtype_float,
                    self=self,
                    stride_q_len=stride_q_len,
                    query_valid_mask=query_valid_mask,
                )

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            attn_masks: dict[str, torch.Tensor] = {}
            is_last = layer_index == len(self.decoder_layers) - 1

            if is_last or self.intermediate_tasks:

                assert self.tasks is not None
                for task in self.tasks:
                    if not task.should_run_at_layer(layer_index):
                        continue

                    # Get the outputs of the task given the current embeddings
                    # Pass current layer's outputs so tasks can read from previously executed tasks
                    task_outputs = task(x, outputs=outputs[f"layer_{layer_index}"])

                    outputs[f"layer_{layer_index}"][task.name] = task_outputs

                    # Collect attention masks from different tasks
                    task_attn_masks = task.attn_mask(task_outputs)
                    for input_name, task_attn_mask in task_attn_masks.items():
                        if input_name in attn_masks:
                            attn_masks[input_name] |= task_attn_mask
                        else:
                            attn_masks[input_name] = task_attn_mask

            secondary_kv_embed = None
            secondary_kv_mask = None
            secondary_attn_mask = None
            secondary_key_posenc = None
            if self.secondary_kv_input_name is not None:
                secondary_kv_embed_key, secondary_kv_valid_key = self._decoder_kv_keys(self.secondary_kv_input_name)
                secondary_kv_embed = x[secondary_kv_embed_key]
                secondary_kv_mask = x.get(secondary_kv_valid_key)
                if self.posenc:
                    secondary_key_posenc = self._key_posenc(x, self.secondary_kv_input_name)
                secondary_attn_mask = self._build_attn_mask(x, attn_masks, self.secondary_kv_input_name, secondary_kv_embed.shape[-2])

            if not self.local_strided_attn:
                attn_mask = self._build_attn_mask(x, attn_masks, self.kv_input_name, num_constituents)

            if self.attn_type != "flex" and self.debug:
                if attn_mask is not None:
                    outputs[f"layer_{layer_index}"]["attn_mask"] = attn_mask
                if secondary_attn_mask is not None:
                    outputs[f"layer_{layer_index}"]["secondary_attn_mask"] = secondary_attn_mask

            layer_out = decoder_layer(
                x["query_embed"],
                decoder_kv_embed,
                attn_mask=attn_mask,
                q_mask=x.get("query_mask"),
                kv_mask=decoder_kv_mask,
                query_posenc=x["query_posenc"] if self.posenc else None,
                key_posenc=self._key_posenc(x, self.kv_input_name, decoder_kv_select) if self.posenc else None,
                secondary_kv=secondary_kv_embed,
                secondary_attn_mask=secondary_attn_mask,
                secondary_kv_mask=secondary_kv_mask,
                secondary_key_posenc=secondary_key_posenc,
            )

            if self.secondary_kv_input_name is None:
                x["query_embed"], decoder_kv_embed = layer_out
                secondary_decoder_kv_embed = None
            else:
                x["query_embed"], decoder_kv_embed, secondary_decoder_kv_embed = layer_out

            x = self._update_decoder_kv(x, input_names, self.kv_input_name, decoder_kv_embed, decoder_kv_mask)
            if self.secondary_kv_input_name is not None and secondary_decoder_kv_embed is not None:
                x = self._update_decoder_kv(
                    x,
                    input_names,
                    self.secondary_kv_input_name,
                    secondary_decoder_kv_embed,
                    secondary_kv_mask,
                )

        return x, outputs

    def generate_positional_encodings(self, x: dict):
        idx = torch.arange(self.num_queries(x), device=x["query_embed"].device, dtype=x["query_embed"].dtype)
        x["query_phi"] = 2 * torch.pi * (idx / self.num_queries(x) - self.phi_shift)
        return pos_enc_symmetric(x["query_phi"], self.dim, self.posenc["alpha"], self.posenc["base"])


class MaskFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        depth: int = 0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        secondary_cross_attention: bool = False,
    ) -> None:
        """Initialize a MaskFormer decoder layer.

        Args:
            dim: Embedding dimension.
            norm: Normalization type.
            depth: Layer depth index.
            dense_kwargs: Optional arguments for Dense layers.
            attn_kwargs: Optional arguments for Attention layers.
            qkv_norm: Apply normalization to QKV in attention.
            hybrid_norm: Enable hybrid normalization from 2503.04598.
            secondary_cross_attention: Enable a second query-to-constituent cross-attention stage.
        """
        super().__init__()
        self.dim = dim
        self.secondary_cross_attention = secondary_cross_attention

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

        attn_kwargs = attn_kwargs or {}
        self.attn_type = attn_kwargs.get("attn_type", "torch")
        dense_kwargs = dense_kwargs or {}

        residual = partial(Residual, dim=dim)
        self.q_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.q_sa = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.q_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)
        if self.secondary_cross_attention:
            self.secondary_q_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
            self.secondary_q_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        attn_mask: Tensor | None = None,
        q_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        query_posenc: Tensor | None = None,
        key_posenc: Tensor | None = None,
        secondary_kv: Tensor | None = None,
        secondary_attn_mask: Tensor | None = None,
        secondary_kv_mask: Tensor | None = None,
        secondary_key_posenc: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Forward pass for the decoder layer.

        Args:
            q: Query embeddings.
            kv: Key/value embeddings.
            attn_mask: Optional attention mask.
            q_mask: Optional query mask.
            kv_mask: Optional key/value mask.
            query_posenc: Optional query positional encoding.
            key_posenc: Optional key positional encoding.
            secondary_kv: Optional second key/value tensor for an additional cross-attention stage.
            secondary_attn_mask: Optional attention mask for the second stage.
            secondary_kv_mask: Optional key/value mask for the second stage.
            secondary_key_posenc: Optional positional encoding for the second stage.

        Returns:
            tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - The updated query embeddings (Tensor).
                - The primary key/value embeddings used by the decoder.
                - The secondary key/value embeddings used by the decoder, if present.
        """
        q_pe = q if query_posenc is None else q + query_posenc
        kv_pe = kv if key_posenc is None else kv + key_posenc

        q = self.q_ca(q_pe, k=kv_pe, v=kv, attn_mask=attn_mask, q_mask=q_mask, kv_mask=kv_mask)
        q = self.q_dense(q)

        if secondary_kv is not None:
            if not self.secondary_cross_attention:
                raise ValueError("secondary_kv was provided but secondary_cross_attention is disabled")

            q_pe = q if query_posenc is None else q + query_posenc
            secondary_kv_pe = secondary_kv if secondary_key_posenc is None else secondary_kv + secondary_key_posenc
            q = self.secondary_q_ca(
                q_pe,
                k=secondary_kv_pe,
                v=secondary_kv,
                attn_mask=secondary_attn_mask,
                q_mask=q_mask,
                kv_mask=secondary_kv_mask,
            )
            q = self.secondary_q_dense(q)

        q = self.q_sa(q, k=q, v=q, q_mask=q_mask)
        if secondary_kv is not None:
            return q, kv, secondary_kv
        return q, kv

    def set_backend(self, attn_type: str) -> None:
        """Set the backend for the attention layers.

        Args:
            attn_type: Attention implementation type to use.
        """
        self.q_ca.fn.set_backend(attn_type)
        self.q_sa.fn.set_backend(attn_type)
        if self.secondary_cross_attention:
            self.secondary_q_ca.fn.set_backend(attn_type)
