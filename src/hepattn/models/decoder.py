"""Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former.
"""

from functools import partial

import torch
from torch import Tensor, nn

from hepattn.flex.fast_local_ca import build_strided_sliding_window_blockmask
from hepattn.flex.local_ca import sliding_window_mask_strided, sliding_window_mask_strided_wrapped, transpose_blockmask
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
        mask_attention: bool = True,
        use_query_masks: bool = False,
        posenc: dict[str, float] | None = None,
        local_strided_attn: bool = False,
        window_size: int = 512,
        window_wrap: bool = True,
        fast_local_ca: bool = False,
        block_size: int = 128,
        unified_decoding: bool = False,
        phi_shift: float = 0.0,
        unmask_all_false: bool = True,
        dynamic_queries: bool = False,
        dynamic_query_source: str | None = None,
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
            attn_type: The attention type to use (e.g., 'torch', 'flex').
            fast_local_ca: If True, uses fast local CA.
            block_size: The size of the block for fast local CA.
            unified_decoding: If True, inputs remain merged for task processing instead of being unmerged after each layer.
            phi_shift: The shift in the phi angle for positional encoding.
            unmask_all_false: If True, queries with all-false attention masks will be unmasked to attend everywhere.
            dynamic_queries: If True, queries are initialized dynamically.
            dynamic_query_source: Name of the input type to use as the source for dynamic query initialization.
        """
        super().__init__()

        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.dim = decoder_layer_config["dim"]
        self.tasks: list | None = None  # Will be set by MaskFormer
        self.encoder_tasks: list | None = None  # Will be set by MaskFormer
        self._num_queries = num_queries
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

        # Only initialize learned queries if not using dynamic queries
        if not dynamic_queries:
            self.initial_queries = nn.Parameter(torch.randn(self._num_queries, decoder_layer_config["dim"]))

        self.fast_local_ca = fast_local_ca
        self.block_size = block_size
        self.phi_shift = phi_shift
        self.unmask_all_false = unmask_all_false

        if self.local_strided_attn:
            assert self.attn_type in {"torch", "flex"}, (
                f"Invalid attention type when local_strided_attn is True: {self.attn_type}, must be 'torch' or 'flex'"
            )
        assert not (self.local_strided_attn and self.mask_attention), "local_strided_attn and mask_attention cannot both be True"

    def num_queries(self, x) -> int:
        if self.dynamic_queries:
            return x["query_embed"].shape[1]
        return self._num_queries

    def initialize_dynamic_queries(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Initialize queries dynamically using the `query_init` task.

        This selects hit embeddings whose predicted first-hit probability passes the task threshold,
        then keeps the top-k by probability where k is `self._num_queries`.

        Notes:
            - Only supports batch_size == 1.
            - Expects `x` to contain `source_embed` and `source_valid`.

        Returns:
            (query_embed, query_valid, selected_indices)

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
            raise ValueError("dynamic query initialization selected 0 constituents. Lower the query_init threshold or disable dynamic_queries.")

        # If more candidates than needed, keep top-k by probability
        if selected_indices.numel() > self._num_queries:
            probs = source_prob[0, selected_indices]
            top_k_idx = probs.topk(self._num_queries).indices
            selected_indices = selected_indices[top_k_idx]

        # Sort to preserve original spatial ordering
        final_indices = selected_indices.sort().values
        query_embed = x[source_embed_key][0, final_indices].detach().unsqueeze(0)  # (1, N_queries, dim)
        query_valid = torch.ones(1, final_indices.numel(), dtype=torch.bool, device=query_embed.device)

        return query_embed, query_valid, final_indices

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        """Forward pass through decoder layers.

        Args:
            x: Dictionary containing embeddings and masks.
            input_names: List of input names for constructing attention masks.

        Returns:
            Tuple containing:
                - x: Updated embedding dictionary.
                - outputs: Layer-wise task outputs.
                - metadata: Dictionary containing additional runtime information (e.g., selected_query_indices).

        Raises:
            ValueError: If in merged input mode and multiple attention masks are provided.
        """
        batch_size = x["key_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]

        # Generate or use pre-initialized queries
        metadata = {}
        if not self.dynamic_queries:
            # Static learned queries (backward compatible)
            x["query_embed"] = self.initial_queries.expand(batch_size, -1, -1)
            x["query_valid"] = torch.full((batch_size, self.num_queries(x)), True, device=x["query_embed"].device)
        else:
            # Initialize dynamic queries (will raise if required inputs are missing)
            x["query_embed"], x["query_valid"], selected_indices = self.initialize_dynamic_queries(x)
            metadata["selected_query_indices"] = selected_indices

        if self.posenc:
            x["query_posenc"], x["key_posenc"] = self.generate_positional_encodings(x)

        attn_mask = None
        attn_mask_transpose = None
        if self.local_strided_attn:
            assert x["query_embed"].shape[0] == 1, "Local strided attention only supports batch size 1"
            if self.attn_type == "torch":
                attn_mask = auto_local_ca_mask(x["query_embed"], x["key_embed"], self.window_size, wrap=self.window_wrap)
            elif self.attn_type == "flex":
                device = x["query_embed"].device
                q_len = x["query_embed"].shape[1]
                kv_len = x["key_embed"].shape[1]
                dtype_float = x["query_embed"].dtype
                attn_mask = self.flex_local_ca_mask(q_len, kv_len, device, dtype_float)
                attn_mask_transpose = transpose_blockmask(attn_mask, q_tokens=q_len, kv_tokens=kv_len, dev=device)

        outputs: dict[str, dict] = {}
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            # if maskattention, PE should be added before generating the mask
            if self.posenc and self.mask_attention:
                x["query_embed"] = x["query_embed"] + x["query_posenc"]
                x["key_embed"] = x["key_embed"] + x["key_posenc"]

            attn_masks: dict[str, torch.Tensor] = {}
            query_mask = None

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

            # Store selected_query_indices in layer_0 if using dynamic queries
            if layer_index == 0 and "selected_query_indices" in x:
                outputs["layer_0"]["_selected_query_indices"] = x["selected_query_indices"]

                # Collect query masks
                if self.use_query_masks:
                    task_query_mask = task.query_mask(task_outputs)
                    if task_query_mask is not None:
                        query_mask = task_query_mask if query_mask is None else query_mask | task_query_mask
                        x["query_mask"] = query_mask

            # Construct the full attention mask for MaskAttention decoder
            if attn_masks and self.mask_attention:
                if self.unified_decoding:
                    # In merged input mode, tasks should return masks directly for the full merged tensor
                    # We expect only one mask key (likely "key" or similar) that covers all constituents
                    if len(attn_masks) > 1:
                        raise ValueError(f"In merged input mode, expected only one attention mask, got {len(attn_masks)}")
                    attn_mask = next(iter(attn_masks.values()))
                    # Ensure proper shape: (batch, num_queries, num_constituents)
                    if attn_mask.dim() == 2:  # (batch, num_queries) -> (batch, num_queries, num_constituents)
                        attn_mask = attn_mask.unsqueeze(-1).expand(-1, -1, num_constituents)
                else:
                    # Original logic for separate input types
                    attn_mask = torch.full((batch_size, self.num_queries(x), num_constituents), False, device=x["key_embed"].device)
                    for input_name, task_attn_mask in attn_masks.items():
                        attn_mask[x[f"key_is_{input_name}"].unsqueeze(1).expand_as(attn_mask)] = task_attn_mask.flatten()

                attn_mask = attn_mask.detach()
                # True values indicate a slot will be included in the attention computation, while False will be ignored.
                # If the attn mask is completely invalid for a given query, allow it to attend everywhere
                # TODO: check and see see if this is really necessary
                if self.unmask_all_false:
                    attn_mask = torch.where(torch.all(~attn_mask, dim=-1, keepdim=True), True, attn_mask)

            if (attn_mask is not None) and self.attn_type != "flex":
                outputs[f"layer_{layer_index}"]["attn_mask"] = attn_mask
            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"],
                x["key_embed"],
                attn_mask=attn_mask,
                q_mask=x.get("query_mask"),
                kv_mask=x.get("key_valid"),
                query_posenc=x["query_posenc"] if self.posenc else None,
                key_posenc=x["key_posenc"] if self.posenc else None,
                attn_mask_transpose=attn_mask_transpose,
            )

            # update the individual input constituent representations only if not in merged input mode
            if not self.unified_decoding:
                x = unmerge_inputs(x, input_names)

        return x, outputs, metadata

    def flex_local_ca_mask(self, q_len: int, kv_len: int, device, dtype_float):
        # Calculate stride based on the ratio of key length to query length
        stride = kv_len / q_len
        if self.fast_local_ca:
            return build_strided_sliding_window_blockmask(
                window_size=self.window_size,
                block_size=self.block_size,
                stride=kv_len / q_len,
                q_len=q_len,
                kv_len=kv_len,
                device=device,
                wrap=self.window_wrap,
                dtype_float=dtype_float,
            )
        window_mask_func = sliding_window_mask_strided_wrapped if self.window_wrap else sliding_window_mask_strided
        return window_mask_func(self.window_size, stride=stride, q_len=q_len, kv_len=kv_len, device=str(device))

    def generate_positional_encodings(self, x: dict):
        idx = torch.arange(self.num_queries(x), device=x["query_embed"].device, dtype=x["query_embed"].dtype)
        x["query_phi"] = 2 * torch.pi * (idx / self.num_queries(x) - self.phi_shift)
        query_posenc = pos_enc_symmetric(x["query_phi"], self.dim, self.posenc["alpha"], self.posenc["base"])
        key_posenc = pos_enc_symmetric(x["key_phi"], self.dim, self.posenc["alpha"], self.posenc["base"])
        return query_posenc, key_posenc


class MaskFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        depth: int = 0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        bidirectional_ca: bool = True,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
    ) -> None:
        """Initialize a MaskFormer decoder layer.

        Args:
            dim: Embedding dimension.
            norm: Normalization type.
            depth: Layer depth index.
            dense_kwargs: Optional arguments for Dense layers.
            attn_kwargs: Optional arguments for Attention layers.
            bidirectional_ca: Enable bidirectional cross-attention.
            qkv_norm: Apply normalization to QKV in attention.
            hybrid_norm: Enable hybrid normalization from 2503.04598.
        """
        super().__init__()
        self.dim = dim
        self.bidirectional_ca = bidirectional_ca

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

        attn_kwargs = attn_kwargs or {}
        self.attn_type = attn_kwargs.get("attn_type", "torch")
        dense_kwargs = dense_kwargs or {}

        residual = partial(Residual, dim=dim)
        self.q_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.q_sa = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.q_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

        if self.bidirectional_ca:
            self.kv_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
            self.kv_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        attn_mask: Tensor | None = None,
        q_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        query_posenc: Tensor | None = None,
        key_posenc: Tensor | None = None,
        attn_mask_transpose: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for the decoder layer.

        Args:
            q: Query embeddings.
            kv: Key/value embeddings.
            attn_mask: Optional attention mask.
            q_mask: Optional query mask.
            kv_mask: Optional key/value mask.
            query_posenc: Optional query positional encoding.
            key_posenc: Optional key positional encoding.
            attn_mask_transpose: Optional transposed attention mask.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - The updated query embeddings (Tensor).
                - The updated key/value embeddings (Tensor).
        """
        q_pe = q if query_posenc is None else q + query_posenc
        kv_pe = kv if key_posenc is None else kv + key_posenc

        q = self.q_ca(q_pe, k=kv_pe, v=kv, attn_mask=attn_mask, q_mask=q_mask, kv_mask=kv_mask)
        q = self.q_dense(q)

        q = self.q_sa(q, k=q, v=q, q_mask=q_mask)

        # Update key/constituent embeddings with the query/object embeddings
        if self.bidirectional_ca:
            if attn_mask is not None:
                if self.attn_type == "flex":
                    assert attn_mask_transpose is not None, "attn_mask_transpose must be provided for flex attention"
                # Index from the back so we are batch shape agnostic
                attn_mask = attn_mask_transpose if attn_mask_transpose is not None else attn_mask.transpose(-2, -1)

            q_pe = q if query_posenc is None else q + query_posenc
            kv_pe = kv if key_posenc is None else kv + key_posenc

            kv = self.kv_ca(kv_pe, k=q_pe, v=q, attn_mask=attn_mask, q_mask=kv_mask, kv_mask=q_mask)
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
