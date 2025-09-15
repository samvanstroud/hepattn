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
        fast_local_ca: bool = False,
        block_size: int = 128,
        unified_decoding: bool = False,
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
        """
        super().__init__()

        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.dim = decoder_layer_config["dim"]
        self.tasks: list | None = None  # Will be set by MaskFormer
        self.num_queries = num_queries
        self.mask_attention = mask_attention
        self.use_query_masks = use_query_masks
        self.posenc = posenc
        self.local_strided_attn = local_strided_attn
        self.attn_type = decoder_layer_config.get("attn_kwargs", {}).get("attn_type", "torch")
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.unified_decoding = unified_decoding
        self.initial_queries = nn.Parameter(torch.randn(self.num_queries, decoder_layer_config["dim"]))
        self.fast_local_ca = fast_local_ca
        self.block_size = block_size

        if self.local_strided_attn:
            assert self.attn_type in {"torch", "flex"}, (
                f"Invalid attention type when local_strided_attn is True: {self.attn_type}, must be 'torch' or 'flex'"
            )
        assert not (self.local_strided_attn and self.mask_attention), "local_strided_attn and mask_attention cannot both be True"

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        """Forward pass through decoder layers.

        Args:
            x: Dictionary containing embeddings and masks.
            input_names: List of input names for constructing attention masks.

        Returns:
            Tuple containing updated embeddings and outputs from each decoder layer and final outputs.

        Raises:
            ValueError: If in merged input mode and multiple attention masks are provided.
        """
        batch_size = x["key_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]

        # Generate the queries that represent objects
        x["query_embed"] = self.initial_queries.expand(batch_size, -1, -1)
        x["query_valid"] = torch.full((batch_size, self.num_queries), True, device=x["query_embed"].device)

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
                for input_name, task_attn_mask in task_attn_masks.items():
                    if input_name in attn_masks:
                        attn_masks[input_name] |= task_attn_mask
                    else:
                        attn_masks[input_name] = task_attn_mask

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
                    attn_mask = torch.full((batch_size, self.num_queries, num_constituents), False, device=x["key_embed"].device)
                    for input_name, task_attn_mask in attn_masks.items():
                        attn_mask[x[f"key_is_{input_name}"].unsqueeze(1).expand_as(attn_mask)] = task_attn_mask.flatten()

                attn_mask = attn_mask.detach()
                # True values indicate a slot will be included in the attention computation, while False will be ignored.
                # If the attn mask is completely invalid for a given query, allow it to attend everywhere
                # TODO: check and see see if this is really necessary
                attn_mask = torch.where(torch.all(~attn_mask, dim=-1, keepdim=True), True, attn_mask)

            if attn_mask is not None and self.attn_type != "flex":
                outputs[f"layer_{layer_index}"]["attn_mask"] = attn_mask

            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"],
                x["key_embed"],
                attn_mask=attn_mask,
                q_mask=x.get("query_mask"),
                kv_mask=x.get("key_valid"),
                query_posenc=x["query_posenc"] if (self.posenc and not self.mask_attention) else None,
                key_posenc=x["key_posenc"] if (self.posenc and not self.mask_attention) else None,
                attn_mask_transpose=attn_mask_transpose,
            )

            # update the individual input constituent representations only if not in merged input mode
            if not self.unified_decoding:
                x = unmerge_inputs(x, input_names)

        return x, outputs

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
        x["query_phi"] = 2 * torch.pi * torch.arange(self.num_queries, device=x["query_embed"].device) / self.num_queries
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
        hybrid_norm: bool = False,
    ) -> None:
        """Initialize a MaskFormer decoder layer.

        Args:
            dim: Embedding dimension.
            norm: Normalization type.
            depth: Layer depth index.
            dense_kwargs: Optional arguments for Dense layers.
            attn_kwargs: Optional arguments for Attention layers.
            bidirectional_ca: If True, enables bidirectional cross-attention.
            hybrid_norm: If True, enables hybrid normalization.
        """
        super().__init__()

        self.dim = dim
        self.bidirectional_ca = bidirectional_ca

        # handle hybridnorm
        qkv_norm = hybrid_norm
        if depth == 0:
            hybrid_norm = False
        attn_norm = norm if not hybrid_norm else None
        dense_post_norm = not hybrid_norm

        attn_kwargs = attn_kwargs or {}
        self.attn_type = attn_kwargs.get("attn_type", "torch")
        dense_kwargs = dense_kwargs or {}

        residual = partial(Residual, dim=dim, norm=norm)
        self.q_ca = residual(Attention(dim, qkv_norm=qkv_norm, **attn_kwargs), norm=attn_norm)
        self.q_sa = residual(Attention(dim, qkv_norm=qkv_norm, **attn_kwargs), norm=attn_norm)
        self.q_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

        if self.bidirectional_ca:
            self.kv_ca = residual(Attention(dim, qkv_norm=qkv_norm, **attn_kwargs), norm=attn_norm)
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
            Tuple of updated query and key/value embeddings.
        """
        if query_posenc is not None:
            q = q + query_posenc
        if key_posenc is not None:
            kv = kv + key_posenc

        # Update query/object embeddings with the key/constituent embeddings
        q = self.q_ca(q, kv=kv, attn_mask=attn_mask, q_mask=q_mask, kv_mask=kv_mask)
        q = self.q_sa(q, q_mask=q_mask)
        q = self.q_dense(q)

        # Update key/constituent embeddings with the query/object embeddings
        if self.bidirectional_ca:
            if attn_mask is not None:
                if self.attn_type == "flex":
                    assert attn_mask_transpose is not None, "attn_mask_transpose must be provided for flex attention"
                # Index from the back so we are batch shape agnostic
                attn_mask = attn_mask_transpose if attn_mask_transpose is not None else attn_mask.transpose(-2, -1)

            if query_posenc is not None:
                q = q + query_posenc
            if key_posenc is not None:
                kv = kv + key_posenc

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
