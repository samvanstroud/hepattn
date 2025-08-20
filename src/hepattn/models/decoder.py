"""Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former.
"""

from functools import partial

import torch
from torch import Tensor, nn

from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.encoder import Residual
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask
from hepattn.utils.local_ca import auto_local_ca_mask


class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        mask_attention: bool = True,
        use_query_masks: bool = False,
        key_posenc: nn.Module | None = None,
        query_posenc: nn.Module | None = None,
        preserve_posenc: bool = False,
        local_strided_attn: bool = False,
        window_size: int = 512,
        window_wrap: bool = True,
    ):
        """MaskFormer decoder that handles multiple decoder layers and task integration.

        Args:
            num_queries: The number of object-level queries.
            decoder_layer_config: Configuration dictionary used to initialize each MaskFormerDecoderLayer.
            num_decoder_layers: The number of decoder layers to stack.
            mask_attention: If True, attention masks will be used to control which input constituents are attended to.
            use_query_masks: If True, predicted query masks will be used to control which queries are valid.
            key_posenc: Optional module for key positional encoding.
            query_posenc: Optional module for query positional encoding.
            preserve_posenc: If True, preserves positional encoding in embeddings.
            local_strided_attn: If True, uses local strided window attention.
            window_size: The size of the window for local strided window attention.
            window_wrap: If True, wraps the window for local strided window attention.
        """
        super().__init__()

        # Ensure mask_attention is passed to decoder layers
        decoder_layer_config = decoder_layer_config.copy()
        decoder_layer_config["mask_attention"] = mask_attention

        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.tasks: list | None = None  # Will be set by MaskFormer
        self.num_queries = num_queries
        self.mask_attention = mask_attention
        self.use_query_masks = use_query_masks
        self.key_posenc = key_posenc
        self.query_posenc = query_posenc
        self.preserve_posenc = preserve_posenc
        self.local_strided_attn = local_strided_attn
        self.attn_type = decoder_layer_config.get("attn_type", "torch")
        self.window_size = window_size
        self.window_wrap = window_wrap
        if self.local_strided_attn:
            assert self.attn_type == "torch", f"Invalid attention type when local_strided_attn is True: {self.attn_type}, must be 'torch'"
        assert not (self.local_strided_attn and self.mask_attention), "local_strided_attn and mask_attention cannot both be True"

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        """Forward pass through decoder layers.

        Args:
            x: Dictionary containing embeddings and masks.
            input_names: List of input names for constructing attention masks.

        Returns:
            Tuple containing updated embeddings and outputs from each decoder layer and final outputs.
        """
        batch_size = x["query_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]

        if (self.key_posenc is not None) or (self.query_posenc is not None):
            x["query_posenc"], x["key_posenc"] = self.generate_positional_encodings(x)
        if not self.preserve_posenc:
            x["query_embed"], x["key_embed"] = self.add_positional_encodings(x)

        attn_mask = None
        if self.local_strided_attn:
            assert x["query_embed"].shape[0] == 1, "Local strided attention only supports batch size 1"
            attn_mask = auto_local_ca_mask(x["query_embed"], x["key_embed"], self.window_size, wrap=self.window_wrap)

        outputs: dict[str, dict] = {}
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            if self.preserve_posenc:
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

                # Collect attention masks from tasks
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
                attn_mask = torch.full((batch_size, self.num_queries, num_constituents), True, device=x["key_embed"].device)
                for input_name, task_attn_mask in attn_masks.items():
                    attn_mask[..., x[f"key_is_{input_name}"]] = task_attn_mask

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

            # Unmerge the updated features back into separate input types for intermediate tasks
            for input_name in input_names:
                x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        return x, outputs

    def add_positional_encodings(self, x: dict):
        if self.query_posenc is not None:
            x["query_embed"] = x["query_embed"] + x["query_posenc"]
        if self.key_posenc is not None:
            x["key_embed"] = x["key_embed"] + x["key_posenc"]
        return x["query_embed"], x["key_embed"]

    def generate_positional_encodings(self, x: dict):
        query_posenc = None
        key_posenc = None
        if self.query_posenc is not None:
            x["query_phi"] = 2 * torch.pi * (torch.arange(self.num_queries, device=x["query_embed"].device) / self.num_queries - 0.5)
            query_posenc = self.query_posenc(x)
        if self.key_posenc is not None:
            key_posenc = self.key_posenc(x)
        return query_posenc, key_posenc


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
