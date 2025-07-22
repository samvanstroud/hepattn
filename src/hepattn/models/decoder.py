"""Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former.
"""

from functools import partial

import torch
from torch import Tensor, nn

from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask
from hepattn.models.transformer import Residual


class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        mask_attention: bool = True,
        use_query_masks: bool = False,
        log_attn_mask: bool = False,
        query_posenc: nn.Module | None = None,
        preserve_posenc: bool = False,
    ):
        """MaskFormer decoder that handles multiple decoder layers and task integration.

        Parameters
        ----------
        num_queries : int
            The number of object-level queries.
        decoder_layer_config : dict
            Configuration dictionary used to initialize each MaskFormerDecoderLayer.
        num_decoder_layers : int
            The number of decoder layers to stack.
        mask_attention : bool, optional
            If True, attention masks will be used to control which input objects are attended to.
        use_query_masks : bool, optional
            If True, predicted query masks will be used to control which queries are valid.
            May be useful when providing initial queries as inputs.
        log_attn_mask : bool, optional
            If True, log attention masks for debugging.
        """
        super().__init__()

        # Ensure mask_attention is passed to decoder layers
        decoder_layer_config = decoder_layer_config.copy()
        decoder_layer_config["mask_attention"] = mask_attention

        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.tasks = None  # Will be set by MaskFormer
        self.num_queries = num_queries
        self.mask_attention = mask_attention
        self.use_query_masks = use_query_masks
        self.log_attn_mask = log_attn_mask
        self.query_posenc = query_posenc
        self.preserve_posenc = preserve_posenc
        self.log_step = 0

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        """Forward pass through decoder layers.

        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing embeddings and masks.
        input_names : list[str]
            List of input names for constructing attention masks.

        Returns:
        -------
        dict[str, dict]
            Outputs from each decoder layer and final outputs.
        """
        batch_size = x["query_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]
        self.log_step += 1

        outputs = {}

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            attn_masks = {}
            query_mask = None

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

            # Construct the full attention mask for MaskAttention decoder
            attn_mask = None
            if attn_masks and self.mask_attention:
                attn_mask = torch.full((batch_size, self.num_queries, num_constituents), True, device=x["key_embed"].device)
                for input_name, task_attn_mask in attn_masks.items():
                    attn_mask[..., x[f"key_is_{input_name}"]] = task_attn_mask

            # Log attention mask if requested
            if self.log_attn_mask and (attn_mask is not None) and (self.log_step % 1000 == 0):
                if not hasattr(self, "attn_masks_to_log"):
                    self.attn_masks_to_log = {}
                if layer_index == 0 or layer_index == len(self.decoder_layers) - 1:
                    self.attn_masks_to_log[layer_index] = {
                        "mask": attn_mask[0].detach().cpu().clone(),
                        "step": self.log_step,
                        "layer": layer_index,
                    }

            # Add query positional encodings
            x = self.add_query_posenc(x)

            # Update embeddings through decoder layer
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"], x["key_embed"], attn_mask=attn_mask, q_mask=query_mask, kv_mask=x.get("key_valid")
            )

            # Re-add original embeddings (similar to SAM's prompt token re-addition)
            x = self.re_add_original_embeddings(x)

            # Unmerge the updated features back into separate input types for intermediate tasks
            for input_name in input_names:
                x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        return x, outputs

    def re_add_original_embeddings(self, x: dict):
        # Re-add original query embeddings (similar to SAM's prompt token re-addition)
        if self.preserve_posenc:
            x["key_embed"] = x["key_embed"] + x["key_posenc"]
            if self.query_posenc is not None:
                x["query_embed"] = x["query_embed"] + x["query_posenc"]
        return x

    def add_query_posenc(self, x: dict):
        if self.query_posenc is not None:
            # The query positional encoding is static, so we compute it once and cache it in `x`.
            if "query_posenc" not in x:
                x["query_phi"] = 2 * torch.pi * (torch.arange(self.num_queries, device=x["query_embed"].device) / self.num_queries - 0.5)
                x["query_posenc"] = self.query_posenc(x)
            x["query_embed"] = x["query_embed"] + x["query_posenc"]
        return x


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
        if self.mask_attention:
            assert attn_mask is not None, "attn_mask must be provided for mask attention"
            attn_mask = attn_mask.detach()
            # True values indicate a slot will be included in the attention computation, while False will be ignored.
            # If the attn mask is completely invalid for a given query, allow it to attend everywhere
            attn_mask[torch.where(attn_mask.sum(-1) == 0)] = True
        else:
            attn_mask = None

        # Update query/object embeddings with the key/hit embeddings
        q = self.q_ca(q, kv=kv, attn_mask=attn_mask, q_mask=q_mask, kv_mask=kv_mask)
        q = self.q_sa(q, q_mask=q_mask)
        q = self.q_dense(q)

        # Update key/hit embeddings with the query/object embeddings
        if self.bidirectional_ca:
            if attn_mask is not None:
                # Index from the back so we are batch shape agnostic
                attn_mask = attn_mask.transpose(-2, -1)

            kv = self.kv_ca(kv, kv=q, attn_mask=attn_mask, q_mask=kv_mask, kv_mask=q_mask)
            kv = self.kv_dense(kv)

        return q, kv

    def set_backend(self, attn_type: str) -> None:
        """Set the backend for the attention layers.
        This is useful for switching between different attention implementations.
        """
        self.q_ca.fn.set_backend(attn_type)
        self.q_sa.fn.set_backend(attn_type)

        if self.bidirectional_ca:
            self.kv_ca.fn.set_backend(attn_type)
