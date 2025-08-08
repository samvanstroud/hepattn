"""Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former.
"""

import warnings
from functools import partial

import torch
from torch import Tensor, nn

from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask
from hepattn.models.transformer import Residual

N_STEPS_LOG_ATTN_MASK = 1000


class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        mask_attention: bool = True,
        use_query_masks: bool = False,
        log_attn_mask: bool = False,
        key_posenc: nn.Module | None = None,
        query_posenc: nn.Module | None = None,
        preserve_posenc: bool = False,
        posenc_analysis: bool = False,
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
        key_posenc : nn.Module | None, optional
            The positional encoding module for the key embeddings.
        query_posenc : nn.Module | None, optional
            The positional encoding module for the query embeddings.
        preserve_posenc : bool, optional
            If True, the positional encodings will be preserved.
        posenc_analysis : bool, optional
            If True, the positional encoding analysis will be performed.
        """
        super().__init__()

        # Ensure mask_attention is passed to decoder layers
        decoder_layer_config = decoder_layer_config.copy()
        decoder_layer_config["mask_attention"] = mask_attention

        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.tasks: list = []  # Will be set by MaskFormer
        self.num_queries = num_queries
        self.mask_attention = mask_attention
        self.use_query_masks = use_query_masks
        self.log_attn_mask = log_attn_mask
        self.key_posenc = key_posenc
        self.query_posenc = query_posenc
        self.preserve_posenc = preserve_posenc
        self.posenc_analysis = posenc_analysis
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

        if (self.key_posenc is not None) or (self.query_posenc is not None):
            x["query_posenc"], x["key_posenc"] = self.generate_positional_encodings(x)
        if not self.preserve_posenc:
            x["query_embed"], x["key_embed"] = self.add_positional_encodings(x)

        sort_indices = self.get_sort_indices(x)

        outputs: dict[str, dict] = {}

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            if self.preserve_posenc:
                x["query_embed"], x["key_embed"] = self.add_positional_encodings(x)

            attn_masks: dict[str, torch.Tensor] = {}
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
                        x["query_mask"] = query_mask

            # Construct the full attention mask for MaskAttention decoder
            attn_mask = None
            if attn_masks and self.mask_attention:
                attn_mask = torch.full((batch_size, self.num_queries, num_constituents), True, device=x["key_embed"].device)
                for input_name, task_attn_mask in attn_masks.items():
                    attn_mask[..., x[f"key_is_{input_name}"]] = task_attn_mask

            # do sorting
            # TODO: would like to move this outside of layers for loop but slightly more complicated in terms of what to undo when
            # - would need to sort and unsort more vars and easy to make errors or miss something? could change?
            # TODO: would also be nice if we sorted before the encoder and then unsorted after producing decoder layer?

            attn_mask = self.sort_attn_mask_by_phi(
                attn_mask, sort_indices["key"], sort_indices["query"]
            )
            # can't use sorted embeds from earlier because we update embeddings in between
            for input_key, sort_key in zip(
                ["query_embed", "key_embed", "query_mask", "key_valid"],
                ["query", "key", "query", "key"],
                strict=True,
            ):
                if input_key in x:
                    x[input_key] = self.sort_var_by_phi(x[input_key], sort_indices[sort_key])
                else:
                    warnings.warn(f"Variable {input_key} not found in x - skipping sorting for this variable")

            # Log attention mask if requested
            if self.log_attn_mask:
                self.attn_mask_logging(attn_mask, layer_index)

            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"],
                x["key_embed"],
                attn_mask=attn_mask,
                q_mask=x.get("query_mask"),
                kv_mask=x.get("key_valid"),
            )

            # only need to unsort embeds because these are the only ones that are passed on - if pass on other sorted vars would need to change this
            for input_key, sort_key in zip(
                ["query_embed", "key_embed"],
                ["query", "key"],
                strict=True,
            ):
                x[input_key] = self.sort_var_by_phi(
                    x[input_key], self.get_unsort_idx(sort_indices[sort_key])
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
        if (self.query_posenc is not None) and (self.key_posenc is not None) and (self.posenc_analysis):
            self.posenc_analysis_logging(x, query_posenc, key_posenc)
        return query_posenc, key_posenc

    def attn_mask_logging(self, attn_mask, layer_index):
        if ((attn_mask is not None) and (self.log_step % N_STEPS_LOG_ATTN_MASK == 0)) or (not self.training):
            if not hasattr(self, "attn_masks_to_log"):
                self.attn_masks_to_log = {}
            if layer_index == 0 or layer_index == len(self.decoder_layers) - 1:
                attn_mask_im = attn_mask[0].detach().cpu().clone().int()
                self.attn_masks_to_log[layer_index] = {
                    "mask": attn_mask_im,
                    "step": self.log_step,
                    "layer": layer_index,
                }

    def posenc_analysis_logging(self, x, query_posenc, key_posenc):
        if query_posenc is not None:
            self.last_query_posenc = query_posenc.detach().cpu().numpy()
            self.last_query_phi = x["query_phi"].detach().cpu().numpy()
        if key_posenc is not None:
            key_phi = x["key_phi"].detach().cpu().numpy()
            self.last_key_phi = key_phi
            key_posenc = key_posenc[0].cpu()
            self.last_key_posenc = key_posenc.numpy()
            key_sort_idx = torch.argsort(torch.tensor(key_phi), axis=-1)
            key_posencs_sorted = key_posenc[key_sort_idx[0]]
            self.last_key_posenc_sorted = key_posencs_sorted

    def sort_attn_mask_by_phi(self, attn_mask, key_sort_idx, query_sort_idx):
        if len(key_sort_idx.shape) == 2:
            key_sort_idx = key_sort_idx[0]
        assert len(key_sort_idx.shape) == 1, "Key sort index must be 1D"
        if len(query_sort_idx.shape) == 2:
            query_sort_idx = query_sort_idx[0]
        assert len(query_sort_idx.shape) == 1, "Query sort index must be 1D"

        if attn_mask is not None:
            attn_mask = attn_mask.index_select(2, key_sort_idx.to(attn_mask.device))
            attn_mask = attn_mask.index_select(1, query_sort_idx.to(attn_mask.device))
        return attn_mask

    def sort_var_by_phi(self, var, sort_idx):
        if len(sort_idx.shape) == 2:
            sort_idx = sort_idx[0]
        assert len(sort_idx.shape) == 1, "Sort index must be 1D"

        if var is not None:
            if len(var.shape) == 2:
                var_sorted = var[0][sort_idx]
                var_sorted = var_sorted.unsqueeze(0)  # Preserve batch dimension
            elif len(var.shape) == 1:
                var_sorted = var[sort_idx]
            else:
                raise ValueError(f"Variable has invalid shape: {var.shape}")
        else:
            var_sorted = None
        return var_sorted

    def get_sort_indices(self, x):
        sort_indices = {}
        sort_indices["key"] = self.get_sort_idx(x.get("key_phi"))
        sort_indices["query"] = self.get_sort_idx(x.get("query_phi"))
        return sort_indices

    def get_sort_idx(self, phi: Tensor) -> Tensor:
        if len(phi.shape) == 2:
            phi = phi[0]
        assert len(phi.shape) == 1, "Phi must be 1D"
        return torch.argsort(phi)

    def get_unsort_idx(self, sort_idx: Tensor) -> Tensor:
        if len(sort_idx.shape) == 2:
            sort_idx = sort_idx[0]
        assert len(sort_idx.shape) == 1, "Sort index must be 1D"
        return torch.argsort(sort_idx, dim=0)


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
            attn_mask = torch.where(torch.all(~attn_mask, dim=-1, keepdim=True), True, attn_mask)
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
