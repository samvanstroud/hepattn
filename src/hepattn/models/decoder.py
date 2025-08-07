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
        self.tasks = None  # Will be set by MaskFormer
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

        outputs = {}

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            if self.preserve_posenc:
                x = self.add_positional_encodings(x)

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

            # order attn mask, query & key embeds, query mask and key valid in phi, etc.
            # TODO: would like to move this outside of layers for loop but slightly more complicated in terms of what to undo when - looking into it
            attn_mask, x, query_mask, key_valid, key_sort_idx, query_sort_idx = self.sort_by_phi(attn_mask, x, query_mask, x.get("key_valid"))
            # Log attention mask if requested
            if self.log_attn_mask:
                self.attn_mask_logging(attn_mask, layer_index)

            # Update embeddings through decoder layer
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"], x["key_embed"], attn_mask=attn_mask, q_mask=query_mask, kv_mask=key_valid
            )

            attn_mask, x, query_mask, key_valid = self.unsort_by_phi(attn_mask, x, query_mask, key_valid, key_sort_idx, query_sort_idx)

            # Unmerge the updated features back into separate input types for intermediate tasks
            for input_name in input_names:
                x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        return x, outputs

    def add_positional_encodings(self, x: dict):
        if self.query_posenc is not None:
            x["query_embed"] = x["query_embed"] + x["query_posenc"]
        if self.key_posenc is not None:
            x["key_embed"] = x["key_embed"] + x["key_posenc"]
        return x

    def generate_positional_encodings(self, x: dict):
        query_posenc = None
        key_posenc = None
        if self.query_posenc is not None:
            x["query_phi"] = 2 * torch.pi * (torch.arange(self.num_queries, device=x["query_embed"].device) / self.num_queries - 0.5)
            query_posenc = self.query_posenc(x)
        if self.key_posenc is not None:
            key_posenc = self.key_posenc(x)
        if (self.query_posenc is not None) and (self.key_posenc is not None) and (self.posenc_analysis):
            self.posenc_analysis_logging(x)
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

    def posenc_analysis_logging(self, x):
        self.last_query_phi = x["query_phi"].detach().cpu().numpy()
        self.last_query_posenc = x["query_posenc"].detach().cpu().numpy()
        key_phi = x["key_phi"].detach().cpu().numpy()
        self.last_key_phi = key_phi
        key_posenc = x["key_posenc"][0].cpu()
        self.last_key_posenc = key_posenc.numpy()
        key_sort_idx = torch.argsort(torch.tensor(key_phi), axis=-1)
        key_posencs_sorted = key_posenc[key_sort_idx[0]]
        self.last_key_posenc_sorted = key_posencs_sorted

    def sort_by_phi(self, attn_mask, x, query_mask, key_valid):
        key_phi = x.get("key_phi")
        key_sort_idx = torch.argsort(key_phi, axis=-1)

        if attn_mask is not None:
            # Sort attention mask along key dimension (dim 2) - [batch, queries, constituents]
            attn_mask = attn_mask.index_select(2, key_sort_idx[0])

        # Sort key phi for storing too
        key_phi_sorted = key_phi[0][key_sort_idx[0]]
        key_phi_sorted = key_phi_sorted.unsqueeze(0)
        x["key_phi"] = key_phi_sorted

        # Sort key posenc for storing too
        key_posenc = x.get("key_posenc")
        key_posenc_sorted = key_posenc[0][key_sort_idx[0]]
        key_posenc_sorted = key_posenc_sorted.unsqueeze(0)
        x["key_posenc"] = key_posenc_sorted

        # Sort key embeddings - [batch, constituents, dim]
        key_embed = x.get("key_embed")
        key_embed_sorted = key_embed[0][key_sort_idx[0]]  # [constituents, dim]
        key_embed_sorted = key_embed_sorted.unsqueeze(0)  # [1, constituents, dim]
        x["key_embed"] = key_embed_sorted

        # Sort key_valid mask - [batch, constituents]
        if key_valid is not None:
            key_valid_sorted = key_valid[0][key_sort_idx[0]]  # [constituents]
            key_valid_sorted = key_valid_sorted.unsqueeze(0)  # [1, constituents]
            x["key_valid"] = key_valid_sorted

        # Sort queries
        query_phi = x.get("query_phi")
        query_sort_idx = torch.argsort(query_phi, axis=-1)

        # Sort query phi for storing too
        query_phi_sorted = query_phi[0][query_sort_idx[0]]
        query_phi_sorted = query_phi_sorted.unsqueeze(0)
        x["query_phi"] = query_phi_sorted

        # Sort query posenc for storing too
        query_posenc = x.get("query_posenc")
        query_posenc_sorted = query_posenc[0][query_sort_idx[0]]
        query_posenc_sorted = query_posenc_sorted.unsqueeze(0)
        x["query_posenc"] = query_posenc_sorted

        if attn_mask is not None:
            # Sort attention mask along query dimension (dim 1) - [batch, queries, constituents]
            attn_mask = attn_mask.index_select(1, query_sort_idx.to(attn_mask.device))

        # Sort query embeddings - [batch, queries, dim]
        query_embed = x.get("query_embed")
        query_embed_sorted = query_embed[0][query_sort_idx]  # [queries, dim]
        query_embed_sorted = query_embed_sorted.unsqueeze(0)  # [1, queries, dim]
        x["query_embed"] = query_embed_sorted

        # Sort query_mask - [batch, queries]
        if query_mask is not None:
            query_mask_sorted = query_mask[0][query_sort_idx]  # [queries]
            query_mask_sorted = query_mask_sorted.unsqueeze(0)  # [1, queries]
            query_mask = query_mask_sorted

        return (
            attn_mask,
            x,
            query_mask,
            x.get("key_valid"),
            key_sort_idx,
            query_sort_idx,
        )

    def unsort_by_phi(self, attn_mask, x, query_mask, key_valid, key_sort_idx, query_sort_idx):
        # need to look at the unsorting - do I need to do it???
        # Get the original sorting indices
        # the unsort indices are the reverse of the sort indices

        # Compute unsorting indices by finding the inverse permutation
        # For a permutation p, the inverse permutation p_inv satisfies: p_inv[p[i]] = i
        key_unsort_idx = torch.argsort(key_sort_idx[0], dim=0)
        query_unsort_idx = torch.argsort(query_sort_idx[0], dim=0)

        # Unsort key phi for storing too
        key_phi_unsorted = x.get("key_phi")[0][key_unsort_idx]
        key_phi_unsorted = key_phi_unsorted.unsqueeze(0)
        x["key_phi"] = key_phi_unsorted

        # Unsort key posenc for storing too
        key_posenc_unsorted = x.get("key_posenc")[0][key_unsort_idx]
        key_posenc_unsorted = key_posenc_unsorted.unsqueeze(0)
        x["key_posenc"] = key_posenc_unsorted

        if attn_mask is not None:
            # Unsort attention mask along key dimension (dim 2) - [batch, queries, constituents]
            attn_mask = attn_mask.index_select(2, key_unsort_idx.to(attn_mask.device))

        # Unsort key embeddings - [batch, constituents, dim]
        key_embed = x.get("key_embed")
        key_embed_unsorted = key_embed[0][key_unsort_idx]  # [constituents, dim]
        key_embed_unsorted = key_embed_unsorted.unsqueeze(0)  # [1, constituents, dim]
        x["key_embed"] = key_embed_unsorted

        # Unsort key_valid mask - [batch, constituents]
        if key_valid is not None:
            key_valid_unsorted = key_valid[0][key_unsort_idx]  # [constituents]
            key_valid_unsorted = key_valid_unsorted.unsqueeze(0)  # [1, constituents]
            x["key_valid"] = key_valid_unsorted

        # Unsort query phi for storing too
        query_phi_unsorted = x.get("query_phi")[0][query_unsort_idx]
        query_phi_unsorted = query_phi_unsorted.unsqueeze(0)
        x["query_phi"] = query_phi_unsorted

        # Unsort query posenc for storing too
        query_posenc_unsorted = x.get("query_posenc")[0][query_unsort_idx]
        query_posenc_unsorted = query_posenc_unsorted.unsqueeze(0)
        x["query_posenc"] = query_posenc_unsorted

        # Unsort query posenc for storing too
        if attn_mask is not None:
            # Unsort attention mask along query dimension (dim 1) - [batch, queries, constituents]
            attn_mask = attn_mask.index_select(1, query_unsort_idx.to(attn_mask.device))

        # Unsort query embeddings - [batch, queries, dim]
        query_embed = x.get("query_embed")
        query_embed_unsorted = query_embed[0][query_unsort_idx]  # [queries, dim]
        query_embed_unsorted = query_embed_unsorted.unsqueeze(0)  # [1, queries, dim]
        x["query_embed"] = query_embed_unsorted

        # Unsort query mask - [batch, queries]
        if query_mask is not None:
            query_mask_unsorted = query_mask[0][query_unsort_idx]  # [queries]
            query_mask_unsorted = query_mask_unsorted.unsqueeze(0)  # [1, queries]
            query_mask = query_mask_unsorted

        return attn_mask, x


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
