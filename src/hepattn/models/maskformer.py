from typing import Any

import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.utils.model_utils import unmerge_inputs


class MaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        decoder: MaskFormerDecoder,
        tasks: nn.ModuleList,
        dim: int,
        target_object: str = "particle",
        pooling: nn.Module | None = None,
        matcher: nn.Module | None = None,
        input_sort_field: str | None = None,
        sorter: nn.Module | None = None,
        unified_decoding: bool = False,
        dynamic_query_source: str = "hit",
        encoder_tasks: nn.ModuleList | None = None,
    ):
        """Initializes the MaskFormer model, which is a modular transformer-style architecture designed
        for multi-task object reconstruction with attention-based decoding and optional encoder blocks.

        Args:
            input_nets: A list of input modules, each responsible for embedding a specific constituent type.
            encoder: An optional encoder module that processes merged constituent embeddings with optional sorting.
            decoder: The decoder module that handles multi-layer decoding and task integration.
            tasks: A list of task modules, each responsible for producing and processing predictions from decoder outputs.
            dim: The dimensionality of the query and key embeddings.
            target_object: The target object name which is used to mark valid/invalid objects during matching.
            pooling: An optional pooling module used to aggregate features from the input constituents.
            matcher: A module used to match predictions to targets (e.g., using the Hungarian algorithm) for loss computation.
            input_sort_field: An optional key used to sort the input constituents (e.g., for windowed attention).
            sorter: An optional sorter module used to reorder input constituents before processing.
            unified_decoding: If True, inputs remain merged for task processing instead of being unmerged after encoding.
            dynamic_query_source: Name of the input type to use as the source for dynamic query initialization (default: "hit").
            encoder_tasks: Optional list of tasks to run after the encoder (before decoder). These tasks operate on post-encoder features.
        """
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.tasks = tasks
        self.encoder_tasks = encoder_tasks or nn.ModuleList()
        self.decoder.encoder_tasks = self.encoder_tasks
        self.pooling = pooling
        self.tasks = tasks
        self.target_object = target_object
        self.matcher = matcher
        self.unified_decoding = unified_decoding
        self.decoder.unified_decoding = unified_decoding
        self.dynamic_query_source = dynamic_query_source
        self.decoder.dynamic_query_source = dynamic_query_source

        assert not (input_sort_field and sorter), "Cannot specify both input_sort_field and sorter."
        self.input_sort_field = input_sort_field
        self.sorter = sorter
        if self.sorter is not None:
            self.sorter.input_names = self.input_names

        assert "key" not in self.input_names, "'key' input name is reserved."
        assert "query" not in self.input_names, "'query' input name is reserved."
        assert not any("_" in name for name in self.input_names), "Input names cannot contain underscores."

    @property
    def input_names(self) -> list[str]:
        return [input_net.input_name for input_net in self.input_nets]

    @staticmethod
    def build_dynamic_targets(
        selected_indices: Tensor, targets: dict[str, Tensor], source_name: str, target_name: str
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Filter out target objects which do not have any constituents selected as queries.

        The idea is that if we don't select a constituent from a given target object to 
        be an initial query, we should not try to predict that target object at all.

        Note: Only batch size 1 is supported for dynamic queries.

        Args:
            selected_indices: (N_queries,) indices of constituents selected as queries
            targets: dict with constituent_target_id (1, N_constituents), target_id (1, N_targets), etc.
            source_name: Name of the query source (e.g., "hit")
            target_name: Name of the target object (e.g., "particle")

        Returns:
            Tuple of:
                - filtered_constituent_mask: (1, N_queries, N_constituents) mask indicating which constituents belong to each query's target
                - first_occurrence: (1, N_queries) mask indicating which queries are the first of their target
                - filtered_target_valid: (1, N_queries) mask indicating which query slots are valid
                - filtered_target_idx: (1, N_queries) mapping from query index to original target index
        """
        # extract info and remove batch dimension
        source_target_id_key = f"{source_name}_{target_name}_id"
        target_id_key = f"{target_name}_id"
        constituent_target_id = targets[source_target_id_key][0]  # (N_constituents,)
        target_id = targets[target_id_key][0]  # (N_targets,)

        # Get the target ID for each selected constituent/query
        filtered_target_id = constituent_target_id[selected_indices]  # (N_queries,)

        # Create mapping from query index to target index
        # Find which target index corresponds to each query's target_id
        # Shape: (N_queries, N_targets) where each row has True at matching target indices
        # Detach necessary to avoid gradient retention on large comparison tensor
        matches_mask = (filtered_target_id.unsqueeze(1) == target_id.unsqueeze(0)).detach()
        # Get first matching target index for each query (argmax returns 0 if no match, which is ok)
        filtered_target_idx = matches_mask.long().argmax(dim=1)

        # Handle duplicates: only keep FIRST occurrence of each target
        # This ensures multiple selected constituents from same target don't create duplicate targets
        # Vectorized approach: sort by inverse_indices and find boundaries
        _, inverse_indices = torch.unique(filtered_target_id, return_inverse=True, sorted=False)
        sorted_inverse, sorted_idx = torch.sort(inverse_indices, stable=True)
        # Identify positions where inverse_index changes (first occurrences in sorted order)
        is_first_sorted = torch.cat([torch.tensor([True], device=inverse_indices.device), sorted_inverse[1:] != sorted_inverse[:-1]])
        # Map back to original positions
        first_occurrence = torch.zeros_like(filtered_target_id, dtype=torch.bool)
        first_occurrence[sorted_idx[is_first_sorted]] = True

        # Build the filtered target mask using only filtered particles
        # filtered_target_id: (N_queries,), constituent_target_id: (N_constituents,)
        # Shape: (N_queries, N_constituents)
        filtered_constituent_mask = filtered_target_id.unsqueeze(-1) == constituent_target_id.unsqueeze(0)

        # Zero out duplicate query rows (only first constituent per target gets targets)
        filtered_constituent_mask[~first_occurrence] = False

        # Create validity mask: only first occurrence queries are valid
        filtered_target_valid = first_occurrence

        # Add back the batch dimension
        filtered_constituent_mask = filtered_constituent_mask.unsqueeze(0)  # (1, N_queries, N_constituents)
        first_occurrence = first_occurrence.unsqueeze(0)  # (1, N_queries)
        filtered_target_valid = filtered_target_valid.unsqueeze(0)  # (1, N_queries)
        filtered_target_idx = filtered_target_idx.unsqueeze(0)  # (1, N_queries)

        return filtered_constituent_mask, first_occurrence, filtered_target_valid, filtered_target_idx

    @staticmethod
    def _align_outputs_to_original_targets(outputs: dict, query_target_idx: Tensor, num_original_targets: int) -> dict:
        """Align query-sized outputs from dynamic queries to original target dimension.

        Uses scatter_reduce with 'amax' to handle duplicate queries mapping to the same target.
        This ensures that valid (non-zero) predictions are not overwritten by zeroed-out
        duplicate query values.

        Node: this realignment uses truth info which needs to be fixed. Because during inference
        it's not possible to deduplicate targets in the way that is done here

        Args:
            outputs: Outputs dict with structure {layer: {task: {field: tensor}}}
            query_target_idx: (B, N_queries) mapping from query index to target index
            num_original_targets: Number of targets in original input

        Returns:
            New outputs dict with tensors expanded to original target dimension
        """
        batch_size = query_target_idx.shape[0]
        num_queries = query_target_idx.shape[1]
        device = query_target_idx.device
        idx = query_target_idx.detach().long() # this is truth info!

        def scatter_align(value: Tensor) -> Tensor:
            """Scatter query-sized tensor to original target dimension using amax reduction."""
            original_dtype = value.dtype
            value = value.detach().float()

            # Build output shape: replace query dim with num_original_targets
            out_shape = list(value.shape)
            out_shape[1] = num_original_targets
            aligned = torch.zeros(out_shape, dtype=torch.float32, device=device)

            # Expand index to match value shape for scatter_reduce
            idx_expanded = idx.view(batch_size, num_queries, *([1] * (value.dim() - 2))).expand_as(value)
            # This op is problematic as it uses truth indices to deduplcate queries if they are selected
            # from the same original target. This cannot be done during inference time as we do not have access to truth.
            aligned.scatter_reduce_(dim=1, index=idx_expanded, src=value, reduce="amax", include_self=True)

            # Convert back to original dtype
            if original_dtype == torch.bool:
                return aligned > 0.5
            return aligned.to(original_dtype)

        aligned_outputs = {}
        for layer_name, layer_outputs in outputs.items():
            if layer_name == "encoder":
                aligned_outputs[layer_name] = layer_outputs
                continue

            aligned_layer = layer_outputs.copy()
            for key, val in layer_outputs.items():
                if key.startswith("_"):
                    continue

                if isinstance(val, Tensor) and val.shape[1] == num_queries:
                    aligned_layer[key] = scatter_align(val)
                elif isinstance(val, dict):
                    aligned_task = val.copy()
                    for field, field_val in val.items():
                        if isinstance(field_val, Tensor) and field_val.shape[1] == num_queries:
                            aligned_task[field] = scatter_align(field_val)
                    aligned_layer[key] = aligned_task

            aligned_outputs[layer_name] = aligned_layer

        return aligned_outputs

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, dict[str, dict[str, Tensor]]]:
        batch_size = inputs[self.input_names[0] + "_valid"].shape[0]
        x = {"inputs": inputs}

        # Track per-input slices into the merged key tensor.
        # This is only used to keep dynamic_queries compatible with unified_decoding.
        key_slices: dict[str, slice] = {}
        key_start = 0

        # Embed the input constituents
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]

            n_objects = x[input_name + "_embed"].shape[-2]
            key_slices[input_name] = slice(key_start, key_start + n_objects)
            key_start += n_objects

            # These slices can be used to pick out specific
            # objects after we have merged them all together
            # Only needed when not doing unified decoding
            if not self.unified_decoding:
                device = inputs[input_name + "_valid"].device
                mask = torch.cat([torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device) for i in self.input_names], dim=-1)
                x[f"key_is_{input_name}"] = mask.unsqueeze(0).expand(batch_size, -1)

        # Merge the input constituents and the padding mask into a single set
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in self.input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in self.input_names], dim=-1)
        # Preserve a non-None version for downstream logic that expects a tensor mask.
        x["key_valid_full"] = x["key_valid"]

        # If all key_valid are true, then we can just set it to None, however,
        # if we are using flash-varlen, we have to always provide a kv_mask argument
        if batch_size == 1 and x["key_valid"].all() and self.encoder.attn_type != "flash-varlen":
            x["key_valid"] = None

        # LEGACY. TODO: remove
        if self.input_sort_field and not self.sorter:
            x[f"key_{self.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.input_sort_field] for input_name in self.input_names], dim=-1
            )

        # Dedicated sorting step before encoder
        if self.sorter is not None:
            x[f"key_{self.sorter.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.sorter.input_sort_field] for input_name in self.input_names], dim=-1
            )
            for input_name in self.input_names:
                field = f"{input_name}_{self.sorter.input_sort_field}"
                x[field] = inputs[field]
            x = self.sorter.sort_inputs(x)

        # Pass merged input constituents through the encoder
        x_sort_value = x.get(f"key_{self.input_sort_field}") if self.sorter is None else None
        x["key_embed"] = self.encoder(x["key_embed"], x_sort_value=x_sort_value, kv_mask=x.get("key_valid"))

        # Keep dynamic query initialization compatible with unified decoding by ensuring
        # source_embed/source_valid refer to *post-encoder* features.
        if self.decoder.dynamic_queries and self.unified_decoding:
            if self.sorter is not None:
                raise ValueError("dynamic_queries with unified_decoding is not supported when sorter is enabled")
            if self.dynamic_query_source not in key_slices:
                raise ValueError(f"dynamic_queries=True requires an input named '{self.dynamic_query_source}'")
            source_slice = key_slices[self.dynamic_query_source]
            x[f"{self.dynamic_query_source}_embed"] = x["key_embed"][:, source_slice, :]
            x[f"{self.dynamic_query_source}_valid"] = x["key_valid_full"][:, source_slice]

        # Unmerge the updated features back into the separate input types only if not doing unified decoding
        if not self.unified_decoding:
            x = unmerge_inputs(x, self.input_names)

        # Run encoder tasks
        outputs = {"encoder": {}}
        for task in self.encoder_tasks:
            outputs["encoder"][task.name] = task(x)

        # Pass through decoder layers
        x, decoder_outputs = self.decoder(x, self.input_names)
        outputs["encoder"].update(decoder_outputs.pop("encoder", {}))
        outputs.update(decoder_outputs)

        # Do any pooling if desired
        if self.pooling is not None:
            x_pooled = self.pooling(x[f"{self.pooling.input_name}_embed"], x[f"{self.pooling.input_name}_valid"])
            x[f"{self.pooling.output_name}_embed"] = x_pooled

        # Get the final outputs
        outputs["final"] = {}
        for task in self.tasks:
            # Pass outputs dict so tasks can read from previously executed tasks
            outputs["final"][task.name] = task(x, outputs=outputs["final"])

        # store info about the input sort field for each input type
        if self.sorter is not None:
            sort = self.sorter.input_sort_field
            sort_dict = {f"{name}_{sort}": inputs[f"{name}_{sort}"] for name in self.input_names}
            outputs["final"][sort] = sort_dict

        return outputs

    def predict(self, outputs: dict) -> dict:
        """Takes the raw model outputs and produces a set of actual inferences / predictions.
        For example will take output probabilies and apply threshold cuts to prduce boolean predictions.

        Args:
            outputs: The outputs produced by the forward pass of the model.

        Returns:
            preds: A dictionary containing the predicted values for each task.
        """
        preds: dict[str, dict[str, Any]] = {}

        # Compute predictions for each task in each block
        for layer_name, layer_outputs in outputs.items():
            if layer_name.startswith("_"):
                continue

            preds[layer_name] = {}

            # Handle encoder tasks
            if layer_name == "encoder":
                for task in self.encoder_tasks:
                    if task.name not in layer_outputs:
                        continue
                    preds[layer_name][task.name] = task.predict(layer_outputs[task.name])

            # Handle decoder tasks
            else:
                for task in self.tasks:
                    if task.name not in layer_outputs:
                        continue
                    preds[layer_name][task.name] = task.predict(layer_outputs[task.name])

        # If realignment information is available, align predictions to original target dimension.
        # This occurs when dynamic queries are active and targets were available during loss().
        if "encoder" in outputs and "query_target_idx" in outputs["encoder"]:
            enc_meta = outputs["encoder"]
            preds = self._align_outputs_to_original_targets(preds, enc_meta["query_target_idx"], enc_meta["num_original_targets"])

        return preds

    def _prepare_targets_and_outputs(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict, dict]:
        """Prepare targets and separate encoder/decoder outputs.

        For dynamic queries, this creates a separate loss_targets dict with the filtered targets,
        while keeping the original targets untouched (with additional fields for
        metric computation). For static queries, loss_targets is the same as targets.

        Args:
            outputs: The outputs produced by the forward pass of the model.
            targets: The data containing the targets (not mutated, but enriched for metrics).

        Returns:
            Tuple of (targets, loss_targets, encoder_outputs, decoder_outputs) where:
            - targets: Original targets dict, enriched with query_target_idx for metrics
            - loss_targets: Targets dict to use for loss computation. For dynamic queries,
              this is a subset of the original targets corresponding to the selected queries.
            - encoder_outputs: Separated encoder outputs
            - decoder_outputs: Separated decoder layer outputs
        """
        # Separate encoder and decoder outputs for cleaner logic
        encoder_outputs = {"encoder": outputs["encoder"]} if "encoder" in outputs else {}
        decoder_outputs = {k: v for k, v in outputs.items() if k != "encoder"}

        # For static queries, use targets directly for loss computation
        loss_targets = targets

        # Build dynamic targets if using dynamic queries
        if self.decoder.dynamic_queries and "encoder" in outputs and "selected_query_indices" in outputs["encoder"]:
            selected_indices = outputs["encoder"]["selected_query_indices"].detach()
            query_constituent_mask, first_occurrence, query_target_valid, query_target_idx = self.build_dynamic_targets(
                selected_indices, targets, self.dynamic_query_source, self.target_object
            )

            # Create a separate dict for loss computation with filtered targets
            loss_targets = targets.copy()
            loss_targets[f"{self.target_object}_{self.dynamic_query_source}_valid"] = query_constituent_mask
            loss_targets["query_first_occurrence"] = first_occurrence
            loss_targets[f"{self.target_object}_valid"] = query_target_valid

            # Add mapping needed for post-hoc alignment of predictions to full targets
            targets["query_target_idx"] = query_target_idx

        # Sort targets if using a sorter
        if self.sorter is not None:
            targets = self.sorter.sort_targets(targets, decoder_outputs["final"][self.sorter.input_sort_field])
            loss_targets = self.sorter.sort_targets(loss_targets, decoder_outputs["final"][self.sorter.input_sort_field])

        return targets, loss_targets, encoder_outputs, decoder_outputs

    def _compute_encoder_losses(self, encoder_outputs: dict, targets: dict) -> dict[str, dict[str, Tensor]]:
        """Compute losses for encoder tasks (no matching required).

        Args:
            encoder_outputs: Dictionary of encoder layer outputs.
            targets: The data containing the targets.

        Returns:
            Dictionary of encoder losses keyed by layer name and task name.
        """
        losses: dict[str, dict[str, Tensor]] = {}
        for layer_name, layer_outputs in encoder_outputs.items():
            losses[layer_name] = {}
            for task in self.encoder_tasks:
                if task.name not in layer_outputs:
                    continue
                losses[layer_name][task.name] = task.loss(layer_outputs[task.name], targets)
        return losses

    def _compute_decoder_costs(self, decoder_outputs: dict, targets: dict) -> dict[str, Tensor]:
        """Compute costs for decoder layers by aggregating task costs.

        Args:
            decoder_outputs: Dictionary of decoder layer outputs.
            targets: The data containing the targets.

        Returns:
            Dictionary of costs keyed by layer name. Cost axes are (batch, pred, true).
        """
        costs = {}

        # Compute costs for decoder layers
        for layer_name, layer_outputs in decoder_outputs.items():
            layer_costs = None

            # Get the cost contribution from each of the decoder tasks
            for task in self.tasks:
                # Skip tasks that do not contribute intermediate losses
                if task.name not in layer_outputs:
                    continue

                # Compute costs
                task_costs = task.cost(layer_outputs[task.name], targets)

                # Add the cost on to our running cost total, otherwise initialise a running cost matrix
                for cost in task_costs.values():
                    if layer_costs is None:
                        layer_costs = cost
                    else:
                        layer_costs += cost

            # Added to allow completely turning off inter layer loss
            # Possibly redundant as completely switching them off performs worse
            if layer_costs is not None:
                layer_costs = layer_costs.detach()

            costs[layer_name] = layer_costs

        return costs

    def _match_and_permute_outputs(self, decoder_outputs: dict, costs: dict[str, Tensor], targets: dict) -> None:
        """Perform optimal matching and permute decoder outputs accordingly.

        After permutation, outputs are aligned with target order, so the original
        particle_valid mask can be used directly by all tasks for loss computation.

        Args:
            decoder_outputs: Dictionary of decoder layer outputs (will be modified in-place).
            costs: Dictionary of costs keyed by layer name.
            targets: The data containing the targets.
        """
        # Stack all layer costs into a single 4D tensor for parallel matching
        layer_names = list(costs.keys())
        num_layers = len(layer_names)

        if num_layers > 0:
            # Stack costs: [num_layers, batch, num_pred, num_target]
            stacked_costs = torch.stack([costs[name] for name in layer_names], dim=0)
            batch_size = stacked_costs.shape[1]
            num_pred = stacked_costs.shape[2]
            num_target = stacked_costs.shape[3]

            # Reshape to [num_layers * batch, num_pred, num_target] to use layers as additional batch dim
            stacked_costs = stacked_costs.view(num_layers * batch_size, num_pred, num_target)

            # Expand validity mask to match stacked batch dimension: [num_layers * batch, num_target]
            target_valid = targets[f"{self.target_object}_valid"]
            stacked_target_valid = target_valid.unsqueeze(0).expand(num_layers, -1, -1).reshape(num_layers * batch_size, -1)

            # Get the indices that can permute the predictions to yield their optimal matching
            # Output shape: [num_layers * batch, num_pred]
            stacked_pred_idxs = self.matcher(stacked_costs, stacked_target_valid)

            # Reshape back to [num_layers, batch, num_pred]
            stacked_pred_idxs = stacked_pred_idxs.view(num_layers, batch_size, num_pred)

            # Create batch indices for indexing
            batch_idxs_expanded = torch.arange(batch_size, device=stacked_pred_idxs.device).unsqueeze(1)

            # Apply layer-specific permutations
            for layer_idx, layer_name in enumerate(layer_names):
                pred_idxs = stacked_pred_idxs[layer_idx]

                for task in self.tasks:
                    if not task.should_permute_outputs(layer_name, decoder_outputs[layer_name]):
                        continue

                    for output_name in task.outputs:
                        output_tensor = decoder_outputs[layer_name][task.name][output_name]
                        decoder_outputs[layer_name][task.name][output_name] = output_tensor[batch_idxs_expanded, pred_idxs]

    def _compute_decoder_losses(self, decoder_outputs: dict, loss_targets: dict) -> dict[str, dict[str, Tensor]]:
        """Compute final losses for decoder tasks using permuted outputs.

        Args:
            decoder_outputs: Dictionary of decoder layer outputs (already permuted).
            loss_targets: The targets dict to use for loss computation (filtered for dynamic queries).

        Returns:
            Dictionary of decoder losses keyed by layer name and task name.
        """
        losses: dict[str, dict[str, Tensor]] = {}

        for layer_name, layer_outputs in decoder_outputs.items():
            losses[layer_name] = {}

            for task in self.tasks:
                if task.name not in layer_outputs:
                    continue

                task_losses = task.loss(layer_outputs[task.name], loss_targets)
                losses[layer_name][task.name] = task_losses

        return losses

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        """Computes the loss between the forward pass of the model and the data / targets.

        This method performs Hungarian matching to align predictions with targets before computing
        losses. The matching works as follows:

        1. **Cost Matrix**: A cost matrix of shape [batch, num_pred, num_target] is computed by summing
           task-specific costs (e.g., BCE for classification, L1 for regression).

        2. **Hungarian Matching**: The matcher solves the linear assignment problem on the transposed
           cost matrix [batch, num_target, num_pred] and returns `pred_idxs` of shape [batch, num_pred]
           where `pred_idxs[i]` = which prediction slot should be placed at target position `i`.

        3. **Output Permutation**: Outputs are gathered using `output[batch_idxs, pred_idxs]`, which
           reorders predictions so that `output[i]` corresponds to `target[i]`. After this permutation,
           the original `particle_valid` mask can be used directly to filter matched pairs.

        For dynamic queries, a separate `loss_targets` dict is used for loss computation,
        containing a subset of the original targets that correspond to the dynamically initialised queries.
        Outputs are stored with realignment info and later aligned to the original target
        dimension by `predict()`.

        Args:
            outputs: The outputs produced by the forward pass of the model.
            targets: The data containing the targets.

        Returns:
            Tuple of (outputs, targets, losses) where:
            - outputs: The outputs dict (realigned for dynamic queries).
            - targets: The targets dict, enriched with query_target_idx for dynamic queries.
            - losses: A dictionary containing the computed losses for each task.
        """
        # Prepare targets and separate encoder/decoder outputs
        # For dynamic queries: targets = original (for metrics), loss_targets is filtered (for loss)
        targets, loss_targets, encoder_outputs, decoder_outputs = self._prepare_targets_and_outputs(outputs, targets)

        # Compute encoder losses (no matching required)
        losses = self._compute_encoder_losses(encoder_outputs, loss_targets)

        # Compute costs for decoder layers
        costs = self._compute_decoder_costs(decoder_outputs, loss_targets)

        # Perform matching and permute decoder outputs to align with target order
        # After this, output[i] corresponds to target[i] for all i
        self._match_and_permute_outputs(decoder_outputs, costs, loss_targets)

        # Compute final decoder losses using permuted outputs
        decoder_losses = self._compute_decoder_losses(decoder_outputs, loss_targets)
        losses.update(decoder_losses)

        # For dynamic queries, store realignment information for use by predict().
        # This ensures predict() can produce outputs with consistent shape regardless of query type.
        if "query_target_idx" in targets:
            if "encoder" not in outputs:
                outputs["encoder"] = {}
            outputs["encoder"]["query_target_idx"] = targets["query_target_idx"]
            outputs["encoder"]["num_original_targets"] = targets[f"{self.target_object}_valid"].shape[1]

        return outputs, targets, losses
