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
        self.decoder.encoder_tasks = encoder_tasks if encoder_tasks is not None else nn.ModuleList()
        self.pooling = pooling
        self.tasks = tasks
        self.encoder_tasks = encoder_tasks if encoder_tasks is not None else nn.ModuleList()
        self.target_object = target_object
        self.matcher = matcher
        self.unified_decoding = unified_decoding
        self.decoder.unified_decoding = unified_decoding
        self.dynamic_query_source = dynamic_query_source

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
    def build_dynamic_targets(selected_hit_indices: Tensor, targets: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build dynamic targets for selected query hits.

        Args:
            selected_hit_indices: (N_queries,) indices of hits selected as queries
            targets: dict with hit_particle_id (1, N_hits), particle_id (1, N_particles), etc.

        Returns:
            Tuple of:
                - query_hit_valid: (1, N_queries, N_hits) mask indicating which hits belong to each query's particle
                - first_occurrence: (1, N_queries) mask indicating which queries are the first hit of their particle
                - query_particle_valid: (1, N_queries) mask indicating which query slots are valid
                - query_particle_idx: (1, N_queries) mapping from query index to source particle index
        """
        hit_particle_id = targets["hit_particle_id"][0]  # (N_hits,)
        particle_id = targets["particle_id"][0]  # (N_particles,)

        # Get the particle ID for each selected hit/query
        query_particle_id = hit_particle_id[selected_hit_indices]  # (N_queries,)

        # Create mapping from query index to particle index (vectorized)
        # Find which particle index corresponds to each query's particle_id
        # Shape: (N_queries, N_particles) where each row has True at matching particle indices
        # Detach to avoid gradient retention on large comparison tensor
        matches_mask = (query_particle_id.unsqueeze(1) == particle_id.unsqueeze(0)).detach()
        # Get first matching particle index for each query (argmax returns 0 if no match, which is ok)
        query_particle_idx = matches_mask.long().argmax(dim=1)

        # Handle duplicates: only keep FIRST occurrence of each particle
        # This ensures multiple hits from same particle don't create duplicate targets
        # Vectorized approach: sort by inverse_indices and find boundaries
        _, inverse_indices = torch.unique(query_particle_id, return_inverse=True, sorted=False)
        sorted_inverse, sorted_idx = torch.sort(inverse_indices, stable=True)
        # Identify positions where inverse_index changes (first occurrences in sorted order)
        is_first_sorted = torch.cat([torch.tensor([True], device=inverse_indices.device), sorted_inverse[1:] != sorted_inverse[:-1]])
        # Map back to original positions
        first_occurrence = torch.zeros_like(query_particle_id, dtype=torch.bool)
        first_occurrence[sorted_idx[is_first_sorted]] = True

        # Build the mask: query attends to all hits belonging to same particle
        # query_particle_id: (N_queries,), hit_particle_id: (N_hits,)
        query_hit_valid = query_particle_id.unsqueeze(-1) == hit_particle_id.unsqueeze(0)
        # Shape: (N_queries, N_hits)

        # Zero out duplicate query rows (only first hit per particle gets targets)
        query_hit_valid[~first_occurrence] = False

        # Create validity mask: only first occurrence queries are valid
        query_particle_valid = first_occurrence

        # Add batch dimension
        query_hit_valid = query_hit_valid.unsqueeze(0)  # (1, N_queries, N_hits)
        first_occurrence = first_occurrence.unsqueeze(0)  # (1, N_queries)
        query_particle_valid = query_particle_valid.unsqueeze(0)  # (1, N_queries)
        query_particle_idx = query_particle_idx.unsqueeze(0)  # (1, N_queries)

        return query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx

    def forward(self, inputs: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
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

        return preds

    def _prepare_targets_and_outputs(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        """Prepare targets and separate encoder/decoder outputs.

        Args:
            outputs: The outputs produced by the forward pass of the model.
            targets: The data containing the targets.

        Returns:
            Tuple of (prepared_targets, encoder_outputs, decoder_outputs)
        """
        # Create a copy of targets to avoid mutating the input dict
        # This is especially important when using dynamic queries
        targets = targets.copy()

        # Separate encoder and decoder outputs for cleaner logic (copy instead of pop to avoid mutation)
        encoder_outputs = {"encoder": outputs["encoder"]} if "encoder" in outputs else {}
        decoder_outputs = {k: v for k, v in outputs.items() if k != "encoder"}  # All non-encoder outputs are decoder layers

        # Build dynamic targets if using dynamic queries
        # Extract selected_query_indices from layer_0 outputs (stored by decoder)
        if self.decoder.dynamic_queries and "layer_0" in decoder_outputs and "_selected_query_indices" in decoder_outputs["layer_0"]:
            # Preserve full targets before overwriting them for metrics computation
            targets["particle_valid_full"] = targets[f"{self.target_object}_valid"].clone().detach()
            targets["particle_hit_valid_full"] = targets["particle_hit_valid"].clone().detach()

            selected_indices = decoder_outputs["layer_0"]["_selected_query_indices"].detach()
            query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx = self.build_dynamic_targets(selected_indices, targets)
            # Override the static particle_hit_valid with dynamic query-based mask
            targets["particle_hit_valid"] = query_hit_valid
            targets["query_first_occurrence"] = first_occurrence
            # Override particle_valid to match the number of dynamic queries
            targets[f"{self.target_object}_valid"] = query_particle_valid
            # Store query-to-particle mapping for metrics alignment
            targets["query_particle_idx"] = query_particle_idx

        # Sort targets if using a sorter (both encoder and decoder need sorted targets)
        if self.sorter is not None:
            targets = self.sorter.sort_targets(targets, decoder_outputs["final"][self.sorter.input_sort_field])

        return targets, encoder_outputs, decoder_outputs

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

    def _match_and_permute_outputs(self, decoder_outputs: dict, costs: dict[str, Tensor], targets: dict) -> dict:
        """Perform optimal matching and permute decoder outputs accordingly.

        Args:
            decoder_outputs: Dictionary of decoder layer outputs (will be modified in-place).
            costs: Dictionary of costs keyed by layer name.
            targets: The data containing the targets (will be modified to include aligned validity masks).

        Returns:
            Updated targets dictionary with aligned validity masks.
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

                # Create a prediction-aligned validity mask for the targets
                # This is needed when using dynamic queries where num_predictions != num_targets
                targets[f"{layer_name}_{self.target_object}_valid_aligned"] = target_valid[batch_idxs_expanded, pred_idxs]

                for task in self.tasks:
                    if not task.should_permute_outputs(layer_name, decoder_outputs[layer_name]):
                        continue

                    for output_name in task.outputs:
                        output_tensor = decoder_outputs[layer_name][task.name][output_name]
                        decoder_outputs[layer_name][task.name][output_name] = output_tensor[batch_idxs_expanded, pred_idxs]

        return targets

    def _compute_decoder_losses(self, decoder_outputs: dict, targets: dict) -> dict[str, dict[str, Tensor]]:
        """Compute final losses for decoder tasks using permuted outputs.

        Args:
            decoder_outputs: Dictionary of decoder layer outputs (already permuted).
            targets: The data containing the targets (includes aligned validity masks).

        Returns:
            Dictionary of decoder losses keyed by layer name and task name.
        """
        losses: dict[str, dict[str, Tensor]] = {}

        for layer_name, layer_outputs in decoder_outputs.items():
            losses[layer_name] = {}

            # Create a modified targets dict with the aligned validity mask for this layer
            layer_targets = targets.copy()
            if f"{layer_name}_{self.target_object}_valid_aligned" in targets:
                layer_targets[f"{self.target_object}_valid"] = targets[f"{layer_name}_{self.target_object}_valid_aligned"]

            for task in self.tasks:
                if task.name not in layer_outputs:
                    continue

                task_losses = task.loss(layer_outputs[task.name], layer_targets)
                losses[layer_name][task.name] = task_losses

        return losses

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict]:
        """Computes the loss between the forward pass of the model and the data / targets.
        It first computes the cost / loss between each of the predicted and true object
        and then uses the Hungarian algorithm to perform an optimal bipartite matching. The model
        predictions are then permuted to match this optimal matching, after which the final loss
        between the model and target is computed.

        Args:
            outputs: The outputs produced by the forward pass of the model.
            targets: The data containing the targets.

        Returns:
            losses: A dictionary containing the computed losses for each task.
        """
        # Prepare targets and separate encoder/decoder outputs
        targets, encoder_outputs, decoder_outputs = self._prepare_targets_and_outputs(outputs, targets)

        # Compute encoder losses (no matching required)
        losses = self._compute_encoder_losses(encoder_outputs, targets)

        # Compute costs for decoder layers
        costs = self._compute_decoder_costs(decoder_outputs, targets)

        # Perform matching and permute decoder outputs
        targets = self._match_and_permute_outputs(decoder_outputs, costs, targets)

        # Compute final decoder losses
        decoder_losses = self._compute_decoder_losses(decoder_outputs, targets)
        losses.update(decoder_losses)

        return losses, targets
