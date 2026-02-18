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
        self.dim = decoder.dim

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

        # Get query_mask from encoder outputs for masking padded queries in predictions
        query_mask = outputs.get("encoder", {}).get("query_mask")

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
                    preds[layer_name][task.name] = task.predict(layer_outputs[task.name], query_mask=query_mask)

        return preds

    def _pad_decoder_outputs_for_loss(self, decoder_outputs: dict, targets: dict, encoder_outputs: dict) -> tuple[dict, dict, dict]:
        """Pad decoder outputs and query_mask up to the fixed num_queries for matching/loss."""
        encoder_outputs = encoder_outputs.copy()
        encoder_outputs["encoder"] = encoder_outputs["encoder"].copy()
        query_embed = encoder_outputs["encoder"].get("query_embed")
        num_queries = self.decoder.num_queries(encoder_outputs["encoder"])
        num_pred = query_embed.shape[1]
        num_padding = num_queries - num_pred

        if num_padding < 0:
            raise ValueError(
                f"Number of predicted queries ({num_pred}) exceeds configured num_queries ({num_queries})."
            )
        if num_padding == 0:
            return decoder_outputs, targets, encoder_outputs

        null_padding = torch.zeros(query_embed.shape[0], num_padding, self.dim, device=query_embed.device, dtype=query_embed.dtype)
        query_embed = torch.cat([query_embed, null_padding], dim=1)  # (1, num_queries, dim)

        query_mask = torch.cat(
            [
                torch.ones(query_embed.shape[0], num_pred, dtype=torch.bool, device=query_embed.device),
                torch.zeros(query_embed.shape[0], num_padding, dtype=torch.bool, device=query_embed.device),
            ],
            dim=1,
        )

        encoder_outputs["encoder"]["query_embed"] = query_embed
        encoder_outputs["encoder"]["query_mask"] = query_mask

        targets = targets.copy()
        targets["query_mask"] = query_mask

        def pad_tensor(t: Tensor) -> Tensor:
            if t.dim() < 1:
                return t
            try:
                pad_dim = next(i for i, s in enumerate(t.shape) if s == num_pred)
            except StopIteration:
                return t
            padded_shape = list(t.shape)
            padded_shape[pad_dim] = num_queries
            padded = t.new_zeros(padded_shape)
            slices = [slice(None)] * t.dim()
            slices[pad_dim] = slice(0, num_pred)
            padded[tuple(slices)] = t
            return padded

        padded_decoder_outputs: dict = {}
        for layer_name, layer_outputs in decoder_outputs.items():
            if isinstance(layer_outputs, Tensor):
                padded_decoder_outputs[layer_name] = pad_tensor(layer_outputs)
                continue

            padded_layer_outputs: dict = {}
            for name, value in layer_outputs.items():
                if isinstance(value, Tensor):
                    if name == "attn_mask_log":
                        padded_layer_outputs[name] = value
                    else:
                        padded_layer_outputs[name] = pad_tensor(value)
                elif isinstance(value, dict):
                    padded_task_outputs: dict = {}
                    for out_name, out_val in value.items():
                        if isinstance(out_val, Tensor) and out_name != "attn_mask_log":
                            padded_task_outputs[out_name] = pad_tensor(out_val)
                        else:
                            padded_task_outputs[out_name] = out_val
                    padded_layer_outputs[name] = padded_task_outputs
                else:
                    padded_layer_outputs[name] = value
            padded_decoder_outputs[layer_name] = padded_layer_outputs

        return padded_decoder_outputs, targets, encoder_outputs

    def _prepare_targets_and_outputs(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        """Prepare targets and separate encoder/decoder outputs.

        Args:
            outputs: The outputs produced by the forward pass of the model.
            targets: The data containing the targets.

        Returns:
            Tuple of (targets, encoder_outputs, decoder_outputs) where:
            - targets: Targets dict with query_mask added if present
            - encoder_outputs: Separated encoder outputs
            - decoder_outputs: Separated decoder layer outputs
        """
        # Separate encoder and decoder outputs for cleaner logic
        encoder_outputs = {"encoder": outputs["encoder"]} if "encoder" in outputs else {}
        decoder_outputs = {k: v for k, v in outputs.items() if k != "encoder"}

        # Include query_mask in targets if present (for masking padded query losses)
        if "encoder" in outputs and "query_mask" in outputs["encoder"] and "query_mask" not in targets:
            targets = targets.copy()
            targets["query_mask"] = outputs["encoder"]["query_mask"]

        # Sort targets if using a sorter
        if self.sorter is not None:
            targets = self.sorter.sort_targets(targets, decoder_outputs["final"][self.sorter.input_sort_field])

        # Pad decoder outputs and query_mask for matching/loss if dynamic queries are shorter
        decoder_outputs, targets, encoder_outputs = self._pad_decoder_outputs_for_loss(decoder_outputs, targets, encoder_outputs)

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
                losses[layer_name][task.name] = task.loss(layer_outputs[task.name], targets, layer_outputs=layer_outputs)
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
            stacked_costs = stacked_costs.reshape(num_layers * batch_size, num_pred, num_target)

            # Expand validity mask to match stacked batch dimension: [num_layers * batch, num_target]
            target_valid = targets[f"{self.target_object}_valid"]
            stacked_target_valid = target_valid.unsqueeze(0).expand(num_layers, -1, -1).reshape(num_layers * batch_size, -1)

            # Get query_mask if present (for masking padded queries in matching)
            query_mask = targets.get("query_mask")
            stacked_query_valid = None
            if query_mask is not None:
                stacked_query_valid = query_mask.unsqueeze(0).expand(num_layers, -1, -1).reshape(num_layers * batch_size, -1)

            # Get the indices that can permute the predictions to yield their optimal matching
            # Output shape: [num_layers * batch, num_pred]
            stacked_pred_idxs = self.matcher(stacked_costs, stacked_target_valid, stacked_query_valid)

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

    def _compute_decoder_losses(self, decoder_outputs: dict, targets: dict) -> dict[str, dict[str, Tensor]]:
        """Compute final losses for decoder tasks using permuted outputs.

        Args:
            decoder_outputs: Dictionary of decoder layer outputs (already permuted).
            targets: The targets dict to use for loss computation.

        Returns:
            Dictionary of decoder losses keyed by layer name and task name.
        """
        losses: dict[str, dict[str, Tensor]] = {}

        for layer_name, layer_outputs in decoder_outputs.items():
            losses[layer_name] = {}

            for task in self.tasks:
                if task.name not in layer_outputs:
                    continue

                task_losses = task.loss(layer_outputs[task.name], targets, layer_outputs=layer_outputs)
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

        Args:
            outputs: The outputs produced by the forward pass of the model.
            targets: The data containing the targets.

        Returns:
            Tuple of (outputs, targets, losses) where:
            - outputs: The outputs dict.
            - targets: The targets dict.
            - losses: A dictionary containing the computed losses for each task.
        """
        # Prepare targets and separate encoder/decoder outputs
        targets, encoder_outputs, decoder_outputs = self._prepare_targets_and_outputs(outputs, targets)

        # Compute encoder losses (no matching required)
        losses = self._compute_encoder_losses(encoder_outputs, targets)

        # Compute costs for decoder layers
        costs = self._compute_decoder_costs(decoder_outputs, targets)

        # Perform matching and permute decoder outputs to align with target order
        # After this, output[i] corresponds to target[i] for all i
        self._match_and_permute_outputs(decoder_outputs, costs, targets)

        # Compute final decoder losses using permuted outputs
        decoder_losses = self._compute_decoder_losses(decoder_outputs, targets)
        losses.update(decoder_losses)

        # Ensure outputs reflect any padding/permutation updates done during loss
        outputs = outputs.copy()
        if "encoder" in encoder_outputs:
            outputs["encoder"] = encoder_outputs["encoder"]
        outputs.update(decoder_outputs)

        return outputs, targets, losses
