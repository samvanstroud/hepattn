import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoderLayer
from hepattn.models.task import ObjectHitMaskTask

class MaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        tasks: nn.ModuleList,
        num_queries: int,
        dim: int,
        matcher: nn.Module | None = None,
        input_sort_field: str | None = None,
        use_attn_masks: bool = True,
        use_query_masks: bool = True,
        log_attn_mask: bool = False,
    ):
        """
        Initializes the MaskFormer model, which is a modular transformer-style architecture designed
        for multi-task object inference with attention-based decoding and optional encoder blocks.

        Parameters
        ----------
        input_nets : nn.ModuleList
            A list of input modules, each responsible for embedding a specific input type.
        encoder : nn.Module
            An optional encoder module that processes merged input embeddings with optional sorting.
        decoder_layer_config : dict
            Configuration dictionary used to initialize each MaskFormerDecoderLayer.
        num_decoder_layers : int
            The number of decoder layers to stack.
        tasks : nn.ModuleList
            A list of task modules, each responsible for producing and processing predictions from decoder outputs.
        matcher : nn.Module or None
            A module used to match predictions to targets (e.g., using the Hungarian algorithm) for loss computation.
        num_queries : int
            The number of object-level queries to initialize and decode.
        dim : int
            The dimensionality of the query and key embeddings.
        input_sort_field : str or None, optional
            An optional key used to sort the input objects (e.g., for windowed attention).
        use_attn_masks : bool, optional
            If True, attention masks will be used to control which input objects are attended to.
        use_query_masks : bool, optional
            If True, query masks will be used to control which queries are valid during attention.
        """
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(**decoder_layer_config) for _ in range(num_decoder_layers)])
        self.tasks = tasks
        self.matcher = matcher
        self.num_queries = num_queries
        self.query_initial = nn.Parameter(torch.randn(num_queries, dim))
        self.input_sort_field = input_sort_field
        self.use_attn_masks = use_attn_masks
        self.use_query_masks = use_query_masks
        self.log_attn_mask = log_attn_mask
        self.step_ = 0
        self.logger = None

    def log_figure(self, name, fig, step=None):
        """Log a matplotlib figure to the logger (Comet)."""
        if self.logger is not None and hasattr(self.logger, 'experiment'):
            # Use the provided step or the current step
            current_step = step if step is not None else self.step_
            
            # Log the figure to Comet
            self.logger.experiment.log_figure(
                figure_name=name,
                figure=fig,
                step=current_step
            )
            # Close the figure to free memory
            fig.close()

    def get_last_attention_mask(self):
        """Get the last attention mask that was stored for logging.
        Returns
        -------
        tuple or None
            A tuple of (attention_mask, step, layer) if available, None otherwise.
        """
        if hasattr(self, '_last_attn_mask'):
            return (
                self._last_attn_mask,
                getattr(self, '_last_attn_mask_step', 0),
                getattr(self, '_last_attn_mask_layer', 0)
            )
        return None

    def clear_last_attention_mask(self):
        """Clear the stored attention mask after logging."""
        if hasattr(self, '_last_attn_mask'):
            del self._last_attn_mask
        if hasattr(self, '_last_attn_mask_step'):
            del self._last_attn_mask_step
        if hasattr(self, '_last_attn_mask_layer'):
            del self._last_attn_mask_layer

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Atomic input names
        input_names = [input_net.input_name for input_net in self.input_nets]
        self.step_+=1

        assert "key" not in input_names, "'key' input name is reserved."
        assert "query" not in input_names, "'query' input name is reserved."

        x = {}

        # Embed the input objects
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]

            # These slices can be used to pick out specific
            # objects after we have merged them all together
            # TODO: Clean this up
            device = inputs[input_name + "_valid"].device
            x[f"key_is_{input_name}"] = torch.cat(
                [torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device, dtype=torch.bool) for i in input_names], dim=-1
            )

        # Merge the input objects and he padding mask into a single set
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in input_names], dim=-1)

        # calculate the batch size and combined number of input constituents
        batch_size = x["key_valid"].shape[0]
        num_constituents = x["key_valid"].shape[-1]

        # if all key_valid are true, then we can just set it to None
        if batch_size == 1 and x["key_valid"].all():
            x["key_valid"] = None

        # Also merge the field being used for sorting in window attention if requested
        if self.input_sort_field is not None:
            x[f"key_{self.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.input_sort_field] for input_name in input_names], dim=-1
            )

        # Pass merged input hits through the encoder
        if self.encoder is not None:
            # Note that a padded feature is a feature that is not valid!
            x["key_embed"] = self.encoder(x["key_embed"], x_sort_value=x.get(f"key_{self.input_sort_field}"), kv_mask=x.get("key_valid"))

        # Unmerge the updated features back into the separate input types
        # These are just views into the tensor that old all the merged hits
        for input_name in input_names:
            x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        # Generate the queries that represent objects
        x["query_embed"] = self.query_initial.expand(batch_size, -1, -1)
        x["query_valid"] = torch.full((batch_size, self.num_queries), True)

        # Pass encoded inputs through decoder to produce outputs
        outputs = {}
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            attn_masks = {}
            query_mask = None

            for task in self.tasks:
                # Get the outputs of the task given the current embeddings and record them
                task_outputs = task(x)

                outputs[f"layer_{layer_index}"][task.name] = task_outputs

                # Here we check if each task has an attention mask to contribute, then after
                # we fill in any attention masks for any features that did not get an attention mask
                task_attn_masks = task.attn_mask(task_outputs)

                for input_name, attn_mask in task_attn_masks.items():
                    # We only want to mask an attention slot if every task agrees the slots should be masked
                    # so we only mask if both the existing and new attention mask are masked, which means a slot is valid if
                    # either current or new mask is valid
                    if input_name in attn_masks:
                        attn_masks[input_name] |= attn_mask
                    else:
                        attn_masks[input_name] = attn_mask

                # Now do same but for query masks
                task_query_mask = task.query_mask(task_outputs)

                if task_query_mask is not None and self.use_query_masks:
                    query_mask = query_mask | task_query_mask if query_mask is not None else task_query_mask

            # Fill in attention masks for features that did not get one specified by any task
            if attn_masks and self.use_attn_masks:
                attn_mask = torch.full((batch_size, self.num_queries, num_constituents), True, device=x["key_embed"].device)

                for input_name, input_attn_mask in attn_masks.items():
                    attn_mask[..., x[f"key_is_{input_name}"]] = input_attn_mask

            # If no attention masks were specified, set it to none to avoid redundant masking
            else:
                attn_mask = None

            if (
                self.log_attn_mask
                and (attn_mask is not None)
                and (self.step_ % 1000 == 0)
            ):
                # Store for callback to log later
                self._last_attn_mask = attn_mask[0].detach().cpu().clone()
                self._last_attn_mask_step = self.step_
                self._last_attn_mask_layer = layer_index

            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"], x["key_embed"], attn_mask=attn_mask, q_mask=query_mask, kv_mask=x.get("key_valid")
            )

            # Unmerge the updated features back into the separate input types
            for input_name in input_names:
                x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        # Get the final outputs - we don't need to compute attention masks or update things here
        outputs["final"] = {}
        for task in self.tasks:
            outputs["final"][task.name] = task(x)

        return outputs

    def predict(self, outputs: dict) -> dict:
        """Takes the raw model outputs and produces a set of actual inferences / predictions.
        For example will take output probabilies and apply threshold cuts to prduce boolean predictions.

        Parameters
        ----------
        outputs:
            The outputs produces the forward pass of the model.
        """
        preds = {}

        # Compute predictions for each task in each block
        for layer_name, layer_outputs in outputs.items():
            preds[layer_name] = {}

            for task in self.tasks:
                preds[layer_name][task.name] = task.predict(layer_outputs[task.name])

        return preds

    def loss(self, outputs: dict, targets: dict) -> dict:
        """Computes the loss between the forward pass of the model and the data / targets.
        It first computes the cost / loss between each of the predicted and true tracks in each ROI
        and then uses the Hungarian algorihtm to perform an optimal bipartite matching. The model
        predictions are then permuted to match this optimal matching, after which the final loss
        between the model and target is computed.

        Parameters
        ----------
        outputs:
            The outputs produces the forward pass of the model.
        outputs:
            The data containing the targets.
        """
        # Will hold the costs between all pairs of objects - cost axes are (batch, pred, true)
        costs = {}
        batch_idxs = torch.arange(targets["particle_valid"].shape[0]).unsqueeze(1)
        for layer_name, layer_outputs in outputs.items():
            layer_costs = None

            # Get the cost contribution from each of the tasks
            for task in self.tasks:
                # Skip tasks that do not contribute intermediate losses
                if layer_name != "final" and not task.has_intermediate_loss:
                    continue

                # Only use the cost from the final set of predictions
                task_costs = task.cost(layer_outputs[task.name], targets)

                # Add the cost on to our running cost total, otherwise initialise a running cost matrix
                for cost in task_costs.values():
                    if layer_costs is not None:
                        layer_costs += cost
                    else:
                        layer_costs = cost

            costs[layer_name] = layer_costs.detach()

        # Permute the outputs for each output in each layer
        for layer_name in costs:
            # Get the indicies that can permute the predictions to yield their optimal matching
            pred_idxs = self.matcher(costs[layer_name], targets["particle_valid"])

            # Apply the permutation in place
            for task in self.tasks:
                for output_name in task.outputs:
                    outputs[layer_name][task.name][output_name] = outputs[layer_name][task.name][output_name][batch_idxs, pred_idxs]

        # Compute the losses for each task in each block
        losses = {}
        for layer_name in outputs:
            losses[layer_name] = {}
            for task in self.tasks:
                # Skip tasks that are not ObjectHitMaskTask for intermediate layers
                if layer_name != "final" and not isinstance(task, ObjectHitMaskTask):
                    continue
                losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], targets)

        return losses
