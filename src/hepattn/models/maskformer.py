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
        target_object: str = "particle",
        pooling: nn.Module | None = None,
        matcher: nn.Module | None = None,
        input_sort_field: str | None = None,
        use_attn_masks: bool = True,
        use_query_masks: bool = True,
        query_posenc: nn.Module | None = None,
        preserve_original_queries: bool = False,
        preserve_original_keys: bool = False,
        preserve_input_posenc: bool = False,
        log_attn_mask: bool = False,
        pe_scale_factor: float = 1,
        # New parameters for flexible query re-addition
        query_readd_strategy: str = "fixed_scale",  # "fixed_scale", "learned_gate", "layer_specific"
        query_readd_layers: list[int] | None = None,  # Specific layers to re-add at (None = all layers)
        query_readd_gate_init: float = 0.1,  # Initial value for learned gates
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
        target_object : str
            The target object name which is used to mark valid/invalid objects during matching.
        input_sort_field : str or None, optional
            An optional key used to sort the input objects (e.g., for windowed attention).
        use_attn_masks : bool, optional
            If True, attention masks will be used to control which input objects are attended to.
        use_query_masks : bool, optional
            If True, query masks will be used to control which queries are valid during attention.
        query_posenc : nn.Module or None, optional
            Optional positional encoding module for queries.
        preserve_original_queries : bool, optional
            If True, original query embeddings will be re-added at each layer.
        preserve_original_keys : bool, optional
            If True, original key embeddings will be re-added at each layer.
        preserve_input_posenc : bool, optional
            If True, input positional encodings will be re-added at each layer.
        log_attn_mask : bool, optional
            If True, attention masks will be logged for debugging.
        pe_scale_factor : float, optional
            Scale factor for positional encoding re-addition.
        query_readd_strategy : str, optional
            Strategy for query re-addition: "fixed_scale", "learned_gate", or "layer_specific".
        query_readd_layers : list[int] or None, optional
            Specific layers to re-add queries at (None = all layers).
        query_readd_gate_init : float, optional
            Initial value for learned gates in query re-addition.
        """
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(**decoder_layer_config) for _ in range(num_decoder_layers)])
        self.pooling = pooling
        self.tasks = tasks
        self.target_object = target_object
        self.matcher = matcher
        self.num_queries = num_queries
        self.query_initial = nn.Parameter(torch.randn(num_queries, dim))
        self.input_sort_field = input_sort_field
        self.use_attn_masks = use_attn_masks
        self.use_query_masks = use_query_masks
        self.query_posenc = query_posenc
        self.preserve_original_queries = preserve_original_queries
        self.preserve_original_keys = preserve_original_keys
        self.preserve_input_posenc = preserve_input_posenc
        self.pe_scale_factor = pe_scale_factor
        self.log_attn_mask = log_attn_mask
        self.log_step = 0

        # Initialize query re-addition components
        self.query_readd_strategy = query_readd_strategy
        self.query_readd_layers = query_readd_layers
        
        # Initialize learned gates for query re-addition if using learned_gate strategy
        if query_readd_strategy == "learned_gate":
            self.query_readd_gates = nn.Parameter(
                torch.full((num_decoder_layers,), query_readd_gate_init)
            )
        elif query_readd_strategy == "layer_specific":
            # Layer-specific learned scales
            self.query_readd_scales = nn.Parameter(
                torch.full((num_decoder_layers,), query_readd_gate_init)
            )
    

    def _compute_query_readd_scale(self, layer_index: int) -> float:
        """Compute the scale factor for query re-addition based on the chosen strategy.
        
        Parameters
        ----------
        layer_index : int
            Current decoder layer index
            
        Returns
        -------
        float
            Scale factor for re-addition
        """
        # Check if we should re-add at this layer
        if self.query_readd_layers is not None and layer_index not in self.query_readd_layers:
            return 0.0
        
        if self.query_readd_strategy == "fixed_scale":
            return self.pe_scale_factor
        elif self.query_readd_strategy == "learned_gate":
            return torch.sigmoid(self.query_readd_gates[layer_index]) * self.pe_scale_factor
        elif self.query_readd_strategy == "layer_specific":
            return torch.sigmoid(self.query_readd_scales[layer_index]) * self.pe_scale_factor
        else:
            raise ValueError(f"Unknown query re-addition strategy: {self.query_readd_strategy}")

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Atomic input names
        input_names = [input_net.input_name for input_net in self.input_nets]
        self.log_step+=1

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

        # Store input positional encodings if we need to preserve them
        if self.preserve_input_posenc:
            input_posencs = []
            for input_net in self.input_nets:
                if input_net.posenc is not None:
                    # Get just the positional encoding part
                    posenc = input_net.posenc(inputs)
                    input_posencs.append(posenc)
                else:
                    raise ValueError(f"Input net {input_net.input_name} has no positional encoding.")

            x["key_posenc"] = torch.concatenate(input_posencs, dim=-2)

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
            if self.preserve_original_keys:
                x["key_embed_original"] = x["key_embed"].clone()

            # Note that a padded feature is a feature that is not valid!
            x["key_embed"] = self.encoder(x["key_embed"], x_sort_value=x.get(f"key_{self.input_sort_field}"), kv_mask=x.get("key_valid"))

        # Unmerge the updated features back into the separate input types
        # These are just views into the tensor that old all the merged hits
        for input_name in input_names:
            x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        # Generate the queries that represent objects
        x["query_embed"] = self.query_initial.expand(batch_size, -1, -1)
        x["query_valid"] = torch.full((batch_size, self.num_queries), True, device=x["query_embed"].device)

        # Do any pooling if desired
        if self.pooling is not None:
            x_pooled = self.pooling(x[f"{self.pooling.input_name}_embed"], x[f"{self.pooling.input_name}_valid"])
            x[f"{self.pooling.output_name}_embed"] = x | x_pooled


        # Add positional encoding to the queries
        if self.query_posenc is not None:
            device = x["key_embed"].device
            if self.preserve_original_queries:
                x["query_posenc"] = self.query_posenc(inputs, batch_size, self.num_queries, device)
                x["query_embed"] = x["query_embed"] + x["query_posenc"]
            else:
                x["query_embed"] = x["query_embed"] + self.query_posenc(inputs, batch_size, self.num_queries, device)


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
                and (self.log_step % 1000 == 0)
            ):
                # Store for callback to log later
                attn_mask_im = attn_mask[0].detach().cpu().clone()

                key_sort_value_ = x.get(f"key_phi")
                key_sort_idx = torch.argsort(key_sort_value_, axis=-1)
                attn_mask_im = attn_mask_im.index_select(1, key_sort_idx[0].to(attn_mask_im.device))

                # # Suppose query_phi = ... (shape [batch_size, num_queries])
                # query_sort_value = x.get(f"query_phi")
                # query_sort_idx = torch.argsort(query_sort_value, dim=-1)
                # attn_mask_im = attn_mask_im.index_select(0, query_sort_idx[0].to(attn_mask_im.device))

                if not hasattr(self, "attn_masks_to_log"):
                    self.attn_masks_to_log = {}
                if layer_index == 0 or layer_index == len(self.decoder_layers) - 1:
                    self.attn_masks_to_log[layer_index] = {
                        "mask": attn_mask_im,
                        "step": self.log_step,
                        "layer": layer_index,
                    }

            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"], x["key_embed"], attn_mask=attn_mask, q_mask=query_mask, kv_mask=x.get("key_valid")
            )
            
            # Compute re-addition scale if we're using query re-addition strategies
            if self.preserve_original_queries and self.query_readd_strategy is not None:
                readd_scale = self._compute_query_readd_scale(layer_index)
            else:
                readd_scale = self.pe_scale_factor

            # Re-add original query embeddings (similar to SAM's prompt token re-addition)
            if (self.query_posenc is not None) and (self.preserve_original_queries):
                x["query_embed"] = x["query_embed"] + readd_scale * x["query_posenc"]

            # Re-add original key embeddings if requested
            if self.preserve_original_keys:
                x["key_embed"] = x["key_embed"] + readd_scale * x["key_embed_original"]

            # Re-add input positional encodings if requested
            if self.preserve_input_posenc:
                x["key_embed"] = x["key_embed"] + readd_scale * x["key_posenc"]

            # Unmerge the updated features back into the separate input types
            for input_name in input_names:
                x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

            # Do any pooling if desired
            if self.pooling is not None:
                x_pooled = self.pooling(x[f"{self.pooling.input_name}_embed"], x[f"{self.pooling.input_name}_valid"])
                x[f"{self.pooling.output_name}_embed"] = x | x_pooled

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
        batch_idxs = torch.arange(targets[f"{self.target_object}_valid"].shape[0]).unsqueeze(1)
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

            if layer_costs is not None:
                costs[layer_name] = layer_costs.detach()

        # Permute the outputs for each output in each layer
        for layer_name in costs:
            # Get the indicies that can permute the predictions to yield their optimal matching
            pred_idxs = self.matcher(costs[layer_name], targets[f"{self.target_object}_valid"])

            # Apply the permutation in place
            for task in self.tasks:
                # Some tasks, such as hit-level or sample-level tasks, do not need permutation
                if hasattr(task, "permute_loss"):
                    if not task.permute_loss:
                        continue

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

    def get_readd_layers_info(self) -> dict:
        """Get information about which layers are being used for query re-addition.
        
        Returns
        -------
        dict
            Dictionary containing layer selection information
        """
        if self.query_readd_layers is None:
            readd_layers = range(len(self.decoder_layers))
            num_readd_layers = len(self.decoder_layers)
        else:
            readd_layers = sorted(self.query_readd_layers)
            num_readd_layers = len(self.query_readd_layers)
            
        return {
            "readd_layers": readd_layers,
            "num_readd_layers": num_readd_layers,
            "total_layers": len(self.decoder_layers),
            "readd_frequency": num_readd_layers / len(self.decoder_layers),
            "strategy": self.query_readd_strategy,
            "layer_range": None,
            "layer_step": None,
            "layer_freq": None,
            "layer_pattern": None,
        }
