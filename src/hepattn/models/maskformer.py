import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask, ObjectHitMaskTask

N_STEPS_LOG_ATTN_MASK = 1000


class MaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module | None,
        decoder: MaskFormerDecoder,
        tasks: nn.ModuleList,
        dim: int,
        target_object: str = "particle",
        pooling: nn.Module | None = None,
        matcher: nn.Module | None = None,
        input_sort_field: str | None = None,
        raw_variables: list[str] | None = None,
        log_task_attn_mask: bool = False,
    ):
        """Initializes the MaskFormer model, which is a modular transformer-style architecture designed
        for multi-task object inference with attention-based decoding and optional encoder blocks.

        Parameters
        ----------
        input_nets : nn.ModuleList
            A list of input modules, each responsible for embedding a specific input type.
        encoder : nn.Module
            An optional encoder module that processes merged input embeddings with optional sorting.
        decoder : MaskFormerDecoder
            The decoder module that handles multi-layer decoding and task integration.
        tasks : nn.ModuleList
            A list of task modules, each responsible for producing and processing predictions from decoder outputs.
        matcher : nn.Module or None
            A module used to match predictions to targets (e.g., using the Hungarian algorithm) for loss computation.
        dim : int
            The dimensionality of the query and key embeddings.
        target_object : str
            The target object name which is used to mark valid/invalid objects during matching.
        input_sort_field : str or None, optional
            An optional key used to sort the input objects (e.g., for windowed attention).
        raw_variables : list[str] or None, optional
            A list of variable names that passed to tasks without embedding.
        """
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder = decoder

        # Set tasks as a member of the decoder and extract num_queries
        self.decoder.tasks = tasks
        self.num_queries = decoder.num_queries

        self.pooling = pooling
        self.tasks = tasks
        self.target_object = target_object
        self.matcher = matcher
        self.query_initial = nn.Parameter(torch.randn(self.num_queries, dim))
        self.input_sort_field = input_sort_field
        self.raw_variables = raw_variables or []
        self.log_task_attn_mask = log_task_attn_mask

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        self.log_step += 1
        # Atomic input names
        input_names = [input_net.input_name for input_net in self.input_nets]

        assert "key" not in input_names, "'key' input name is reserved."
        assert "query" not in input_names, "'query' input name is reserved."

        x = {}

        for raw_var in self.raw_variables:
            # If the raw variable is present in the inputs, add it directly to the output
            if raw_var in inputs:
                x[raw_var] = inputs[raw_var]

        # Store input positional encodings if we need to preserve them for the decoder
        if self.decoder.preserve_posenc:
            assert all(input_net.posenc is not None for input_net in self.input_nets)
            x["key_posenc"] = torch.concatenate([input_net.posenc(inputs) for input_net in self.input_nets], dim=-2)

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
        x["query_valid"] = torch.full((batch_size, self.num_queries), True, device=x["query_embed"].device)

        # Pass through decoder layers
        x, outputs = self.decoder(x, input_names)

        # Do any pooling if desired
        if self.pooling is not None:
            x_pooled = self.pooling(x[f"{self.pooling.input_name}_embed"], x[f"{self.pooling.input_name}_valid"])
            x[f"{self.pooling.output_name}_embed"] = x_pooled

        # Get the final outputs - we don't need to compute attention masks or update things here
        outputs["final"] = {}
        for task in self.tasks:
            outputs["final"][task.name] = task(x)

            # Need this for incidence-based regression task
            if isinstance(task, IncidenceRegressionTask):
                # Assume that the incidence task has only one output
                x["incidence"] = outputs["final"][task.name][task.outputs[0]].detach()
            if isinstance(task, ObjectClassificationTask):
                # Assume that the classification task has only one output
                x["class_probs"] = outputs["final"][task.name][task.outputs[0]].detach()
            if isinstance(task, ObjectHitMaskTask) and self.log_task_attn_mask:
                attn_logits = outputs["final"][task.name][task.outputs[0]].detach()
                self.output_attn_mask_logging(attn_logits, x)
        return outputs

    def sort_attn_mask_im(self, attn_mask_im, x):
        key_sort_value_ = x.get("key_phi")
        key_sort_idx = torch.argsort(key_sort_value_, axis=-1)
        attn_mask_im = attn_mask_im.index_select(1, key_sort_idx[0].to(attn_mask_im.device))
        query_sort_value = x.get("query_phi")
        query_sort_idx = torch.argsort(query_sort_value, axis=-1)
        return attn_mask_im.index_select(0, query_sort_idx.to(attn_mask_im.device))

    def output_attn_mask_logging(self, attn_logits, x, threshold=0.1):
        if ((attn_logits is not None) and (self.log_step % N_STEPS_LOG_ATTN_MASK == 0)) or (not self.training):
            if not hasattr(self, "output_attn_masks_to_log"):
                self.output_attn_masks_to_log = {}
            # sigmoid the attn mask to get the probability of the hit being attended to
            attn_mask_im = attn_logits[0].detach().cpu().clone().sigmoid()
            attn_mask_im = self.sort_attn_mask_im(attn_mask_im, x)
            self.output_attn_masks_to_log["final"] = {
                "mask": attn_mask_im >= threshold,
                "probs": attn_mask_im,
                "step": self.log_step,
                "layer": "final",
            }

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
                if layer_name != "final" and not task.has_intermediate_loss:
                    continue
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
        targets:
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
                    if layer_costs is None:
                        layer_costs = cost
                    else:
                        layer_costs += cost

            # Added to allow completely turning off inter layer loss
            # Possibly redundant as completely switching them off performs worse
            if layer_costs is not None:
                layer_costs = layer_costs.detach()

            costs[layer_name] = layer_costs

        # Permute the outputs for each output in each layer
        for layer_name, cost in costs.items():
            if cost is None:
                continue

            # Get the indicies that can permute the predictions to yield their optimal matching
            pred_idxs = self.matcher(cost, targets[f"{self.target_object}_valid"])

            for task in self.tasks:
                # Tasks without a object dimension do not need permutation (constituent-level or sample-level)
                if not task.permute_loss:
                    continue

                # The task didn't produce an output for this layer, so skip it
                if layer_name != "final" and not task.has_intermediate_loss:
                    continue

                for output_name in task.outputs:
                    outputs[layer_name][task.name][output_name] = outputs[layer_name][task.name][output_name][batch_idxs, pred_idxs]

        # Compute the losses for each task in each block
        losses = {}
        for layer_name in outputs:
            losses[layer_name] = {}
            for task in self.tasks:
                if layer_name != "final" and not task.has_intermediate_loss:
                    continue
                # In case if some tasks needed to get access to other task's output
                extra_kwargs = task.loss_kwargs(outputs[layer_name], targets)
                losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], targets, **extra_kwargs)

        return losses
