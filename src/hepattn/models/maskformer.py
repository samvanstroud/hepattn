import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoderLayer
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask, ObjectRegressionTask, ObjectHitMaskTask
from hepattn.flex.local_ca import auto_local_ca_mask

LOG_EVERY_N_STEPS=1000

class MaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module | None,
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
        raw_variables: list[str] | None = None,
        log_attn_mask: bool = False,
        query_posenc: nn.Module | None = None,
        key_posenc: nn.Module | None = None,
        preserve_posenc: bool = False,
        phi_analysis: bool = False,
        learnable_query_phi: bool = False,
        use_decoder_mask: bool = False,
        diagonal: bool = False,
        window_size: int = 512,
        combine_masks = "and",
    ):
        """Initializes the MaskFormer model, which is a modular transformer-style architecture designed
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
        raw_variables : list[str] or None, optional
            A list of variable names that passed to tasks without embedding.
        """
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.pooling = pooling
        self.tasks = tasks
        self.target_object = target_object
        self.matcher = matcher
        self.num_queries = num_queries
        self.query_initial = nn.Parameter(torch.randn(num_queries, dim))
        self.input_sort_field = input_sort_field
        self.use_attn_masks = use_attn_masks
        self.use_query_masks = use_query_masks
        self.raw_variables = raw_variables or []
        self.log_attn_mask = log_attn_mask
        self.log_step = 0
        self.query_posenc = query_posenc
        self.key_posenc = key_posenc
        self.preserve_posenc = preserve_posenc
        self.phi_analysis = phi_analysis
        self.learnable_query_phi = learnable_query_phi
        if self.learnable_query_phi:
            self.query_phi_param = nn.Parameter(
                2 * torch.pi * (torch.arange(num_queries) / num_queries - 0.5)
            )
        self.diagonal = diagonal
        self.window_size = window_size
        self.use_decoder_mask = use_decoder_mask
        self.combine_masks = combine_masks

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Atomic input names
        input_names = [input_net.input_name for input_net in self.input_nets]
        self.log_step += 1

        assert "key" not in input_names, "'key' input name is reserved."
        assert "query" not in input_names, "'query' input name is reserved."

        x = {}

        for raw_var in self.raw_variables:
            # If the raw variable is present in the inputs, add it directly to the output
            if raw_var in inputs:
                x[raw_var] = inputs[raw_var]

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
        x["query_valid"] = torch.full((batch_size, self.num_queries), True, device=x["query_embed"].device)

        if (self.key_posenc is not None) or (self.query_posenc is not None):
            x = self.generate_positional_encodings(x)
        if not self.preserve_posenc:
            x = self.add_positional_encodings(x)

        # Do any pooling if desired
        if self.pooling is not None:
            x_pooled = self.pooling(x[f"{self.pooling.input_name}_embed"], x[f"{self.pooling.input_name}_valid"])
            x[f"{self.pooling.output_name}_embed"] = x | x_pooled

        # Pass encoded inputs through decoder to produce outputs
        outputs = {}
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}

            attn_masks = {}
            query_mask = None

            # Re-add positional encodings (similar to SAM's prompt token re-addition)
            if self.preserve_posenc:
                x = self.add_positional_encodings(x)

            for task in self.tasks:
                # Get the outputs of the task given the current embeddings and record them
                task_outputs = task(x)

                # Need this for incidence-based regression task
                if isinstance(task, IncidenceRegressionTask):
                    # Assume that the incidence task has only one output
                    x["incidence"] = task_outputs[task.outputs[0]].detach()
                if isinstance(task, ObjectClassificationTask):
                    # Assume that the classification task has only one output
                    x["class_probs"] = task_outputs[task.outputs[0]].detach()
                outputs[f"layer_{layer_index}"][task.name] = task_outputs

                # Store attention masks from hit mask tasks for use by other tasks
                #TODO add here if phialignment task in tasks
                if isinstance(task, ObjectHitMaskTask):
                    if task.outputs and task.outputs[0] in task_outputs and self.use_attn_masks:
                        attn_logits = task_outputs[task.outputs[0]]
                        attn_mask = attn_logits.sigmoid() >= 0.1
                        x[f"{task.input_hit}_attn_logits"] = attn_logits
                        x[f"{task.input_hit}_attn_mask"] = attn_mask

                if self.use_attn_masks:
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

            #TODO order attn mask and key embeds here
            attn_mask, x, query_mask, key_valid = self.sort_attn_mask_device(attn_mask, x, query_mask, x.get("key_valid"))

            if attn_mask is not None:
                self.attn_mask_logging(attn_mask, x, layer_index)

            if self.use_decoder_mask:
                decoder_mask = auto_local_ca_mask(x["query_embed"], x["key_embed"], self.window_size, wrap=False, diagonal=self.diagonal)     
                self.decoder_attn_mask_logging(decoder_mask, x, layer_index)
                if attn_mask is not None:
                    if self.combine_masks=="and":
                        attn_mask = torch.logical_and(decoder_mask, attn_mask)
                    elif self.combine_masks=="or":
                        attn_mask = torch.logical_or(decoder_mask, attn_mask)
                else:
                    attn_mask = decoder_mask
            
            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"], x["key_embed"], attn_mask=attn_mask, q_mask=query_mask, kv_mask=key_valid
            )

            #TODO unorder the attn_mask and embeds here
            attn_mask, x, query_mask, key_valid = self.unsort_attn_mask_device(attn_mask, x, query_mask, key_valid)

            #TODO if using the decoder attn mask this shouldn't be unsorted but I don't think this is a problem because we don't use it after here anyway?

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

            # Need this for incidence-based regression task
            if isinstance(task, IncidenceRegressionTask):
                # Assume that the incidence task has only one output
                x["incidence"] = outputs["final"][task.name][task.outputs[0]].detach()
            if isinstance(task, ObjectClassificationTask):
                # Assume that the classification task has only one output
                x["class_probs"] = outputs["final"][task.name][task.outputs[0]].detach()
            if isinstance(task, ObjectRegressionTask) and self.phi_analysis:
                # Assume that the regression task has only one output
               regressed_phi = outputs["final"][task.name][task.outputs[0]].detach()
               self.last_regressed_phi = regressed_phi[0].squeeze(-1).float().cpu().numpy()
            if isinstance(task, ObjectHitMaskTask):
               if self.log_attn_mask:
                   attn_mask = outputs["final"][task.name][task.outputs[0]].detach()
                   self.final_attn_mask_logging(attn_mask, x, layer_index)

        return outputs

    
    def sort_attn_mask_device(self, attn_mask, x, query_mask, key_valid):
            key_sort_value_ = x.get(f"key_phi")
            key_sort_idx = torch.argsort(key_sort_value_, axis=-1)
            
            if attn_mask is not None:
                # Sort attention mask along key dimension (dim 2) - [batch, queries, constituents]
                attn_mask = attn_mask.index_select(2, key_sort_idx[0])
            
            # Sort key embeddings - [batch, constituents, dim]
            key_embed = x.get(f"key_embed")
            key_embed_sorted = key_embed[0][key_sort_idx[0]]  # [constituents, dim]
            key_embed_sorted = key_embed_sorted.unsqueeze(0)  # [1, constituents, dim]
            x["key_embed"] = key_embed_sorted

            # Sort key_valid mask - [batch, constituents]
            if key_valid is not None:
                key_valid_sorted = key_valid[0][key_sort_idx[0]]  # [constituents]
                key_valid_sorted = key_valid_sorted.unsqueeze(0)  # [1, constituents]
                x["key_valid"] = key_valid_sorted

            # Sort queries
            query_sort_value = x.get(f"query_phi")
            query_sort_idx = torch.argsort(query_sort_value, axis=-1)
            
            if attn_mask is not None:
                # Sort attention mask along query dimension (dim 1) - [batch, queries, constituents]
                attn_mask = attn_mask.index_select(1, query_sort_idx.to(attn_mask.device))
            
            # Sort query embeddings - [batch, queries, dim]
            query_embed = x.get(f"query_embed")
            query_embed_sorted = query_embed[0][query_sort_idx]  # [queries, dim]
            query_embed_sorted = query_embed_sorted.unsqueeze(0)  # [1, queries, dim]
            x["query_embed"] = query_embed_sorted
            
            # Sort query_mask - [batch, queries]
            if query_mask is not None:
                query_mask_sorted = query_mask[0][query_sort_idx]  # [queries]
                query_mask_sorted = query_mask_sorted.unsqueeze(0)  # [1, queries]
                query_mask = query_mask_sorted
            
            return attn_mask, x, query_mask, x.get("key_valid")
    
    def unsort_attn_mask_device(self, attn_mask, x, query_mask, key_valid):
        # Get the original sorting indices
        key_sort_value_ = x.get(f"key_phi")
        key_sort_idx = torch.argsort(key_sort_value_, axis=-1)
        key_unsort_idx = torch.argsort(key_sort_idx[0])  # Reverse the sorting
        
        if attn_mask is not None:
            # Unsort attention mask along key dimension (dim 2) - [batch, queries, constituents]
            attn_mask = attn_mask.index_select(2, key_unsort_idx.to(attn_mask.device))
        
        # Unsort key embeddings - [batch, constituents, dim]
        key_embed = x.get(f"key_embed")
        key_embed_unsorted = key_embed[0][key_unsort_idx]  # [constituents, dim]
        key_embed_unsorted = key_embed_unsorted.unsqueeze(0)  # [1, constituents, dim]
        x["key_embed"] = key_embed_unsorted

        # Unsort key_valid mask - [batch, constituents]
        if key_valid is not None:
            key_valid_unsorted = key_valid[0][key_unsort_idx]  # [constituents]
            key_valid_unsorted = key_valid_unsorted.unsqueeze(0)  # [1, constituents]
            x["key_valid"] = key_valid_unsorted

        # Unsort queries
        query_sort_value = x.get(f"query_phi")
        query_sort_idx = torch.argsort(query_sort_value, axis=-1)
        query_unsort_idx = torch.argsort(query_sort_idx)  # Reverse the sorting
        
        if attn_mask is not None:
            # Unsort attention mask along query dimension (dim 1) - [batch, queries, constituents]
            attn_mask = attn_mask.index_select(1, query_unsort_idx.to(attn_mask.device))
        
        # Unsort query embeddings - [batch, queries, dim]
        query_embed = x.get(f"query_embed")
        query_embed_unsorted = query_embed[0][query_unsort_idx]  # [queries, dim]
        query_embed_unsorted = query_embed_unsorted.unsqueeze(0)  # [1, queries, dim]
        x["query_embed"] = query_embed_unsorted

        # Unsort query mask - [batch, queries]
        if query_mask is not None:
            query_mask_unsorted = query_mask[0][query_unsort_idx]  # [queries]
            query_mask_unsorted = query_mask_unsorted.unsqueeze(0)  # [1, queries]
            query_mask = query_mask_unsorted
        
        return attn_mask, x, query_mask, x.get("key_valid")
    

       
    def sort_attn_mask(self, attn_mask_im, x):
        key_sort_value_ = x.get(f"key_phi")
        key_sort_idx = torch.argsort(key_sort_value_, axis=-1)
        attn_mask_im = attn_mask_im.index_select(1, key_sort_idx[0].to(attn_mask_im.device))
        # sort key phi for storing too
        key_phi = x.get(f"key_phi").detach().cpu().numpy()  # [num_keys]
        key_phi_sorted = key_phi[0][key_sort_idx[0].cpu().numpy()]
        self.last_key_phi = key_phi_sorted
        # sort queries
        query_sort_value = x.get(f"query_phi")
        query_sort_idx = torch.argsort(query_sort_value, axis=-1)
        attn_mask_im = attn_mask_im.index_select(0, query_sort_idx.to(attn_mask_im.device))
        # sort query phi for storing too
        query_phi = x.get(f"query_phi").detach().cpu().numpy()  # [num_queries]
        query_phi_sorted = query_phi[query_sort_idx.cpu().numpy()]
        self.last_query_phi = query_phi_sorted
        return attn_mask_im
    
    def attn_mask_logging(self, attn_mask, x, layer_index):
        if (self.log_attn_mask
            # and (attn_mask is not None)
            and (self.log_step % LOG_EVERY_N_STEPS == 0) or (not self.training)
            ):
            if not hasattr(self, "attn_masks_to_log"):
                self.attn_masks_to_log = {}
            if layer_index == 0 or layer_index == len(self.decoder_layers) - 1:
                attn_mask_im = attn_mask[0].detach().cpu().clone().int()
                # attn_mask_im = self.sort_attn_mask(attn_mask_im, x)
                self.attn_masks_to_log[layer_index] = {
                    "mask": attn_mask_im,
                    "step": self.log_step,
                    "layer": layer_index,
                }

    def decoder_attn_mask_logging(self, attn_mask, x, layer_index):
        if (self.log_attn_mask
            # and (attn_mask is not None)
            and (self.log_step % LOG_EVERY_N_STEPS == 0) or (not self.training)
            ):
            if not hasattr(self, "decoder_attn_masks_to_log"):
                self.decoder_attn_masks_to_log = {}
            if layer_index == 0 or layer_index == len(self.decoder_layers) - 1:
                attn_mask_im = attn_mask[0].detach().cpu().clone().int()
                # attn_mask_im = self.sort_attn_mask(attn_mask_im, x)
                self.decoder_attn_masks_to_log[layer_index] = {
                    "mask": attn_mask_im,
                    "step": self.log_step,
                    "layer": layer_index,
                }

    def final_attn_mask_logging(self, attn_logits, x, layer_index, threshold=0.1):
        if (self.log_attn_mask
            and (attn_logits is not None)
            and (self.log_step % LOG_EVERY_N_STEPS == 0) or (not self.training)
            ):
            if not hasattr(self, "final_attn_masks_to_log"):
                self.final_attn_masks_to_log = {}
            if layer_index == 0 or layer_index == len(self.decoder_layers) - 1:
                # sigmoid the attn mask to get the probability of the hit being attended to
                attn_mask_im = attn_logits[0].detach().cpu().clone().sigmoid()
                attn_mask_im = self.sort_attn_mask(attn_mask_im, x)
                self.final_attn_masks_to_log[layer_index] = {
                    "mask": attn_mask_im >= threshold,
                    "probs": attn_mask_im,
                    "step": self.log_step,
                    "layer": layer_index,
                }

    # def store_key_phi_info(self, x: dict):
    #     self.last_key_phi =x['key_phi'][0].cpu().numpy()  
    #     if "key_posenc" in x:
    #         self.last_key_posenc = x["key_posenc"][0].cpu().numpy()
    #     else:
    #         self.last_key_posenc = None

    def add_positional_encodings(self, x: dict):
        if (self.query_posenc is not None):
            x["query_embed"] = x["query_embed"] + x["query_posenc"]
        if self.key_posenc is not None:
            x["key_embed"] = x["key_embed"] + x["key_posenc"]
        return x

    def generate_positional_encodings(self, x: dict):
        if self.query_posenc is not None:
            if self.learnable_query_phi:
                # Use learnable parameter, expand to batch size
                query_phi = self.query_phi_param
            else:
                query_phi = 2 * torch.pi * (torch.arange(self.num_queries, device=x["query_embed"].device) / self.num_queries - 0.5)
            x["query_phi"] = query_phi
            x["query_posenc"] = self.query_posenc(x)
            if self.phi_analysis:
                self.last_query_phi = x["query_phi"].detach().cpu().numpy()
                self.last_query_posenc = x["query_posenc"].detach().cpu().numpy()
        if self.key_posenc is not None:
            x["key_posenc"] = self.key_posenc(x)
            if self.phi_analysis:
                key_phi = x["key_phi"].detach().cpu().numpy()
                self.last_key_phi = key_phi
                key_posenc =  x["key_posenc"][0].cpu()
                self.last_key_posenc = key_posenc.numpy()
                key_sort_idx = torch.argsort(torch.tensor(key_phi), axis=-1)
                key_posencs_sorted = key_posenc[key_sort_idx[0]]
                self.last_key_posenc_sorted = key_posencs_sorted
        return x
    

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
                    if layer_costs is not None:
                        layer_costs += cost
                    else:
                        layer_costs = cost

            # Added to allow completely turning off inter layer loss
            # Possibly redundant as completely switching them off performs worse
            if layer_costs is not None:
                layer_costs = layer_costs.detach()

            costs[layer_name] = layer_costs

        # Permute the outputs for each output in each layer
        self.log_pred_idxs = {}
        for layer_name, cost in costs.items():
            if cost is None:
                continue

            # Get the indicies that can permute the predictions to yield their optimal matching
            pred_idxs = self.matcher(cost, targets[f"{self.target_object}_valid"])
            
            if (self.log_attn_mask
            # and (attn_mask is not None)
            and (self.log_step % LOG_EVERY_N_STEPS == 0) or (not self.training)
            ):
                # store pred_idxs for each layer
                self.log_pred_idxs[layer_name] = pred_idxs.detach().cpu().numpy()

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
                if layer_name != "final" and not task.has_intermediate_loss:
                    continue
                # In case if some tasks needed to get access to other task's output
                # extra_kwargs = task.loss_kwargs(outputs[layer_name], targets)
                # losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], targets, **extra_kwargs)
                losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], targets)

        return losses