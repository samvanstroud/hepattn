from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, focal_loss, loss_fns
from hepattn.utils.masks import topk_attn
from hepattn.utils.scaling import FeatureScaler

# Pick a value that is safe for float16
COST_PAD_VALUE = 1e4


class Task(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the forward pass of the task."""

    @abstractmethod
    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Return predictions from model outputs."""

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute loss between outputs and targets."""

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs, **kwargs):
        return {}

    def key_mask(self, outputs, **kwargs):
        return {}

    def query_mask(self, outputs, **kwargs):
        return None


class ObjectValidTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
        mask_queries: bool = False,
    ):
        """Task used for classifying whether object candidates / seeds should be
        taken as reconstructed / pred objects or not.

        Parameters
        ----------
        name : str
            Name of the task - will be used as the key to separate task outputs.
        input_object : str
            Name of the input object feature
        output_object : str
            Name of the output object feature which will denote if the predicted object slot is used or not.
        target_object: str
            Name of the target object feature that we want to predict is valid or not.
        losses : dict[str, float]
            Dict specifying which losses to use. Keys denote the loss function name,
            whiel value denotes loss weight.
        costs : dict[str, float]
            Dict specifying which costs to use. Keys denote the cost function name,
            whiel value denotes cost weight.
        dim : int
            Embedding dimension of the input features.
        null_weight : float
            Weight applied to the null class in the loss. Useful if many instances of
            the target class are null, and we need to reweight to overcome class imbalance.
        """
        super().__init__()

        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight
        self.mask_queries = mask_queries

        # Internal
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_logit"]
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Network projects the embedding down into a scalar
        x_logit = self.net(x[self.input_object + "_embed"])
        return {self.output_object + "_logit": x_logit.squeeze(-1)}

    def predict(self, outputs, threshold=0.5):
        # Objects that have a predicted probability aove the threshold are marked as predicted to exist
        return {self.output_object + "_valid": outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold}

    def cost(self, outputs, targets):
        output = outputs[self.output_object + "_logit"].detach().to(torch.float32)
        target = targets[self.target_object + "_valid"].to(torch.float32)
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = COST_PAD_VALUE
        return costs

    def loss(self, outputs, targets):
        losses = {}
        output = outputs[self.name][self.output_object + "_logit"]
        target = targets[self.target_object + "_valid"].type_as(output)
        weight = target + self.null_weight * (1 - target)
        # Calculate the loss from each specified loss function.
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=None, weight=weight)
        return losses

    def query_mask(self, outputs, threshold=0.1):
        if not self.mask_queries:
            return None

        return outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold


class HitFilterTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        target_field: str,
        dim: int,
        threshold: float = 0.1,
        mask_keys: bool = False,
        loss_fn: Literal["bce", "focal", "both"] = "bce",
    ):
        """Task used for classifying whether hits belong to reconstructable objects or not."""
        super().__init__()

        self.name = name
        self.input_object = input_object
        self.target_field = target_field
        self.dim = dim
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.mask_keys = mask_keys

        # Internal
        self.input_objects = [f"{input_object}_embed"]
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_logit = self.net(x[f"{self.input_object}_embed"])
        return {f"{self.input_object}_logit": x_logit.squeeze(-1)}

    def predict(self, outputs: dict) -> dict:
        return {f"{self.input_object}_{self.target_field}": outputs[f"{self.input_object}_logit"].sigmoid() >= self.threshold}

    def loss(self, outputs: dict, targets: dict) -> dict:
        # Pick out the field that denotes whether a hit is on a reconstructable object or not
        output = outputs[self.name][f"{self.input_object}_logit"]
        target = targets[f"{self.input_object}_{self.target_field}"].type_as(output)

        # Calculate the BCE loss with class weighting
        if self.loss_fn == "bce":
            weight = 1 / target.float().mean()
            loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "focal":
            loss = focal_loss(output, target)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "both":
            weight = 1 / target.float().mean()
            bce_loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
            focal_loss_value = focal_loss(output, target)
            return {
                f"{self.input_object}_bce": bce_loss,
                f"{self.input_object}_focal": focal_loss_value,
            }
        raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def key_mask(self, outputs, threshold=0.1):
        if not self.mask_keys:
            return {}

        return {self.input_object: outputs[f"{self.input_object}_logit"].detach().sigmoid() >= threshold}


class ObjectHitMaskTask(Task):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
        mask_attn: bool = True,
        logit_scale: float = 1.0,
        pred_threshold: float = 0.5,
    ):
        super().__init__()

        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight
        self.mask_attn = mask_attn
        self.logit_scale = logit_scale
        self.pred_threshold = pred_threshold

        self.output_object_hit = output_object + "_" + input_hit
        self.target_object_hit = target_object + "_" + input_hit
        self.inputs = [input_object + "_embed", input_hit + "_embed"]
        self.outputs = [self.output_object_hit + "_logit"]
        self.hit_net = Dense(dim, dim)
        # self.hit_net = nn.Identity()
        self.object_net = Dense(dim, dim)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Produce new task-specific embeddings for the hits and objects
        x_object = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_hit + "_embed"])

        # Object-hit probability is the dot product between the hit and object embedding
        object_hit_logit = self.logit_scale * torch.einsum("bnc,bmc->bnm", x_object, x_hit)

        # Zero out entries for any hit slots that are not valid
        object_hit_logit[~x[self.input_hit + "_valid"].unsqueeze(-2).expand_as(object_hit_logit)] = torch.finfo(object_hit_logit.dtype).min

        return {self.output_object_hit + "_logit": object_hit_logit}

    def attn_mask(self, outputs, threshold=0.1):
        if not self.mask_attn:
            return {}

        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= threshold

        # If the attn mask is completely padded for a given entry, unpad it - tested and is required (?)
        # TODO: See if the query masking stops this from being necessary
        # WHY?
        # attn_mask[torch.where(torch.all(attn_mask, dim=-1))] = False

        return {self.input_hit: attn_mask}

    def predict(self, outputs, threshold=0.5):
        # Object-hit pairs that have a predicted probability above the threshold are predicted as being associated to one-another
        return {self.output_object_hit + "_valid": outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= self.pred_threshold}

    def cost(self, outputs, targets):
        output = outputs[self.output_object_hit + "_logit"].detach()
        target = targets[self.target_object_hit + "_valid"].detach().to(output.dtype)

        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)

            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = COST_PAD_VALUE
        return costs

    def loss(self, outputs, targets):
        output = outputs[self.name][self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_valid"].type_as(output)

        # Build a padding mask for object-hit pairs
        # hit_pad = targets[self.input_hit + "_valid"].unsqueeze(-2).expand_as(target)
        # object_pad = targets[self.target_object + "_valid"].unsqueeze(-1).expand_as(target)
        # An object-hit is valid slot if both its object and hit are valid slots
        # TODO: Maybe calling this a mask is confusing since true entries are
        # object_hit_mask = object_pad & hit_pad

        # Mask only valid objects
        object_hit_mask = targets[self.target_object + "_valid"]

        # weight = target + self.null_weight * (1 - target)
        weight = None

        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            loss = loss_fns[loss_fn](output, target, mask=object_hit_mask, weight=weight)
            losses[loss_fn] = loss_weight * loss
        return losses


class RegressionTask(Task):
    def __init__(
        self,
        name: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
    ):
        super().__init__()

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.k = len(fields)
        # For standard regression number of DoFs is just the number of targets
        self.ndofs = self.k

    def forward(self, x: dict[str, Tensor], pads: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        latent = self.latent(x, pads=pads)
        return {self.output_object + "_regr": latent}

    def predict(self, outputs):
        # Split the regression vectior into the separate fields
        latent = outputs[self.output_object + "_regr"]
        return {self.output_object + "_" + field: latent[..., i] for i, field in enumerate(self.fields)}

    def loss(self, outputs, targets):
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        output = outputs[self.name][self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        # Compute the loss
        loss = torch.nn.functional.smooth_l1_loss(output, target, reduction="none")

        # Average over all the features
        loss = torch.mean(loss, dim=-1)

        # Compute the regression loss only for valid objects
        return {"smooth_l1": self.loss_weight * loss.mean()}

    def metrics(self, preds, targets):
        metrics = {}
        for field in self.fields:
            # Get the target and prediction only for valid targets
            pred = preds[self.output_object + "_" + field][targets[self.target_object + "_valid"]]
            target = targets[self.target_object + "_" + field][targets[self.target_object + "_valid"]]
            # Get the error between the prediction and target for this field
            err = pred - target
            # Compute the RMSE and log it
            metrics[field + "_rmse"] = torch.sqrt(torch.mean(torch.square(err)))
            # Compute the relative error / resolution and log it
            metrics[field + "_mean_res"] = torch.mean(err / target)
            metrics[field + "_std_res"] = torch.std(err / target)

        return metrics


class ObjectRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        dim: int,
    ):
        super().__init__(name, output_object, target_object, fields, loss_weight)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor], pads: dict[str, Tensor] | None = None) -> Tensor:
        return self.net(x[self.input_object + "_embed"])


class ObjectHitRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        dim: int,
    ):
        super().__init__(name, output_object, target_object, fields, loss_weight)

        self.input_hit = input_hit
        self.input_object = input_object

        self.inputs = [input_object + "_embed", input_hit + "_embed"]
        self.outputs = [self.output_object + "_regr"]

        self.dim = dim
        self.dim_per_dof = self.dim // self.ndofs

        self.hit_net = Dense(dim, self.ndofs * self.dim_per_dof)
        self.object_net = Dense(dim, self.ndofs * self.dim_per_dof)

    def latent(self, x: dict[str, Tensor], pads: dict[str, Tensor] | None = None) -> Tensor:
        # Embed the hits and tracks and reshape so we have a separate embedding for each DoF
        x_obj = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_hit + "_embed"])

        x_obj = x_obj.reshape(x_obj.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))  # Shape BNDE
        x_hit = x_hit.reshape(x_hit.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))  # Shape BMDE

        # Take the dot product between the hits and tracks over the last embedding dimension so we are left
        # with just a scalar for each degree of freedom
        x_obj_hit = torch.einsum("...nie,...mie->...nmi", x_obj, x_hit)  # Shape BNMD

        # If padding data is provided, use it to zero out predictions for any hit slots that are not valid
        if pads is not None:
            # Shape of padding goes BM -> B1M -> B1M1 -> BNMD
            x_obj_hit = x_obj_hit * pads[self.input_hit + "_valid"].unsqueeze(-2).unsqueeze(-1).expand_as(x_obj_hit).float()
        return x_obj_hit


class ObjectClassificationTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        net: nn.Module,
        num_classes: int,
        loss_class_weights: list[float] | None = None,
        null_weight: float = 1.0,
        mask_queries: bool = False,
    ):
        """Task used for object classification.

        Parameters
        ----------
        name : str
            Name of the task - will be used as the key to separate task outputs.
        input_object : str
            Name of the input object feature
        output_object : str
            Name of the output object feature which will denote if the predicted object slot is used or not.
        target_object: str
            Name of the target object feature that we want to predict is valid or not.
        losses : dict[str, float]
            Dict specifying which losses to use. Keys denote the loss function name,
            whiel value denotes loss weight.
        costs : dict[str, float]
            Dict specifying which costs to use. Keys denote the cost function name,
            whiel value denotes cost weight.
        net : nn.Module
            Network that will be used to classify the object classes.
        null_weight : float
            Weight applied to the null class in the loss. Useful if many instances of
            the target class are null, and we need to reweight to overcome class imbalance.
        """
        super().__init__()

        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.num_classes = num_classes

        class_weights = torch.ones(self.num_classes + 1, dtype=torch.float32)
        if loss_class_weights is not None:
            # If class weights are provided, use them to weight the loss
            if len(loss_class_weights) != self.num_classes:
                raise ValueError(f"Length of loss_class_weights ({len(loss_class_weights)}) does not match number of classes ({self.num_classes})")
            class_weights[: self.num_classes] = torch.tensor(loss_class_weights, dtype=torch.float32)
        class_weights[-1] = null_weight  # Last class is the null class, so set its weight to the null weight
        self.register_buffer("class_weights", class_weights)
        self.mask_queries = mask_queries

        # Internal
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_class_prob"]

        self.net = net

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Network projects the embedding down into a scalar
        x_class_prob = self.net(x[self.input_object + "_embed"])
        return {self.output_object + "_class_prob": x_class_prob.squeeze(-1)}

    def predict(self, outputs):
        classes = outputs[self.output_object + "_class_prob"].detach().argmax(-1)
        return {
            self.output_object + "_class": classes,
            self.output_object + "_valid": classes < self.num_classes,  # Valid if class is less than num_classes
        }

    def cost(self, outputs, targets):
        output = outputs[self.output_object + "_class_prob"].detach().to(torch.float32)
        target = targets[self.target_object + "_class"].long()
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = COST_PAD_VALUE
        return costs

    def loss(self, outputs, targets):
        losses = {}
        output = outputs[self.name][self.output_object + "_class_prob"]
        target = targets[self.target_object + "_class"].long()
        # Calculate the loss from each specified loss function.
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=None, weight=self.class_weights)
        return losses

    def query_mask(self, outputs):
        if not self.mask_queries:
            return None

        return outputs[self.output_object + "_class_prob"].detach().argmax(-1) < self.num_classes  # Valid if class is less than num_classes


class IncidenceRegressionTask(Task):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        net: nn.Module,
        node_net: nn.Module | None = None,
    ):
        """Incidence regression task."""
        super().__init__()
        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.net = net
        self.node_net = node_net if node_net is not None else nn.Identity()

        self.inputs = [input_object + "_embed", input_hit + "_embed"]
        self.outputs = [self.output_object + "_incidence"]

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_object = self.net(x[self.input_object + "_embed"])
        x_hit = self.node_net(x[self.input_hit + "_embed"])

        incidence_pred = torch.einsum("bqe,ble->bql", x_object, x_hit)
        incidence_pred = incidence_pred.softmax(dim=1) * x[self.input_hit + "_valid"].unsqueeze(1).expand_as(incidence_pred)

        return {self.output_object + "_incidence": incidence_pred}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return {self.output_object + "_incidence": outputs[self.output_object + "_incidence"].detach()}

    def cost(self, outputs, targets):
        output = outputs[self.output_object + "_incidence"].detach().to(torch.float32)
        target = targets[self.target_object + "_incidence"].to(torch.float32)

        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)

            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = COST_PAD_VALUE
        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses = {}
        output = outputs[self.name][self.output_object + "_incidence"]
        target = targets[self.target_object + "_incidence"].type_as(output)

        # Create a mask for valid nodes and objects
        node_mask = targets[self.input_hit + "_valid"].unsqueeze(1).expand_as(output)
        object_mask = targets[self.target_object + "_valid"].unsqueeze(-1).expand_as(output)
        mask = node_mask & object_mask
        # Calculate the loss from each specified loss function.
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=mask)

        return losses


class IncidenceBasedRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
        scale_dict_path: str,
        net: nn.Module,
        use_incidence: bool = True,
        use_nodes: bool = False,
        split_charge_neutral_loss: bool = False,
    ):
        """Regression task that uses incidence information to predict regression targets.

        Parameters
        ----------
        targets : list
            List of target names
        add_momentum : bool
            Whether to add scalar momentum to the predictions, computed from the px, py, pz predictions
        """
        super().__init__(name=name, output_object=output_object, target_object=target_object, fields=fields, loss_weight=loss_weight)
        self.input_hit = input_hit
        self.input_object = input_object
        self.scaler = FeatureScaler(scale_dict_path=scale_dict_path)
        self.use_incidence = use_incidence
        self.cost_weight = cost_weight
        self.net = net
        self.split_charge_neutral_loss = split_charge_neutral_loss
        self.use_nodes = use_nodes

        self.loss_masks = {
            "e": self.get_neutral,  # Only neutral particles
            "pt": self.get_charged,  # Only charged particles
        }

        self.inputs = [input_object + "_embed"] + [input_hit + "_" + field for field in fields]
        self.outputs = [output_object + "_regr", output_object + "_proxy_regr"]

    def get_charged(self, pred: Tensor, target: Tensor) -> Tensor:
        """Get a boolean mask for charged particles based on their class."""
        return (pred <= 2) & (target <= 2)

    def get_neutral(self, pred: Tensor, target: Tensor) -> Tensor:
        """Get a boolean mask for neutral particles based on their class."""
        return (pred > 2) & (target > 2)

    def forward(self, x: dict[str, Tensor], pads: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        # get the predictions
        if self.use_incidence:
            inc = x["incidence"].detach()
            proxy_feats, is_charged = self.get_proxy_feats(inc, x, class_probs=x["class_probs"].detach())
            input_data = torch.cat(
                [
                    x[self.input_object + "_embed"],
                    proxy_feats,
                    is_charged.unsqueeze(-1),
                ],
                -1,
            )
            if self.use_nodes:
                valid_mask = x[self.input_hit + "_valid"].unsqueeze(-1)
                masked_embed = x[self.input_hit + "_embed"] * valid_mask
                node_feats = torch.bmm(inc, masked_embed)
                input_data = torch.cat([input_data, node_feats], dim=-1)
        else:
            input_data = x[self.input_object + "_embed"]
            proxy_feats = torch.zeros_like(input_data[..., : len(self.fields)])
        preds = self.net(input_data)
        return {self.output_object + "_regr": preds, self.output_object + "_proxy_regr": proxy_feats}

    def predict(self, outputs):
        # Split the regression vectior into the separate fields
        pflow_regr = outputs[self.output_object + "_regr"]
        proxy_regr = outputs[self.output_object + "_proxy_regr"]
        return {self.output_object + "_" + field: pflow_regr[..., i] for i, field in enumerate(self.fields)} | {
            self.output_object + "_proxy_" + field: proxy_regr[..., i] for i, field in enumerate(self.fields)
        }

    def cost(self, outputs, targets) -> dict[str, Tensor]:
        eta_pos = self.fields.index("eta")
        sinphi_pos = self.fields.index("sinphi")
        cosphi_pos = self.fields.index("cosphi")

        pred_phi = torch.atan2(
            outputs[self.output_object + "_regr"][..., sinphi_pos],
            outputs[self.output_object + "_regr"][..., cosphi_pos],
        )[:, :, None]
        pred_eta = outputs[self.output_object + "_regr"][..., eta_pos][:, :, None]
        target_phi = torch.atan2(
            targets[self.target_object + "_sinphi"],
            targets[self.target_object + "_cosphi"],
        )[:, None, :]
        target_eta = targets[self.target_object + "_eta"][:, None, :]
        # Compute the cost based on the difference in phi and eta
        dphi = (pred_phi - target_phi + torch.pi) % (2 * torch.pi) - torch.pi
        deta = (pred_eta - target_eta) * self.scaler["eta"].scale

        # Compute the cost as the sum of the squared differences
        cost = self.cost_weight * torch.sqrt(dphi**2 + deta**2)
        cost[~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(cost)] = COST_PAD_VALUE

        return {"regression": cost}

    def loss(self, outputs, targets):
        loss = None
        target_class = targets[self.target_object + "_class"]
        output_class = outputs["classification"][self.output_object + "_class_prob"].detach().argmax(-1)
        for i, field in enumerate(self.fields):
            target = targets[self.target_object + "_" + field]
            output = outputs[self.name][self.output_object + "_regr"][..., i]
            mask = targets[self.target_object + "_valid"].clone()
            if self.split_charge_neutral_loss and field in self.loss_masks:
                mask = mask & self.loss_masks[field](output_class, target_class)
            if loss is None:
                loss = torch.nn.functional.smooth_l1_loss(output[mask], target[mask], reduction="mean")
            else:
                loss += torch.nn.functional.smooth_l1_loss(output[mask], target[mask], reduction="mean")
        # Average over all the features
        loss /= len(self.fields)

        # Compute the regression loss only for valid objects
        return {"smooth_l1": self.loss_weight * loss}

    def metrics(self, preds, targets):
        metrics = {}
        for field in self.fields:
            # Get the target and prediction only for valid targets
            pred = preds[self.output_object + "_" + field][targets[self.target_object + "_valid"]]
            target = targets[self.target_object + "_" + field][targets[self.target_object + "_valid"]]
            # Get the error between the prediction and target for this field
            err = pred - target
            # Compute the RMSE and log it
            metrics[field + "_rmse"] = torch.sqrt(torch.mean(torch.square(err)))
            # Compute the relative error / resolution and log it
            metrics[field + "_mean_res"] = torch.mean(err / target)
            metrics[field + "_std_res"] = torch.std(err / target)

        return metrics

    def scale_proxy_feats(self, proxy_feats: Tensor):
        return torch.cat([self.scaler[field].transform(proxy_feats[..., i]).unsqueeze(-1) for i, field in enumerate(self.fields)], -1)

    def get_proxy_feats(
        self,
        incidence: Tensor,
        inputs: dict[str, Tensor],
        class_probs: Tensor,
    ):
        proxy_feats = torch.cat(
            [inputs[self.input_hit + "_" + field].unsqueeze(-1) for field in self.fields],
            axis=-1,
        )

        charged_inc = incidence * inputs[self.input_hit + "_is_track"].unsqueeze(1)
        # Use the most weighted track as proxy for charged particles
        charged_inc_top2 = (topk_attn(charged_inc, 2, dim=-2) & (charged_inc > 0)).float()
        charged_inc_max = charged_inc.max(-2, keepdim=True)[0]
        charged_inc_new = (charged_inc == charged_inc_max) & (charged_inc > 0)
        # TODO: check this
        # charged_inc_new = charged_inc.float()
        zero_track_mask = charged_inc_new.sum(-1, keepdim=True) == 0
        charged_inc = torch.where(zero_track_mask, charged_inc_top2, charged_inc_new)

        # Split charged and neutral
        is_charged = class_probs.argmax(-1) < 3

        proxy_feats_charged = torch.bmm(charged_inc, proxy_feats)
        proxy_feats_charged[..., 0] = proxy_feats_charged[..., 1] * torch.cosh(proxy_feats_charged[..., 2])
        proxy_feats_charged = self.scale_proxy_feats(proxy_feats_charged) * is_charged.unsqueeze(-1)

        inc_e_weighted = incidence * proxy_feats[..., 0].unsqueeze(1)
        inc_e_weighted = inc_e_weighted * (1 - inputs[self.input_hit + "_is_track"].unsqueeze(1))
        inc = inc_e_weighted / (inc_e_weighted.sum(dim=-1, keepdim=True) + 1e-6)

        proxy_feats_neutral = torch.einsum("bnf,bpn->bpf", proxy_feats, inc)
        proxy_feats_neutral[..., 0] = inc_e_weighted.sum(-1)
        proxy_feats_neutral[..., 1] = proxy_feats_neutral[..., 0] / torch.cosh(proxy_feats_neutral[..., 2])

        proxy_feats_neutral = self.scale_proxy_feats(proxy_feats_neutral) * (~is_charged).unsqueeze(-1)
        proxy_feats = proxy_feats_charged + proxy_feats_neutral

        return proxy_feats, is_charged
