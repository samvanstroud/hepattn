import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, loss_fns, mask_focal_loss


class Task(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.has_intermediate_loss = False

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
            Name of the input object object
        output_object : str
            Name of the output object object which will denote if the predicted object slot is used or not.
        target_object: str
            Name of the target object object that we want to predict is valid or not.
        losses : dict[str, float]
            Dict specifying which losses to use. Keys are loss function name and values are loss weights.
        costs : dict[str, float]
            Dict specifying which costs to use. Keys are cost function name and values are cost weights.
        dim : int
            Embedding dimension of the input objects.
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
        return costs

    def loss(self, outputs, targets):
        losses = {}
        output = outputs[self.output_object + "_logit"]
        target = targets[self.target_object + "_valid"].type_as(output)
        sample_weight = target + self.null_weight * (1 - target)
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, sample_weight=sample_weight)
        return losses

    def query_mask(self, outputs, threshold=0.1):
        if not self.mask_queries:
            return None

        return outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold


class HitFilterTask(Task):
    def __init__(
        self,
        name: str,
        hit_name: str,
        target_field: str,
        dim: int,
        threshold: float = 0.1,
        mask_keys: bool = False,
        loss_fn: Literal["bce", "focal", "both"] = "bce",
    ):
        """Task used for classifying whether hits belong to reconstructable objects or not."""
        super().__init__()

        self.name = name
        self.hit_name = hit_name
        self.target_field = target_field
        self.dim = dim
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.mask_keys = mask_keys

        # Internal
        self.hit_names = [f"{hit_name}_embed"]
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_logit = self.net(x[f"{self.hit_name}_embed"])
        return {f"{self.hit_name}_logit": x_logit.squeeze(-1)}

    def predict(self, outputs: dict) -> dict:
        return {f"{self.hit_name}_{self.target_field}": outputs[f"{self.hit_name}_logit"].sigmoid() >= self.threshold}

    def loss(self, outputs: dict, targets: dict) -> dict:
        # Pick out the field that denotes whether a hit is on a reconstructable object or not
        output = outputs[f"{self.hit_name}_logit"]
        target = targets[f"{self.hit_name}_{self.target_field}"].type_as(output)

        # Calculate the BCE loss with class weighting
        if self.loss_fn == "bce":
            pos_weight = 1 / target.float().mean()
            loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
            return {f"{self.hit_name}_{self.loss_fn}": loss}
            weight = 1 / target.float().mean()
            loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
            return {f"{self.hit_name}_{self.loss_fn}": loss}
        if self.loss_fn == "focal":
            loss = mask_focal_loss(output, target)
            return {f"{self.hit_name}_{self.loss_fn}": loss}
        if self.loss_fn == "both":
            pos_weight = 1 / target.float().mean()
            bce_loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
            focal_loss_value = mask_focal_loss(output, target)
            return {
                f"{self.hit_name}_bce": bce_loss,
                f"{self.hit_name}_focal": focal_loss_value,
            }
        raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def key_mask(self, outputs, threshold=0.1):
        if not self.mask_keys:
            return {}

        return {self.hit_name: outputs[f"{self.hit_name}_logit"].detach().sigmoid() >= threshold}


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
        target_field: str = "valid",
    ):
        super().__init__()

        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.target_field = target_field

        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight
        self.mask_attn = mask_attn
        self.has_intermediate_loss = mask_attn

        self.output_object_hit = output_object + "_" + input_hit
        self.target_object_hit = target_object + "_" + input_hit
        self.inputs = [input_object + "_embed", input_hit + "_embed"]
        self.outputs = [self.output_object_hit + "_logit"]
        self.hit_net = Dense(dim, dim)
        self.object_net = Dense(dim, dim)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Produce new task-specific embeddings for the hits and objects
        x_object = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_hit + "_embed"])

        # Object-hit probability is the dot product between the hit and object embedding
        object_hit_logit = torch.einsum("bnc,bmc->bnm", x_object, x_hit)

        # Zero out entries for any hit slots that are not valid
        object_hit_logit[~x[self.input_hit + "_valid"].unsqueeze(-2).expand_as(object_hit_logit)] = torch.finfo(object_hit_logit.dtype).min

        return {self.output_object_hit + "_logit": object_hit_logit}

    def attn_mask(self, outputs, threshold=0.1):
        if not self.mask_attn:
            return {}

        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= threshold

        # If the attn mask is completely padded for a given entry, unpad it - tested and is required (?)
        # TODO: See if the query masking stops this from being necessary
        attn_mask[torch.where(torch.all(attn_mask, dim=-1))] = False

        return {self.input_hit: attn_mask}

    def predict(self, outputs, threshold=0.5):
        # Object-hit pairs that have a predicted probability above the threshold are predicted as being associated to one-another
        return {self.output_object_hit + "_valid": outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= threshold}

    def cost(self, outputs, targets):
        output = outputs[self.output_object_hit + "_logit"].detach().to(torch.float32)
        target = targets[self.target_object_hit + "_" + self.target_field].to(torch.float32)

        costs = {}
        # sample_weight = target + self.null_weight * (1 - target)
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    def loss(self, outputs, targets):
        output = outputs[self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_" + self.target_field].type_as(output)

        hit_pad = targets[self.input_hit + "_valid"]
        object_pad = targets[self.target_object + "_valid"]

        sample_weight = target + self.null_weight * (1 - target)
        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](
                output, target, object_valid_mask=object_pad, input_pad_mask=hit_pad, sample_weight=sample_weight
            )
        return losses


class RegressionTask(Task):
    def __init__(
        self,
        name: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
    ):
        super().__init__()

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.cost_weight = cost_weight
        self.k = len(fields)
        # For standard regression number of DoFs is just the number of targets
        self.ndofs = self.k

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        latent = self.latent(x)
        return {self.output_object + "_regr": latent}

    def predict(self, outputs):
        # Split the regression vectior into the separate fields
        latent = outputs[self.output_object + "_regr"]
        return {self.output_object + "_" + field: latent[..., i] for i, field in enumerate(self.fields)}

    def loss(self, outputs, targets):
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        # Compute the loss
        loss = torch.nn.functional.smooth_l1_loss(output, target, reduction="none")

        # Average over all the objects
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
            metrics[field + "_mean_rel_err"] = torch.mean(err / target)
            metrics[field + "_std_rel_err"] = torch.std(err / target)

        return metrics


class GaussianRegressionTask(Task):
    def __init__(
        self,
        name: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
    ):
        super().__init__()

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.cost_weight = cost_weight
        self.k = len(fields)
        # For multivaraite gaussian case we have extra DoFs from the variance and covariance terms
        self.ndofs = self.k + int(self.k * (self.k + 1) / 2)
        self.likelihood_norm = self.k * 0.5 * math.log(2 * math.pi)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        latent = self.latent(x)
        k = self.k
        triu_idx = torch.triu_indices(k, k, device=latent.device)

        # Mean vector
        mu = latent[..., :k]
        # Upper-diagonal Cholesky decomposition of the precision matrix
        u = torch.zeros(latent.size()[:-1] + torch.Size((k, k)), device=latent.device)
        u[..., triu_idx[0, :], triu_idx[1, :]] = latent[..., k:]

        ubar = u.clone()
        # Make sure the diagonal entries are positive (as variance is always positive)
        ubar[..., torch.arange(k), torch.arange(k)] = torch.exp(u[..., torch.arange(k), torch.arange(k)])

        return {self.output_object + "_mu": mu, self.output_object + "_u": u, self.output_object + "_ubar": ubar}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        preds = outputs
        mu = outputs[self.output_object + "_mu"]
        ubar = outputs[self.output_object + "_ubar"]

        # Calculate the precision matrix
        precs = torch.einsum("...kj,...kl->...jl", ubar, ubar)

        # Get the predicted mean for each field
        for i, field in enumerate(self.fields):
            preds[self.output_object + "_" + field] = mu[..., i]

        # Get the predicted precision for each field and the predicted covariance / coprecision
        for i, field_i in enumerate(self.fields):
            for j, field_j in enumerate(self.fields):
                if i > j:
                    continue
                preds[field_i + "_" + field_j + "_prec"] = precs[..., i, j]

        return preds

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)

        # Compute the standardised score vector between the targets and the predicted distribution paramaters
        z = torch.einsum("...ij,...j->...i", outputs[self.output_object + "_ubar"], y - outputs[self.output_object + "_mu"])
        # Compute the NLL from the score vector
        zsq = torch.einsum("...i,...i->...", z, z)
        jac = torch.sum(torch.diagonal(outputs[self.output_object + "_u"], offset=0, dim1=-2, dim2=-1), dim=-1)
        log_likelihood = self.likelihood_norm - 0.5 * zsq + jac

        # Only compute NLL for valid tracks or track-hit pairs
        # nll = nll[targets[self.target_object + "_valid"]]
        log_likelihood = log_likelihood * targets[self.target_object + "_valid"].type_as(log_likelihood)
        # Take the average and apply the task weight
        return {"nll": -self.loss_weight * log_likelihood.mean()}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)  # Point target
        res = y - preds[self.output_object + "_mu"]  # Residual
        z = torch.einsum("...ij,...j->...i", preds[self.output_object + "_ubar"], res)  # Scaled resdiaul / z score

        # Select only values that havea valid target
        valid_mask = targets[self.target_object + "_valid"]

        metrics = {}
        for i, field in enumerate(self.fields):
            metrics[field + "_rmse"] = torch.sqrt(torch.mean(torch.square(res[..., i][valid_mask])))
            # The mean and standard deviation of the pulls to check predictions are calibrated
            metrics[field + "_pull_mean"] = torch.mean(z[..., i][valid_mask])
            metrics[field + "_pull_std"] = torch.std(z[..., i][valid_mask])

        return metrics


class ObjectGaussianRegressionTask(GaussianRegressionTask):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
        dim: int,
    ):
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [
            output_object + "_mu",
            output_object + "_ubar",
            output_object + "_u",
        ]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def cost(self, outputs, targets):
        mu = outputs[self.output_object + "_mu"].to(torch.float32)  # (B, N, D)
        ubar = outputs[self.output_object + "_ubar"].to(torch.float32)  # (B, N, D, D)
        u = outputs[self.output_object + "_u"].to(torch.float32)
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1).to(torch.float32)  # (B, N, D)

        # Now we need compute the Gaussian NLL for every target/pred pair, remember costs have shape (batch, pred, true)
        num_objects = y.shape[1]  # num_objects = N
        mu = mu.unsqueeze(2).expand(-1, -1, num_objects, -1)  # (B, N, N, D)
        ubar = ubar.unsqueeze(2).expand(-1, -1, num_objects, -1, -1)  # (B, N, N, D, D)
        u = u.unsqueeze(2).expand(-1, -1, num_objects, -1, -1)
        diagu = torch.diagonal(u, offset=0, dim1=-2, dim2=-1)  # (B, N, N, D)
        y = y.unsqueeze(1).expand(-1, num_objects, -1, -1)  # (B, N, N, D)

        # Compute the standardised score vector between the targets and the predicted distribution paramaters
        z = torch.einsum("...ij,...j->...i", ubar, y - mu)  # (B, N, N, D)
        # Compute the NLL from the score vector
        zsq = torch.einsum("...i,...i->...", z, z)  # (B, N, N)
        jac = torch.sum(diagu, dim=-1)  # (B, N, N)

        log_likelihood = self.likelihood_norm - 0.5 * zsq + jac
        log_likelihood = log_likelihood * targets[f"{self.target_object}_valid"].unsqueeze(1).type_as(log_likelihood)
        costs = -log_likelihood

        return {"nll": self.cost_weight * costs}


class ObjectRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
        dim: int,
    ):
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def cost(self, outputs, targets):
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1).to(torch.float32)
        num_objects = output.shape[1]
        # Index from the front so it works for both object and mask regression
        # The expand is not necessary but stops a broadcasting warning from smooth_l1_loss
        costs = torch.nn.functional.smooth_l1_loss(
            output.unsqueeze(2).expand(-1, -1, num_objects, -1),
            target.unsqueeze(1).expand(-1, num_objects, -1, -1),
            reduction="none",
        )
        # Average over the regression fields dimension
        costs = costs.mean(-1)
        return {"regr_smooth_l1": self.cost_weight * costs}


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
        cost_weight: float,
        dim: int,
    ):
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight)

        self.input_hit = input_hit
        self.input_object = input_object

        self.inputs = [input_object + "_embed", input_hit + "_embed"]
        self.outputs = [self.output_object + "_regr"]

        self.dim = dim
        self.dim_per_dof = self.dim // self.ndofs

        self.hit_net = Dense(dim, self.ndofs * self.dim_per_dof)
        self.object_net = Dense(dim, self.ndofs * self.dim_per_dof)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        # Embed the hits and tracks and reshape so we have a separate embedding for each DoF
        x_obj = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_hit + "_embed"])

        x_obj = x_obj.reshape(x_obj.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))  # Shape BNDE
        x_hit = x_hit.reshape(x_hit.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))  # Shape BMDE

        # Take the dot product between the hits and tracks over the last embedding dimension so we are left
        # with just a scalar for each degree of freedom
        x_obj_hit = torch.einsum("...nie,...mie->...nmi", x_obj, x_hit)  # Shape BNMD

        # Shape of padding goes BM -> B1M -> B1M1 -> BNMD
        x_obj_hit = x_obj_hit * x[self.input_hit + "_valid"].unsqueeze(-2).unsqueeze(-1).expand_as(x_obj_hit).float()
        return x_obj_hit


class ClassificationTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        classes: list[str],
        dim: int,
        class_weights: dict[str, float] | None = None,
        loss_weight: float = 1.0,
        multilabel: bool = False,
        permute_loss: bool = True,
    ):
        super().__init__()

        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.classes = classes
        self.dim = dim
        self.class_weights = class_weights
        self.loss_weight = loss_weight
        self.multilabel = multilabel
        self.class_net = Dense(dim, len(classes))
        self.permute_loss = permute_loss

        if self.class_weights is not None:
            self.class_weights_values = torch.tensor([class_weights[class_name] for class_name in self.classes])

        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_logits"]

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Now get the class logits from the embedding (..., N, ) -> (..., E)
        x = self.class_net(x[f"{self.input_object}_embed"])
        return {f"{self.output_object}_logits": x}

    def predict(self, outputs, threshold=0.5):
        # Split the regression vectior into the separate fields
        logits = outputs[self.output_object + "_logits"].detach()
        if self.multilabel:
            predictions = torch.nn.functional.sigmoid(logits) >= threshold
        else:
            predictions = torch.nn.functional.one_hot(torch.argmax(logits, dim=-1), num_classes=len(self.classes))
        return {self.output_object + "_" + class_name: predictions[..., i] for i, class_name in enumerate(self.classes)}

    def loss(self, outputs, targets):
        # Get the targets and predictions
        target = torch.stack([targets[self.target_object + "_" + class_name] for class_name in self.classes], dim=-1)
        logits = outputs[f"{self.output_object}_logits"]

        # Put the class weights into a tensor with the correct dtype
        class_weights = None
        if self.class_weights is not None:
            class_weights = self.class_weights_values.type_as(target)

        # Compute the loss, using the class weights
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target.view(-1, target.shape[-1]),
            weight=class_weights,
            reduction="none",
        )

        # Only consider valid targets
        losses = losses[targets[f"{self.target_object}_valid"].view(-1)]
        return {"bce": self.loss_weight * losses.mean()}

    def metrics(self, preds, targets):
        metrics = {}
        for class_name in self.classes:
            target = targets[f"{self.target_object}_{class_name}"][targets[f"{self.target_object}_valid"]].bool()
            pred = preds[f"{self.output_object}_{class_name}"][targets[f"{self.target_object}_valid"]].bool()

            metrics[f"{class_name}_eff"] = (target & pred).sum() / target.sum()
            metrics[f"{class_name}_pur"] = (target & pred).sum() / pred.sum()

        return metrics
