import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, loss_fns, mask_focal_loss
from hepattn.utils.masks import topk_attn
from hepattn.utils.scaling import FeatureScaler

# Mapping of loss function names to torch.nn.functional loss functions
REGRESSION_LOSS_FNS = {
    "l1": torch.nn.functional.l1_loss,
    "l2": torch.nn.functional.mse_loss,
    "smooth_l1": torch.nn.functional.smooth_l1_loss,
}

# Define the literal type for regression losses based on the dictionary keys
RegressionLossType = Literal["l1", "l2", "smooth_l1"]


class Task(nn.Module, ABC):
    """Abstract base class for all tasks.

    A task represents a specific learning objective (e.g., classification, regression)
    that can be trained as part of a multi-task learning setup.
    """

    def __init__(self, has_intermediate_loss: bool, permute_loss: bool = True):
        super().__init__()
        self.has_intermediate_loss = has_intermediate_loss
        self.permute_loss = permute_loss

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

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def key_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def query_mask(self, outputs: dict[str, Tensor], **kwargs) -> Tensor | None:
        return None

    def loss_kwargs(self, outputs: dict[str, dict[str, Tensor]], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}


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
        has_intermediate_loss: bool = True,
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
        super().__init__(has_intermediate_loss=has_intermediate_loss)

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

    def predict(self, outputs: dict[str, Tensor], threshold: float = 0.5) -> dict[str, Tensor]:
        # Objects that have a predicted probability above the threshold are marked as predicted to exist
        return {self.output_object + "_valid": outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_logit"].detach().to(torch.float32)
        target = targets[self.target_object + "_valid"].to(torch.float32)
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses = {}
        output = outputs[self.output_object + "_logit"]
        target = targets[self.target_object + "_valid"].type_as(output)
        sample_weight = target + self.null_weight * (1 - target)
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, sample_weight=sample_weight)
        return losses

    def query_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> Tensor | None:
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
        has_intermediate_loss: bool = True,
    ):
        """Task used for classifying whether hits belong to reconstructable objects or not.

        Parameters
        ----------
        name : str
            Name of the task.
        hit_name : str
            Name of the hit object type.
        target_field : str
            Name of the target field to predict.
        dim : int
            Embedding dimension.
        threshold : float, optional
            Threshold for classification, by default 0.1.
        mask_keys : bool, optional
            Whether to mask keys, by default False.
        loss_fn : Literal["bce", "focal", "both"], optional
            Loss function to use, by default "bce".
        has_intermediate_loss : bool, optional
            Whether task has intermediate loss, by default True.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)

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

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return {f"{self.hit_name}_{self.target_field}": outputs[f"{self.hit_name}_logit"].sigmoid() >= self.threshold}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
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

    def key_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> dict[str, Tensor]:
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
        logit_scale: float = 1.0,
        pred_threshold: float = 0.5,
        has_intermediate_loss: bool = True,
    ):
        """Task for predicting associations between objects and hits.

        Parameters
        ----------
        name : str
            Name of the task.
        input_hit : str
            Name of the input hit object.
        input_object : str
            Name of the input object.
        output_object : str
            Name of the output object.
        target_object : str
            Name of the target object.
        losses : dict[str, float]
            Loss functions and their weights.
        costs : dict[str, float]
            Cost functions and their weights.
        dim : int
            Embedding dimension.
        null_weight : float, optional
            Weight for null class, by default 1.0.
        mask_attn : bool, optional
            Whether to mask attention, by default True.
        target_field : str, optional
            Target field name, by default "valid".
        logit_scale : float, optional
            Scale for logits, by default 1.0.
        pred_threshold : float, optional
            Prediction threshold, by default 0.5.
        has_intermediate_loss : bool, optional
            Whether task has intermediate loss, by default True.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

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
        self.logit_scale = logit_scale
        self.pred_threshold = pred_threshold
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
        object_hit_logit = self.logit_scale * torch.einsum("bnc,bmc->bnm", x_object, x_hit)

        # Zero out entries for any hit slots that are not valid
        object_hit_logit[~x[self.input_hit + "_valid"].unsqueeze(-2).expand_as(object_hit_logit)] = torch.finfo(object_hit_logit.dtype).min

        return {self.output_object_hit + "_logit": object_hit_logit}

    def attn_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> dict[str, Tensor]:
        if not self.mask_attn:
            return {}

        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= threshold

        # If the attn mask is completely padded for a given entry, unpad it - tested and is required (?)
        # TODO: See if the query masking stops this from being necessary
        attn_mask[torch.where(torch.all(attn_mask, dim=-1))] = False

        return {self.input_hit: attn_mask}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Object-hit pairs that have a predicted probability above the threshold are predicted as being associated to one-another
        return {self.output_object_hit + "_valid": outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= self.pred_threshold}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object_hit + "_logit"].detach().to(torch.float32)
        target = targets[self.target_object_hit + "_" + self.target_field].detach().to(output.dtype)

        hit_pad = targets[self.input_hit + "_valid"]

        costs = {}
        # sample_weight = target + self.null_weight * (1 - target)
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target, input_pad_mask=hit_pad)
        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
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
        loss: RegressionLossType = "smooth_l1",
        has_intermediate_loss: bool = True,
    ):
        """Base class for regression tasks.

        Parameters
        ----------
        name : str
            Name of the task.
        output_object : str
            Name of the output object.
        target_object : str
            Name of the target object.
        fields : list[str]
            List of fields to regress.
        loss_weight : float
            Weight for the loss function.
        cost_weight : float
            Weight for the cost function.
        loss : RegressionLossType, optional
            Type of loss function to use, by default "smooth_l1".
        has_intermediate_loss : bool, optional
            Whether task has intermediate loss, by default True.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.cost_weight = cost_weight
        self.loss_fn_name = loss
        self.loss_fn = REGRESSION_LOSS_FNS[loss]
        self.k = len(fields)
        # For standard regression number of DoFs is just the number of targets
        self.ndofs = self.k

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        latent = self.latent(x)
        return {self.output_object + "_regr": latent}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Split the regression vector into the separate fields
        latent = outputs[self.output_object + "_regr"]
        return {self.output_object + "_" + field: latent[..., i] for i, field in enumerate(self.fields)}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        # Compute the loss
        loss = self.loss_fn(output, target, reduction="none")

        # Average over all the objects
        loss = torch.mean(loss, dim=-1)

        # Compute the regression loss only for valid objects
        return {self.loss_fn_name: self.loss_weight * loss.mean()}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        metrics = {}
        for field in self.fields:
            # note these might be scaled features
            pred = preds[self.output_object + "_" + field][targets[self.target_object + "_valid"]]
            target = targets[self.target_object + "_" + field][targets[self.target_object + "_valid"]]
            abs_err = (pred - target).abs()
            metrics[field + "_abs_res"] = torch.mean(abs_err)
            metrics[field + "_abs_norm_res"] = torch.mean(abs_err / target.abs() + 1e-8)
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
        has_intermediate_loss: bool = True,
    ):
        """Regression task with Gaussian output distribution.

        Parameters
        ----------
        name : str
            Name of the task.
        output_object : str
            Name of the output object.
        target_object : str
            Name of the target object.
        fields : list[str]
            List of fields to regress.
        loss_weight : float
            Weight for the loss function.
        cost_weight : float
            Weight for the cost function.
        has_intermediate_loss : bool, optional
            Whether task has intermediate loss, by default True.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

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
        log_likelihood *= targets[self.target_object + "_valid"].type_as(log_likelihood)
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
        """Gaussian regression task for objects.

        Parameters
        ----------
        name : str
            Name of the task.
        input_object : str
            Name of the input object.
        output_object : str
            Name of the output object.
        target_object : str
            Name of the target object.
        fields : list[str]
            List of fields to regress.
        loss_weight : float
            Weight for the loss function.
        cost_weight : float
            Weight for the cost function.
        dim : int
            Embedding dimension.
        """
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

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
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
        log_likelihood *= targets[f"{self.target_object}_valid"].unsqueeze(1).type_as(log_likelihood)
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
        loss: RegressionLossType = "smooth_l1",
        has_intermediate_loss: bool = True,
    ):
        """Regression task for objects.

        Parameters
        ----------
        name : str
            Name of the task.
        input_object : str
            Name of the input object.
        output_object : str
            Name of the output object.
        target_object : str
            Name of the target object.
        fields : list[str]
            List of fields to regress.
        loss_weight : float
            Weight for the loss function.
        cost_weight : float
            Weight for the cost function.
        dim : int
            Embedding dimension.
        loss : RegressionLossType, optional
            Type of loss function to use, by default "smooth_l1".
        has_intermediate_loss : bool, optional
            Whether task has intermediate loss, by default True.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1).to(torch.float32)
        num_objects = output.shape[1]
        # Index from the front so it works for both object and mask regression
        # The expand is not necessary but stops a broadcasting warning from smooth_l1_loss
        costs = self.loss_fn(
            output.unsqueeze(2).expand(-1, -1, num_objects, -1),
            target.unsqueeze(1).expand(-1, num_objects, -1, -1),
            reduction="none",
        )
        # Average over the regression fields dimension
        costs = costs.mean(-1)
        return {f"regr_{self.loss_fn_name}": self.cost_weight * costs}


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
        loss: RegressionLossType = "smooth_l1",
        has_intermediate_loss: bool = True,
    ):
        """Regression task for object-hit associations.

        Parameters
        ----------
        name : str
            Name of the task.
        input_hit : str
            Name of the input hit object.
        input_object : str
            Name of the input object.
        output_object : str
            Name of the output object.
        target_object : str
            Name of the target object.
        fields : list[str]
            List of fields to regress.
        loss_weight : float
            Weight for the loss function.
        cost_weight : float
            Weight for the cost function.
        dim : int
            Embedding dimension.
        loss : RegressionLossType, optional
            Type of loss function to use, by default "smooth_l1".
        has_intermediate_loss : bool, optional
            Whether task has intermediate loss, by default True.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

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
        x_obj_hit *= x[self.input_hit + "_valid"].unsqueeze(-2).unsqueeze(-1).expand_as(x_obj_hit).float()
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
        has_intermediate_loss: bool = True,
    ):
        """Classification task for objects.

        Parameters
        ----------
        name : str
            Name of the task.
        input_object : str
            Name of the input object.
        output_object : str
            Name of the output object.
        target_object : str
            Name of the target object.
        classes : list[str]
            List of class names.
        dim : int
            Embedding dimension.
        class_weights : dict[str, float] | None, optional
            Weights for each class, by default None.
        loss_weight : float, optional
            Weight for the loss function, by default 1.0.
        multilabel : bool, optional
            Whether this is a multilabel classification, by default False.
        permute_loss : bool, optional
            Whether to permute loss, by default True.
        has_intermediate_loss : bool, optional
            Whether task has intermediate loss, by default True.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=permute_loss)

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

        if self.class_weights is not None:
            self.class_weights_values = torch.tensor([class_weights[class_name] for class_name in self.classes])

        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_logits"]

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Now get the class logits from the embedding (..., N, ) -> (..., E)
        x = self.class_net(x[f"{self.input_object}_embed"])
        return {f"{self.output_object}_logits": x}

    def predict(self, outputs: dict[str, Tensor], threshold: float = 0.5) -> dict[str, Tensor]:
        # Split the regression vector into the separate fields
        logits = outputs[self.output_object + "_logits"].detach()
        if self.multilabel:
            predictions = torch.nn.functional.sigmoid(logits) >= threshold
        else:
            predictions = torch.nn.functional.one_hot(torch.argmax(logits, dim=-1), num_classes=len(self.classes))
        return {self.output_object + "_" + class_name: predictions[..., i] for i, class_name in enumerate(self.classes)}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
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

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        metrics = {}
        for class_name in self.classes:
            target = targets[f"{self.target_object}_{class_name}"][targets[f"{self.target_object}_valid"]].bool()
            pred = preds[f"{self.output_object}_{class_name}"][targets[f"{self.target_object}_valid"]].bool()

            metrics[f"{class_name}_eff"] = (target & pred).sum() / target.sum()
            metrics[f"{class_name}_pur"] = (target & pred).sum() / pred.sum()

        return metrics


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
        has_intermediate_loss: bool = True,
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
            while value denotes cost weight.
        net : nn.Module
            Network that will be used to classify the object classes.
        null_weight : float
            Weight applied to the null class in the loss. Useful if many instances of
            the target class are null, and we need to reweight to overcome class imbalance.

        Raises:
            ValueError: If the input arguments are invalid.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

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
        # Network projects the embedding down into a class probability
        x_class_prob = self.net(x[self.input_object + "_embed"])
        return {self.output_object + "_class_prob": x_class_prob}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        classes = outputs[self.output_object + "_class_prob"].detach().argmax(-1)
        return {
            self.output_object + "_class": classes,
            self.output_object + "_valid": classes < self.num_classes,  # Valid if class is less than num_classes
        }

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_class_prob"].detach().to(torch.float32)
        target = targets[self.target_object + "_class"].long()
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses = {}
        output = outputs[self.output_object + "_class_prob"]
        target = targets[self.target_object + "_class"].long()
        # Calculate the loss from each specified loss function.
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=None, weight=self.class_weights)
        return losses

    def query_mask(self, outputs: dict[str, Tensor]) -> Tensor | None:
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
        has_intermediate_loss: bool = True,
    ):
        """Incidence regression task."""
        super().__init__(has_intermediate_loss=has_intermediate_loss)
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

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_incidence"].detach().to(torch.float32)
        target = targets[self.target_object + "_incidence"].to(torch.float32)

        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses = {}
        output = outputs[self.output_object + "_incidence"]
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
        loss: RegressionLossType = "smooth_l1",
        use_incidence: bool = True,
        use_nodes: bool = False,
        use_pt_match: bool = False,
        split_charge_neutral_loss: bool = False,
        has_intermediate_loss: bool = True,
    ):
        """Regression task that uses incidence information to predict regression targets.

        Parameters
        ----------
        targets : list
            List of target names
        add_momentum : bool
            Whether to add scalar momentum to the predictions, computed from the px, py, pz predictions
        loss : RegressionLossType, optional
            Type of loss function to use, by default "smooth_l1".
        """
        super().__init__(
            name=name,
            output_object=output_object,
            target_object=target_object,
            fields=fields,
            loss_weight=loss_weight,
            cost_weight=cost_weight,
            loss=loss,
            has_intermediate_loss=has_intermediate_loss,
        )
        self.input_hit = input_hit
        self.input_object = input_object
        self.scaler = FeatureScaler(scale_dict_path=scale_dict_path)
        self.use_incidence = use_incidence
        self.cost_weight = cost_weight
        self.net = net
        self.split_charge_neutral_loss = split_charge_neutral_loss
        self.use_nodes = use_nodes
        self.use_pt_match = use_pt_match

        self.loss_masks = {
            "e": self.get_neutral,  # Only neutral particles
            "pt": self.get_charged,  # Only charged particles
        }

        self.inputs = [input_object + "_embed"] + [input_hit + "_" + field for field in fields]
        self.outputs = [output_object + "_regr", output_object + "_proxy_regr"]

    """def loss_kwargs(self, outputs: dict[str, dict[str, Tensor]], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        # Adding this to get access to the classification task output
        classification = outputs.get("classification")

        if classification is None:
            return {"output_class": None}

        class_prob = classification.get(self.output_object + "_class_prob")
        if class_prob is None:
            return {"output_class": None}

        return {"output_class": class_prob.detach().argmax(-1)}"""

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
        preds = self.net(input_data) + proxy_feats
        return {self.output_object + "_regr": preds, self.output_object + "_proxy_regr": proxy_feats}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Split the regression vector into the separate fields
        pflow_regr = outputs[self.output_object + "_regr"]
        proxy_regr = outputs[self.output_object + "_proxy_regr"]
        return {self.output_object + "_" + field: pflow_regr[..., i] for i, field in enumerate(self.fields)} | {
            self.output_object + "_proxy_" + field: proxy_regr[..., i] for i, field in enumerate(self.fields)
        }

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        metrics = super().metrics(preds, targets)
        # Add metrics for the proxy regression
        for field in self.fields:
            # note these might be scaled features
            pred = preds[self.output_object + "_proxy_" + field][targets[self.target_object + "_valid"]]
            target = targets[self.target_object + "_" + field][targets[self.target_object + "_valid"]]
            abs_err = (pred - target).abs()
            metrics[field + "_proxy_abs_res"] = abs_err.mean()
            metrics[field + "_proxy_abs_norm_res"] = torch.mean(abs_err / target.abs() + 1e-8)
        return metrics

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
        if self.use_pt_match:
            pred_pt = outputs[self.output_object + "_regr"][..., self.fields.index("pt")][:, :, None]
            target_pt = targets[self.target_object + "_pt"][:, None, :]
            pt_cost = (target_pt - pred_pt) ** 2 / (target_pt**2 + 1e-8)
        else:
            pt_cost = 0
        # Compute the cost as the sum of the squared differences
        cost = self.cost_weight * torch.sqrt(pt_cost + dphi**2 + deta**2)
        return {"regression": cost}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], output_class: Tensor | None = None) -> dict[str, Tensor]:
        if self.split_charge_neutral_loss and output_class is None:
            raise RuntimeError("'output_class' is empty for the IncidenceBasedRegressionTask.")

        loss = None
        target_class = targets[self.target_object + "_class"]
        # output_class = outputs["classification"][self.output_object + "_class_prob"].detach().argmax(-1)
        for i, field in enumerate(self.fields):
            target = targets[self.target_object + "_" + field]
            output = outputs[self.output_object + "_regr"][..., i]
            mask = targets[self.target_object + "_valid"].clone()
            if self.split_charge_neutral_loss and field in self.loss_masks:
                mask &= self.loss_masks[field](output_class, target_class)
            if loss is None:
                loss = self.loss_fn(output[mask], target[mask], reduction="mean")
            else:
                loss += self.loss_fn(output[mask], target[mask], reduction="mean")
        # Average over all the features
        loss /= len(self.fields)

        # Compute the regression loss only for valid objects
        return {self.loss_fn_name: self.loss_weight * loss}

    def scale_proxy_feats(self, proxy_feats: Tensor):
        return torch.cat([self.scaler[field].transform(proxy_feats[..., i]).unsqueeze(-1) for i, field in enumerate(self.fields)], -1)

    def get_proxy_feats(
        self,
        incidence: Tensor,
        inputs: dict[str, Tensor],
        class_probs: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
        inc_e_weighted *= 1 - inputs[self.input_hit + "_is_track"].unsqueeze(1)
        inc = inc_e_weighted / (inc_e_weighted.sum(dim=-1, keepdim=True) + 1e-6)

        proxy_feats_neutral = torch.einsum("bnf,bpn->bpf", proxy_feats, inc)
        proxy_feats_neutral[..., 0] = inc_e_weighted.sum(-1)
        proxy_feats_neutral[..., 1] = proxy_feats_neutral[..., 0] / torch.cosh(proxy_feats_neutral[..., 2])

        proxy_feats_neutral = self.scale_proxy_feats(proxy_feats_neutral) * (~is_charged).unsqueeze(-1)
        proxy_feats = proxy_feats_charged + proxy_feats_neutral

        return proxy_feats, is_charged
