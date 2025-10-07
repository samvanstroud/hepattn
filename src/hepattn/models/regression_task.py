import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, loss_fns, mask_focal_loss
from hepattn.utils.masks import topk_attn
from hepattn.utils.scaling import FeatureScaler, RegressionTargetScaler

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
        self.req_attn_mask = False

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

        Args:
            name: Name of the task.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
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
        self.req_attn_mask = False

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        latent = self.latent(x)
        return {self.output_object + "_regr": latent}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Split the regression vector into the separate fields
        latent = outputs[self.output_object + "_regr"]
        return {self.output_object + "_" + field: latent[..., i] for i, field in enumerate(self.fields)}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
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

        Args:
            name: Name of the task.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            has_intermediate_loss: Whether the task has intermediate loss.
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

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
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

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
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

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
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

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
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


class ObjectHepformerRegressionTask(RegressionTask):
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
        add_momentum: bool = False,
        scaler: RegressionTargetScaler | None = None,
    ):
        """Regression task for objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)
        self.add_momentum = add_momentum

        if self.add_momentum:
            if all([t in self.fields for t in ["px", "py", "pz"]]):
                self.fields.append("p")
                self.i_px = self.fields.index("px")
                self.i_py = self.fields.index("py")
                self.i_pz = self.fields.index("pz")
            else:
                self.add_momentum = False

        self.scaler = scaler

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        latent = self.latent(x)
        # get the predictions
        if self.add_momentum:
            latent = self.add_momentum_to_preds(latent)
        return {self.output_object + "_regr": latent}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Produce per-field predictions, inverse-scaling to original units when a scaler is provided
        latent = outputs[self.output_object + "_regr"].detach()
        field_values: dict[str, Tensor] = {}
        for i, field in enumerate(self.fields):
            value = latent[..., i]
            if self.scaler is not None:
                value = self.scaler.inverse(field)(value)
            field_values[self.output_object + "_" + field] = value
        return field_values

    def add_momentum_to_preds(self, preds: Tensor):
        momentum = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        return torch.cat([preds, momentum.unsqueeze(-1)], dim=-1)

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        # Scale targets per field if a scaler is provided
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1).to(torch.float32)
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

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Stack and optionally scale targets to match network output space
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1)
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        # Compute the loss
        loss = self.loss_fn(output, target, reduction="none")
        loss = torch.mean(loss, dim=-1)
        return {self.loss_fn_name: self.loss_weight * loss.mean()}


class ObjectHitRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_constituent: str,
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
        """Regression task for object-constituent associations.

        Args:
            name: Name of the task.
            input_constituent: Name of the input constituent type (e.g. hits in tracking).
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_constituent = input_constituent
        self.input_object = input_object

        self.inputs = [input_object + "_embed", input_constituent + "_embed"]
        self.outputs = [self.output_object + "_regr"]

        self.dim = dim
        self.dim_per_dof = self.dim // self.ndofs

        self.hit_net = Dense(dim, self.ndofs * self.dim_per_dof)
        self.object_net = Dense(dim, self.ndofs * self.dim_per_dof)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        # Embed the hits and tracks and reshape so we have a separate embedding for each DoF
        x_obj = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_constituent + "_embed"])

        x_obj = x_obj.reshape(x_obj.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))  # Shape BNDE
        x_hit = x_hit.reshape(x_hit.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))  # Shape BMDE

        # Take the dot product between the hits and tracks over the last embedding dimension so we are left
        # with just a scalar for each degree of freedom
        x_obj_hit = torch.einsum("...nie,...mie->...nmi", x_obj, x_hit)  # Shape BNMD

        # Shape of padding goes BM -> B1M -> B1M1 -> BNMD
        x_obj_hit *= x[self.input_constituent + "_valid"].unsqueeze(-2).unsqueeze(-1).expand_as(x_obj_hit).float()
        return x_obj_hit


class ObjectSpecificHitsRegressionTask(RegressionTask):
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
        add_momentum: bool = False,
        scaler: RegressionTargetScaler | None = None,
        hit_var: str = "embed",
    ):
        """Regression task for objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_object = input_object
        self.inputs = [input_object + hit_var]
        self.outputs = [output_object + "_regr"]

        self.dim = 2 * dim
        self.net = Dense(self.dim, self.ndofs)
        self.add_momentum = add_momentum
        self.hit_var = hit_var

        if self.add_momentum:
            if all([t in self.fields for t in ["px", "py", "pz"]]):
                self.fields.append("p")
                self.i_px = self.fields.index("px")
                self.i_py = self.fields.index("py")
                self.i_pz = self.fields.index("pz")
            else:
                self.add_momentum = False
    
        self.scaler = scaler
        self.req_attn_mask = True

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        query_embed = x["query_embed"]
        hit_embed = x[f"hit_{self.hit_var}"]

        track_hit_valid = x["attn_mask_intermediate"]

        assert track_hit_valid is not None
        track_hit_assignment = track_hit_valid  # [batch, num_queries, num_hits]
        track_hit_embeds = self.get_hit_embeds(track_hit_assignment, hit_embed)

        # Combine query embeddings with hit embeddings
        combined_embed = torch.cat([query_embed, track_hit_embeds], dim=-1)

        latent = self.net(combined_embed)
        # get the predictions
        if self.add_momentum:
            latent = self.add_momentum_to_preds(latent)
        return {self.output_object + "_regr": latent}

    def get_hit_embeds(self, track_hit_assignment: Tensor, hit_embed: Tensor) -> Tensor:
        # Count number of hits per track
        num_hits_per_track = track_hit_assignment.sum(dim=-1)  # [batch, num_queries]

        # Initialize output tensor
        batch_size, num_queries, _ = track_hit_assignment.shape
        track_hit_embeds = torch.zeros(batch_size, num_queries, hit_embed.shape[-1], device=hit_embed.device)

        # For each track that has hits assigned, gather and average its hit embeddings
        for b in range(batch_size):
            for q in range(num_queries):
                if num_hits_per_track[b, q] > 0:
                    # Get indices of assigned hits
                    hit_indices = torch.where(track_hit_assignment[b, q])[0]

                    # Gather embeddings for all assigned hits
                    track_hits = hit_embed[b, hit_indices]  # [num_hits, embed_dim]

                    # Average the embeddings
                    track_hit_embeds[b, q] = track_hits.mean(dim=0)

        return track_hit_embeds

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Produce per-field predictions, inverse-scaling to original units when a scaler is provided
        latent = outputs[self.output_object + "_regr"].detach()
        field_values: dict[str, Tensor] = {}
        for i, field in enumerate(self.fields):
            value = latent[..., i]
            if self.scaler is not None:
                value = self.scaler.inverse(field)(value)
            field_values[self.output_object + "_" + field] = value
        return field_values

    def add_momentum_to_preds(self, preds: Tensor):
        momentum = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        return torch.cat([preds, momentum.unsqueeze(-1)], dim=-1)

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        # Scale targets per field if a scaler is provided
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1).to(torch.float32)
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

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Stack and optionally scale targets to match network output space
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1)
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        # Compute the loss
        loss = self.loss_fn(output, target, reduction="none")
        loss = torch.mean(loss, dim=-1)
        return {self.loss_fn_name: self.loss_weight * loss.mean()}


class ObjectSpecificHitsPhiRegressionTask(RegressionTask):
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
        add_momentum: bool = False,
        scaler: RegressionTargetScaler | None = None,
        add_phi = True,
        query_embed = True,
        hit_embed = True,
    ):
        """Regression task for objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = dim + 2
        self.net = Dense(self.dim, self.ndofs)
        self.add_momentum = add_momentum

        self.scaler = scaler

        self.add_phi = add_phi
        self.query_embed = query_embed
        self.hit_embed = hit_embed

        if add_phi and ("sinphi" in self.fields) and ("cosphi" in self.fields):
            self.fields.append("tanphi")
            self.i_sinphi = self.fields.index("sinphi")
            self.i_cosphi = self.fields.index("cosphi")

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        query_embed = x["query_embed"]
        # hit_embed = x["hit_embed"]
        query_phi = x["query_phi"]  # shape [batch, num_queries]
        hit_phi = x["hit_phi"]  # shape [batch, num_hits]

        track_hit_valid = x["attn_mask_intermediate"]
        assert track_hit_valid is not None
        track_hit_assignment = track_hit_valid  # [batch, num_queries, num_hits]
        track_hit_phi = self.get_hit_phi(track_hit_assignment, hit_phi)

        # Ensure all tensors have the same number of dimensions
        # query_embed should be [batch, num_queries, embed_dim]
        # query_phi should be [batch, num_queries]
        # track_hit_phi should be [batch, num_queries]

        # Fix tensor dimensions for concatenation
        # query_embed: [1, 2100, 256] - correct 3D shape
        # query_phi: [2100] - needs to be [1, 2100] (add batch dimension)
        # track_hit_phi: [1, 2100] - correct 2D shape

        # Add batch dimension to query_phi if it's 1D
        if query_phi.dim() == 1:
            query_phi = query_phi.unsqueeze(0)  # [2100] -> [1, 2100]

        # Ensure all tensors are 3D for concatenation
        query_phi_expanded = query_phi.unsqueeze(-1)  # [1, 2100] -> [1, 2100, 1]
        track_hit_phi_expanded = track_hit_phi.unsqueeze(-1)  # [1, 2100] -> [1, 2100, 1]

        # Combine query embeddings with scalar phi features
        combined_embed = torch.cat(
            [query_embed, query_phi_expanded, track_hit_phi_expanded],
            dim=-1,
        )  # [1, 2100, 256+1+1] = [1, 2100, 258]

        latent = self.net(combined_embed)
        if self.add_phi:
            latent = self.add_tanphi_to_preds(latent)
        return {self.output_object + "_regr": latent}

    def add_tanphi_to_preds(self, preds: Tensor):
        tanphi = preds[..., self.i_sinphi] / preds[..., self.i_cosphi]
        return torch.cat([preds, tanphi.unsqueeze(-1)], dim=-1)

    def get_hit_phi(self, track_hit_assignment: Tensor, hit_phi: Tensor) -> Tensor:
        # Count number of hits per track
        num_hits_per_track = track_hit_assignment.sum(dim=-1)  # [batch, num_queries]

        # Initialize output tensor
        batch_size, num_queries, _ = track_hit_assignment.shape
        track_hit_phi = torch.zeros(batch_size, num_queries, device=hit_phi.device)

        # For each track that has hits assigned, gather and average its hit embeddings
        for b in range(batch_size):
            for q in range(num_queries):
                if num_hits_per_track[b, q] > 0:
                    # Get indices of assigned hits
                    hit_indices = torch.where(track_hit_assignment[b, q])[0]

                    # Gather embeddings for all assigned hits
                    track_hits = hit_phi[b, hit_indices]  # [num_hits_selected]

                    # Average the embeddings
                    track_hit_phi[b, q] = track_hits.mean(dim=0)  # [b]

        return track_hit_phi

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Produce per-field predictions, inverse-scaling to original units when a scaler is provided
        latent = outputs[self.output_object + "_regr"].detach()
        field_values: dict[str, Tensor] = {}
        for i, field in enumerate(self.fields):
            value = latent[..., i]
            if self.scaler is not None:
                value = self.scaler.inverse(field)(value)
            field_values[self.output_object + "_" + field] = value
        return field_values

    def add_momentum_to_preds(self, preds: Tensor):
        momentum = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        return torch.cat([preds, momentum.unsqueeze(-1)], dim=-1)

    def add_tanphi_to_preds(self, preds: Tensor):
        tanphi = preds[..., self.i_sinphi] / preds[..., self.i_cosphi]
        return torch.cat([preds, tanphi.unsqueeze(-1)], dim=-1)

    def add_tanphi_from_p_to_preds(self, preds: Tensor):
        tanphi = preds[..., self.i_py] / preds[..., self.i_px]
        return torch.cat([preds, tanphi.unsqueeze(-1)], dim=-1)

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        # Scale targets per field if a scaler is provided
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1).to(torch.float32)
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

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Stack and optionally scale targets to match network output space
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1)
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        # Compute the loss
        loss = self.loss_fn(output, target, reduction="none")
        loss = torch.mean(loss, dim=-1)
        return {self.loss_fn_name: self.loss_weight * loss.mean()}
    



import torch.nn.functional as F


class _Adapter(nn.Module):
    """Tiny projection block: Linear -> GELU -> Linear -> LayerNorm."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hid = max(out_dim, min(4 * out_dim, 512))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, out_dim),
        )
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.ln(self.proj(x))


class ObjectSpecificHitsRegressionTaskImproved(RegressionTask):
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
        add_momentum: bool = False,
        scaler: RegressionTargetScaler | None = None,
        hit_var: str = "embed",
        # ---- New, optional knobs (all default-safe) ----
        use_stats: bool = True,
        d_block: int | None = None,
        use_block_adapters: bool = True,
        post_concat_norm: bool = True,
        log_scale_counts: bool = True,
        stats_scalar_keys: tuple[str, ...] = ("hit_r",),
        enforce_unit_circle = False,
    ):
        super().__init__(
            name,
            output_object,
            target_object,
            fields,
            loss_weight,
            cost_weight,
            loss=loss,
            has_intermediate_loss=has_intermediate_loss,
        )

        self.input_object = input_object
        self.inputs = [input_object + hit_var]
        self.outputs = [output_object + "_regr"]

        self.base_dim = dim
        self.add_momentum = add_momentum
        self.hit_var = hit_var

        if self.add_momentum:
            if all([t in self.fields for t in ["px", "py", "pz"]]):
                self.fields.append("p")
                self.i_px = self.fields.index("px")
                self.i_py = self.fields.index("py")
                self.i_pz = self.fields.index("pz")
            else:
                self.add_momentum = False

        self.scaler = scaler
        self.req_attn_mask = True

        # ---- New modules / config ----
        self.use_stats = use_stats
        self.use_block_adapters = use_block_adapters
        self.post_concat_norm = post_concat_norm
        self.log_scale_counts = log_scale_counts
        self.stats_scalar_keys = stats_scalar_keys
        self.d_block = d_block or dim  # common width for block adapters

        # Placeholders for lazily-built modules
        self.q_adapter: nn.Module | None = None
        self.h_adapter: nn.Module | None = None
        self.s_adapter: nn.Module | None = None
        self.concat_norm: nn.Module | None = None
        self.net: nn.Module | None = None

        self.enforce_unit_circle = enforce_unit_circle

        self.ndofs = len(self.fields)
        self._built = False

    # ----- Helpers -----
    @staticmethod
    def _masked_mean_and_var(mask: Tensor, feats: Tensor):
        """
        mask: [B, Q, N] boolean
        feats: [B, N, D]
        Returns:
          mean: [B, Q, D]
          var:  [B, Q, D]
          count: [B, Q, 1]
        """
        m = mask.float()  # [B,Q,N]
        count = m.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [B,Q,1]
        sum_x = torch.einsum('bqn,bnd->bqd', m, feats)
        mean = sum_x / count
        sum_x2 = torch.einsum('bqn,bnd->bqd', m, feats * feats)
        ex2 = sum_x2 / count
        var = (ex2 - mean * mean).clamp_min(0.0)
        return mean, var, count

    @staticmethod
    def _log1p_if(x: Tensor, enabled: bool) -> Tensor:
        return torch.log1p(x) if enabled else x

    def _move_built_modules_onto(self, device: torch.device) -> None:
        """Ensure any lazily-created submodules live on the given device."""
        for m in (self.q_adapter, self.h_adapter, self.s_adapter, self.concat_norm, self.net):
            if isinstance(m, nn.Module):
                m.to(device=device)

    def _maybe_build(self, query_dim: int, hit_dim: int, stats_dim: int, device: torch.device):
        """Initialize adapters & head once we know real dims. Move them onto `device`."""
        if self._built:
            return

        blocks: list[int] = []

        if self.use_block_adapters:
            self.q_adapter = _Adapter(query_dim, self.d_block)
            blocks.append(self.d_block)

            self.h_adapter = _Adapter(hit_dim, self.d_block)
            blocks.append(self.d_block)

            if self.use_stats:
                self.s_adapter = _Adapter(stats_dim, self.d_block)
                blocks.append(self.d_block)
        else:
            self.q_adapter = nn.Identity()
            blocks.append(query_dim)

            self.h_adapter = nn.Identity()
            blocks.append(hit_dim)

            if self.use_stats:
                self.s_adapter = nn.Identity()
                blocks.append(stats_dim)

        concat_dim = sum(blocks)
        self.concat_norm = nn.LayerNorm(concat_dim) if self.post_concat_norm else nn.Identity()

        # Final regression head
        self.net = Dense(concat_dim, self.ndofs)

        # >>> CRITICAL: move all newly created params to the correct device <<<
        self._move_built_modules_onto(device)

        self._built = True

    def _compute_stats(
        self,
        x: dict[str, Tensor],
        track_hit_assignment: Tensor,
        hit_embed: Tensor,
    ) -> Tensor:

        B, Q, N = track_hit_assignment.shape
        D = hit_embed.shape[-1]
        mask = track_hit_assignment  # [B,Q,N]

        mean_e, var_e, count = self._masked_mean_and_var(mask, hit_embed)   # [B,Q,D], [B,Q,D], [B,Q,1]
        hits_feat = self._log1p_if(count, self.log_scale_counts)             # [B,Q,1]

        layers_feat = None
        if "hit_layer" in x:
            layer_idx: Tensor = x["hit_layer"].long()                        # [B,N]
            L = int(layer_idx.max().item()) + 1
            onehot = F.one_hot(layer_idx, num_classes=L).float()            # [B,N,L] on same device as layer_idx
            presence = torch.einsum('bqn,bnl->bql', mask.float(), onehot)   # [B,Q,L]
            layers_hit = (presence > 0).float().sum(dim=-1, keepdim=True)   # [B,Q,1]
            layers_feat = self._log1p_if(layers_hit, self.log_scale_counts) # [B,Q,1]

        minmax_feats = []
        if "hit_layer" in x and any(k in x for k in self.stats_scalar_keys):
            for key in self.stats_scalar_keys:
                if key not in x:
                    continue
                s = x[key].to(hit_embed.dtype)                               # [B,N]
                m = mask.float()                                            # [B,Q,N]
                plus_inf = torch.full_like(s, float('inf'))
                minus_inf = torch.full_like(s, float('-inf'))
                s_expanded = s.unsqueeze(1).expand(B, Q, N)                 # [B,Q,N]

                s_min_masked = torch.where(m > 0, s_expanded, plus_inf)     # [B,Q,N]
                s_max_masked = torch.where(m > 0, s_expanded, minus_inf)

                s_min = s_min_masked.min(dim=-1, keepdim=True).values       # [B,Q,1]
                s_max = s_max_masked.max(dim=-1, keepdim=True).values       # [B,Q,1]

                s_min = torch.where(torch.isinf(s_min), torch.zeros_like(s_min), s_min)
                s_max = torch.where(torch.isinf(s_max), torch.zeros_like(s_max), s_max)
                minmax_feats.extend([s_min, s_max])

        stats_parts = [var_e, hits_feat]
        if layers_feat is not None:
            stats_parts.append(layers_feat)
        if len(minmax_feats) > 0:
            stats_parts.extend(minmax_feats)

        stats = torch.cat(stats_parts, dim=-1) if len(stats_parts) > 0 else None  # [B,Q,?]
        return mean_e, stats

    # ----- Main forward -----
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        query_embed: Tensor = x["query_embed"]                # [B,Q,Dq]
        hit_embed: Tensor = x[f"hit_{self.hit_var}"]          # [B,N,De]
        track_hit_valid: Tensor = x["attn_mask_intermediate"] # [B,Q,N] boolean

        assert track_hit_valid is not None
        mask = track_hit_valid

        # Vectorized masked mean of hit embeddings (+ optional stats)
        if self.use_stats:
            hit_mean, stats = self._compute_stats(x, mask, hit_embed)
        else:
            hit_mean = self._masked_mean_and_var(mask, hit_embed)[0]
            stats = None

        # Build modules lazily once dims & device are known
        if not self._built:
            device = query_embed.device
            query_dim = query_embed.shape[-1]
            hit_dim = hit_mean.shape[-1]
            stats_dim = 0 if (stats is None) else stats.shape[-1]
            self._maybe_build(query_dim, hit_dim, stats_dim, device)

        # Safety: if something rebuilt earlier on CPU (e.g., after load), keep them aligned
        self._move_built_modules_onto(query_embed.device)

        # Project each block
        q_proj = self.q_adapter(query_embed)             # [B,Q,d_block] or identity
        h_proj = self.h_adapter(hit_mean)                # [B,Q,d_block] or identity
        blocks = [q_proj, h_proj]

        if self.use_stats and stats is not None:
            s_proj = self.s_adapter(stats)               # [B,Q,d_block] or identity
            blocks.append(s_proj)

        combined = torch.cat(blocks, dim=-1)             # [B,Q,concat_dim]
        combined = self.concat_norm(combined)            # LayerNorm (optional)

        latent = self.net(combined)                      # [B,Q,ndofs]
        if self.add_momentum:
            latent = self.add_momentum_to_preds(latent)
        return {self.output_object + "_regr": latent}

    # ----- Predict / loss unchanged -----
    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        latent = outputs[self.output_object + "_regr"].detach()
        
        if self.enforce_unit_circle:
            # Normalize sinphi, cosphi to enforce unit circle
            if "sinphi" in self.fields and "cosphi" in self.fields:
                i_sin = self.fields.index("sinphi")
                i_cos = self.fields.index("cosphi")
                sin_pred = latent[..., i_sin]
                cos_pred = latent[..., i_cos]
                norm = torch.sqrt(sin_pred**2 + cos_pred**2 + 1e-8)
                latent[..., i_sin] = sin_pred / norm
                latent[..., i_cos] = cos_pred / norm
        
        field_values: dict[str, Tensor] = {}
        for i, field in enumerate(self.fields):
            value = latent[..., i]
            if self.scaler is not None:
                value = self.scaler.inverse(field)(value)
            field_values[self.output_object + "_" + field] = value
        return field_values

    def add_momentum_to_preds(self, preds: Tensor):
        momentum = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        return torch.cat([preds, momentum.unsqueeze(-1)], dim=-1)

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1).to(torch.float32)
        num_objects = output.shape[1]
        costs = self.loss_fn(
            output.unsqueeze(2).expand(-1, -1, num_objects, -1),
            target.unsqueeze(1).expand(-1, num_objects, -1, -1),
            reduction="none",
        ).mean(-1)
        return {f"regr_{self.loss_fn_name}": self.cost_weight * costs}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1)
        output = outputs[self.output_object + "_regr"]

        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        loss = self.loss_fn(output, target, reduction="none").mean(dim=-1)

        if self.enforce_unit_circle:
            # --- new: enforce sin^2 + cos^2 ≈ 1 ---
            if ("sinphi" in self.fields) and ("cosphi" in self.fields):
                i_sin = self.fields.index("sinphi")
                i_cos = self.fields.index("cosphi")
                sin_pred = output[..., i_sin]
                cos_pred = output[..., i_cos]
                unit_circle_error = (sin_pred ** 2 + cos_pred ** 2 - 1.0) ** 2
                unit_circle_loss = unit_circle_error.mean()
                loss = loss + 0.1 * unit_circle_loss  # weight can be tuned (0.01–1.0 typical)

        # loss = self.loss_fn(output, target, reduction="none")
        # loss = torch.mean(loss, dim=-1)
        return {self.loss_fn_name: self.loss_weight * loss.mean()}


class DiagonalGaussianRegressionTask(Task):
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
        """Regression task with independent Gaussian per target (heteroscedastic).

        Predicts mean and log-variance for each field and optimizes the diagonal
        Gaussian negative log-likelihood.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.cost_weight = cost_weight
        self.k = len(fields)
        # DoFs: k means + k log-variances
        self.ndofs = 2 * self.k
        self.likelihood_norm = self.k * 0.5 * math.log(2 * math.pi)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        latent = self.latent(x)
        k = self.k
        mu = latent[..., :k]
        logvar = latent[..., k:]
        return {self.output_object + "_mu": mu, self.output_object + "_logvar": logvar}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        preds = outputs
        mu = outputs[self.output_object + "_mu"]
        logvar = outputs[self.output_object + "_logvar"]

        for i, field in enumerate(self.fields):
            preds[self.output_object + "_" + field] = mu[..., i]
            preds[self.output_object + "_" + field + "_logvar"] = logvar[..., i]
            preds[self.output_object + "_" + field + "_prec"] = torch.exp(-logvar[..., i])

        return preds

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        mu = outputs[self.output_object + "_mu"]
        logvar = outputs[self.output_object + "_logvar"]

        # Mask valid objects
        valid = targets[self.target_object + "_valid"]
        y = y[valid]
        mu = mu[valid]
        logvar = logvar[valid]

        # Diagonal Gaussian NLL per example
        # 0.5 * [(res^2) * inv_var + log(var)] + k * 0.5 * log(2pi)
        res = y - mu
        inv_var = torch.exp(-logvar)
        nll_dim = 0.5 * (res * res * inv_var + logvar)
        nll = self.likelihood_norm + torch.sum(nll_dim, dim=-1)
        return {"nll": self.loss_weight * nll.mean()}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        mu = preds[self.output_object + "_mu"]
        logvar = preds[self.output_object + "_logvar"]
        res = y - mu
        sigma = torch.exp(0.5 * logvar)
        z = res / (sigma + 1e-12)

        valid_mask = targets[self.target_object + "_valid"]
        metrics = {}
        for i, field in enumerate(self.fields):
            metrics[field + "_rmse"] = torch.sqrt(torch.mean(torch.square(res[..., i][valid_mask])))
            metrics[field + "_pull_mean"] = torch.mean(z[..., i][valid_mask])
            metrics[field + "_pull_std"] = torch.std(z[..., i][valid_mask])
        return metrics


class ObjectDiagonalGaussianRegressionTask(DiagonalGaussianRegressionTask):
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
            output_object + "_logvar",
        ]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        mu = outputs[self.output_object + "_mu"].to(torch.float32)  # (B, N, D)
        logvar = outputs[self.output_object + "_logvar"].to(torch.float32)  # (B, N, D)
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1).to(torch.float32)  # (B, N, D)

        # Pairwise costs over predicted vs true objects
        num_objects = y.shape[1]
        mu = mu.unsqueeze(2).expand(-1, -1, num_objects, -1)  # (B, N, N, D)
        logvar = logvar.unsqueeze(2).expand(-1, -1, num_objects, -1)  # (B, N, N, D)
        y = y.unsqueeze(1).expand(-1, num_objects, -1, -1)  # (B, N, N, D)

        res = y - mu
        inv_var = torch.exp(-logvar)
        nll_dim = 0.5 * (res * res * inv_var + logvar)  # (B, N, N, D)
        nll = (self.k * 0.5 * math.log(2 * math.pi)) + torch.sum(nll_dim, dim=-1)  # (B, N, N)

        nll *= targets[f"{self.target_object}_valid"].unsqueeze(1).type_as(nll)
        return {"nll": self.cost_weight * nll}
    

class ObjectSpecificHitsRegressionTaskNew(RegressionTask):
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
        add_momentum: bool = False,
        scaler: RegressionTargetScaler | None = None,
        angular_fields: list[str] | None = None,
    ):
        """Regression task for objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)
        self.add_momentum = add_momentum

        if self.add_momentum:
            assert all([t in self.fields for t in ["px", "py", "pz"]])
            self.fields.append("p")
            self.i_px = self.fields.index("px")
            self.i_py = self.fields.index("py")
            self.i_pz = self.fields.index("pz")

        self.scaler = scaler
        self.angular_fields = set(angular_fields or [])

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def forward(self, x: dict[str, Tensor], track_hit_valid: Tensor | None = None, track_valid: Tensor | None = None) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        query_embed = x["query_embed"]
        hit_embed = x["hit_embed"]

        assert track_hit_valid is not None
        track_hit_assignment = track_hit_valid["track_hit_valid"]  # [batch, num_queries, num_hits]
        track_hit_embeds = self.get_hit_embeds(track_hit_assignment, hit_embed)

        # Sanity check: expect equal embed dims so concat width is 2*dim
        assert query_embed.size(-1) == track_hit_embeds.size(-1), "query_embed and hit_embed must have same last-dim for dim*2 projection"

        # Combine query embeddings with hit embeddings
        combined_embed = torch.cat([query_embed, track_hit_embeds], dim=-1)

        latent = self.latent(combined_embed)
        # get the predictions
        if self.add_momentum:
            latent = self.add_momentum_to_preds(latent)
        return {self.output_object + "_regr": latent}

    def get_hit_embeds(self, track_hit_assignment: Tensor, hit_embed: Tensor) -> Tensor:
        # Count number of hits per track
        num_hits_per_track = track_hit_assignment.sum(dim=-1)  # [batch, num_queries]

        # Initialize output tensor
        batch_size, num_queries, _ = track_hit_assignment.shape
        track_hit_embeds = torch.zeros(batch_size, num_queries, hit_embed.shape[-1], device=hit_embed.device)

        # For each track that has hits assigned, gather and average its hit embeddings
        for b in range(batch_size):
            for q in range(num_queries):
                if num_hits_per_track[b, q] > 0:
                    # Get indices of assigned hits
                    hit_indices = torch.where(track_hit_assignment[b, q])[0]

                    # Gather embeddings for all assigned hits
                    track_hits = hit_embed[b, hit_indices]  # [num_hits, embed_dim]

                    # Average the embeddings
                    track_hit_embeds[b, q] = track_hits.mean(dim=0)

        return track_hit_embeds

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Produce per-field predictions, inverse-scaling to original units when a scaler is provided
        latent = outputs[self.output_object + "_regr"].detach()
        field_values: dict[str, Tensor] = {}
        for i, field in enumerate(self.fields):
            value = latent[..., i]
            if self.scaler is not None:
                value = self.scaler.inverse(field)(value)
            field_values[self.output_object + "_" + field] = value
        return field_values

    def add_momentum_to_preds(self, preds: Tensor):
        momentum = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        return torch.cat([preds, momentum.unsqueeze(-1)], dim=-1)

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        num_objects = output.shape[1]

        # Build per-dimension costs, mixing angular loss where requested
        per_dim_costs = []
        for i, field in enumerate(self.fields):
            if field in self.angular_fields:
                # Use inverse-scaled predictions for angular loss
                if self.scaler is not None:
                    pred_unscaled = self.scaler.inverse(field)(output[..., i])
                else:
                    pred_unscaled = output[..., i]
                true_unscaled = targets[self.target_object + "_" + field].to(torch.float32)

                pred_exp = pred_unscaled.unsqueeze(2).expand(-1, -1, num_objects)
                true_exp = true_unscaled.unsqueeze(1).expand(-1, num_objects, -1)
                # Angular cost: 1 - cos(delta)
                cost_i = 1 - torch.cos(pred_exp - true_exp)
            else:
                # Standard regression cost in (possibly) scaled space
                if self.scaler is not None:
                    true_scaled = self.scaler.scale(field)(targets[self.target_object + "_" + field]).to(torch.float32)
                else:
                    true_scaled = targets[self.target_object + "_" + field].to(torch.float32)

                pred_exp = output[..., i].unsqueeze(2).expand(-1, -1, num_objects)
                true_exp = true_scaled.unsqueeze(1).expand(-1, num_objects, -1)

                if self.loss_fn_name == "l1":
                    cost_i = (pred_exp - true_exp).abs()
                elif self.loss_fn_name == "l2":
                    cost_i = (pred_exp - true_exp) ** 2
                else:
                    # smooth_l1
                    cost_i = torch.nn.functional.smooth_l1_loss(pred_exp, true_exp, reduction="none")

            per_dim_costs.append(cost_i.unsqueeze(-1))  # BN N 1

        costs = torch.cat(per_dim_costs, dim=-1).mean(-1)  # BN N
        return {f"regr_{self.loss_fn_name}": self.cost_weight * costs}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        output = output[mask]

        per_dim_losses = []
        for i, field in enumerate(self.fields):
            if field in self.angular_fields:
                # Compute angular loss in unscaled space: 1 - cos(delta)
                if self.scaler is not None:
                    pred_unscaled = self.scaler.inverse(field)(output[..., i])
                else:
                    pred_unscaled = output[..., i]
                true_unscaled = targets[self.target_object + "_" + field][mask]
                loss_i = 1 - torch.cos(pred_unscaled - true_unscaled)
            else:
                # Standard regression in (possibly) scaled space
                if self.scaler is not None:
                    true_scaled = self.scaler.scale(field)(targets[self.target_object + "_" + field])[mask]
                else:
                    true_scaled = targets[self.target_object + "_" + field][mask]

                if self.loss_fn_name == "l1":
                    loss_i = (output[..., i] - true_scaled).abs()
                elif self.loss_fn_name == "l2":
                    loss_i = (output[..., i] - true_scaled) ** 2
                else:
                    loss_i = torch.nn.functional.smooth_l1_loss(output[..., i], true_scaled, reduction="none")

            per_dim_losses.append(loss_i.unsqueeze(-1))

        loss = torch.cat(per_dim_losses, dim=-1).mean(-1)
        return {self.loss_fn_name: self.loss_weight * loss.mean()}




class ObjectSpecificHitsRegressionTask(RegressionTask):
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
        add_momentum: bool = False,
        scaler: RegressionTargetScaler | None = None,
    ):
        """Regression task for objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = 2 * dim
        self.net = Dense(self.dim, self.ndofs)
        self.add_momentum = add_momentum

        if self.add_momentum:
            if all([t in self.fields for t in ["px", "py", "pz"]]):
                self.fields.append("p")
                self.i_px = self.fields.index("px")
                self.i_py = self.fields.index("py")
                self.i_pz = self.fields.index("pz")
            else:
                self.add_momentum = False

        # self.add_tanphi = False
        # self.add_tanphi_from_p = False
        if "px" in self.fields:
            self.add_tanphi_from_p = True
        # if ("sinphi" in self.fields) and ("cosphi" in self.fields):
        #     self.add_tanphi = True
        #     self.fields.append("tanphi")
        #     self.i_sinphi = self.fields.index("sinphi")
        #     self.i_cosphi = self.fields.index("cosphi")
        # if self.add_tanphi_from_p:
        #     self.fields.append("tanphi")

        self.scaler = scaler
        self.req_attn_mask = True

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        query_embed = x["query_embed"]
        hit_embed = x["hit_embed"]

        track_hit_valid = x["attn_mask_intermediate"]

        assert track_hit_valid is not None
        track_hit_assignment = track_hit_valid  # [batch, num_queries, num_hits]
        track_hit_embeds = self.get_hit_embeds(track_hit_assignment, hit_embed)

        # Combine query embeddings with hit embeddings
        combined_embed = torch.cat([query_embed, track_hit_embeds], dim=-1)

        latent = self.net(combined_embed)
        # get the predictions
        if self.add_momentum:
            latent = self.add_momentum_to_preds(latent)
        # if self.add_tanphi:
        #     latent = self.add_tanphi_to_preds(latent)
        # if self.add_tanphi_from_p:
        #     latent = self.add_tanphi_from_p_to_preds(latent)
        return {self.output_object + "_regr": latent}

    def get_hit_embeds(self, track_hit_assignment: Tensor, hit_embed: Tensor) -> Tensor:
        # Count number of hits per track
        num_hits_per_track = track_hit_assignment.sum(dim=-1)  # [batch, num_queries]

        # Initialize output tensor
        batch_size, num_queries, _ = track_hit_assignment.shape
        track_hit_embeds = torch.zeros(batch_size, num_queries, hit_embed.shape[-1], device=hit_embed.device)

        # For each track that has hits assigned, gather and average its hit embeddings
        for b in range(batch_size):
            for q in range(num_queries):
                if num_hits_per_track[b, q] > 0:
                    # Get indices of assigned hits
                    hit_indices = torch.where(track_hit_assignment[b, q])[0]

                    # Gather embeddings for all assigned hits
                    track_hits = hit_embed[b, hit_indices]  # [num_hits, embed_dim]

                    # Average the embeddings
                    track_hit_embeds[b, q] = track_hits.mean(dim=0)

        return track_hit_embeds

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Produce per-field predictions, inverse-scaling to original units when a scaler is provided
        latent = outputs[self.output_object + "_regr"].detach()
        field_values: dict[str, Tensor] = {}
        for i, field in enumerate(self.fields):
            value = latent[..., i]
            if self.scaler is not None:
                value = self.scaler.inverse(field)(value)
            field_values[self.output_object + "_" + field] = value
        return field_values

    def add_momentum_to_preds(self, preds: Tensor):
        momentum = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        return torch.cat([preds, momentum.unsqueeze(-1)], dim=-1)


    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        # Scale targets per field if a scaler is provided
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1).to(torch.float32)
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

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Stack and optionally scale targets to match network output space
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1)
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        # Compute the loss
        loss = self.loss_fn(output, target, reduction="none")
        loss = torch.mean(loss, dim=-1)
        return {self.loss_fn_name: self.loss_weight * loss.mean()}


class ObjectHepformerRegressionTask(RegressionTask):
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
        add_momentum: bool = False,
        scaler: RegressionTargetScaler | None = None,
    ):
        """Regression task for objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)
        self.add_momentum = add_momentum

        if self.add_momentum:
            if all([t in self.fields for t in ["px", "py", "pz"]]):
                self.fields.append("p")
                self.i_px = self.fields.index("px")
                self.i_py = self.fields.index("py")
                self.i_pz = self.fields.index("pz")
            else:
                self.add_momentum = False

        self.scaler = scaler

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        latent = self.latent(x)
        # get the predictions
        if self.add_momentum:
            latent = self.add_momentum_to_preds(latent)
        return {self.output_object + "_regr": latent}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Produce per-field predictions, inverse-scaling to original units when a scaler is provided
        latent = outputs[self.output_object + "_regr"].detach()
        field_values: dict[str, Tensor] = {}
        for i, field in enumerate(self.fields):
            value = latent[..., i]
            if self.scaler is not None:
                value = self.scaler.inverse(field)(value)
            field_values[self.output_object + "_" + field] = value
        return field_values

    def add_momentum_to_preds(self, preds: Tensor):
        momentum = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        return torch.cat([preds, momentum.unsqueeze(-1)], dim=-1)

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        # Scale targets per field if a scaler is provided
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1).to(torch.float32)
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

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # Stack and optionally scale targets to match network output space
        if self.scaler is not None:
            target_fields = [self.scaler.scale(field)(targets[self.target_object + "_" + field]) for field in self.fields]
        else:
            target_fields = [targets[self.target_object + "_" + field] for field in self.fields]
        target = torch.stack(target_fields, dim=-1)
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        # Compute the loss
        loss = self.loss_fn(output, target, reduction="none")
        loss = torch.mean(loss, dim=-1)
        return {self.loss_fn_name: self.loss_weight * loss.mean()}