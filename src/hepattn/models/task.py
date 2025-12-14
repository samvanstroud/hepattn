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

    def __init__(self, has_intermediate_loss: bool, has_first_layer_loss: bool | None = None, permute_loss: bool = True):
        super().__init__()
        self.has_intermediate_loss = has_intermediate_loss
        self.has_first_layer_loss = has_first_layer_loss if has_first_layer_loss is not None else has_intermediate_loss
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


class ObjectClassificationTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        net: Dense | None = None,
        dim: int | None = None,
        num_classes: int = 1,
        class_weights: list[float] | None = None,
        null_weight: float = 1.0,
        mask_queries: bool = False,
        has_intermediate_loss: bool = True,
        has_first_layer_loss: bool = False,
    ):
        """Task for object detection and classification in set prediction scenarios.

        Handles both binary object detection (num_classes=1) and multi-class object
        detection+classification (num_classes>1). Always includes an implicit null class
        for empty object slots. Outputs both logits and class probabilities.

        Class Layout:
        - Valid classes: indices 0, 1, 2, ..., num_classes-1
        - Null class: index num_classes (always the LAST class)
        - For binary case (num_classes=1): [valid_class=0, null_class=1]
        - For multi-class case: [class_0, class_1, ..., class_N-1, null_class=N]

        Args:
            name: Name of the task, used as the key to separate task outputs.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            losses: Dict specifying which losses to use. Keys are loss function names and values are loss weights.
            costs: Dict specifying which costs to use. Keys are cost function names and values are cost weights.
            net: Network that will be used for classification. For binary case (num_classes=1), should output 1 logit.
                For multi-class case (num_classes>1), should output num_classes+1 logits. Cannot be specified with dim.
            dim: Input dimension for creating a default Dense network. Cannot be specified with net.
            num_classes: Number of object classes (excluding null). For binary detection, use 1.
            class_weights: Weights for each non-null class in the loss.
            null_weight: Weight applied to the null class in the loss.
            mask_queries: Whether to mask queries based on predictions.
            has_intermediate_loss: Whether the task has intermediate loss.
            has_first_layer_loss: Whether the task has first layer loss (defaults to has_intermediate_los if not specified).

        Raises:
            ValueError: If the number of class_weights doesn't match num_classes.
            ValueError: If both net and dim are specified, or neither is specified.
            ValueError: If net output size doesn't match expected size for num_classes.

        Raises:
            ValueError: If has_first_layer_loss is True but has_intermediate_loss is False.
        """
        if has_first_layer_loss and not has_intermediate_loss:
            raise ValueError("has_first_layer_loss=True requires has_intermediate_loss=True")

        super().__init__(has_intermediate_loss=has_intermediate_loss, has_first_layer_loss=has_first_layer_loss)

        # Validate net and dim arguments
        if net is not None and dim is not None:
            raise ValueError("Cannot specify both 'net' and 'dim'. Choose one.")
        if net is None and dim is None:
            raise ValueError("Must specify either 'net' or 'dim'.")

        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.num_classes = num_classes
        self.mask_queries = mask_queries

        # Create network based on provided arguments
        self.output_size = 1 if num_classes == 1 else num_classes + 1
        if net is not None:
            if net.output_size != self.output_size:
                raise ValueError(
                    f"Network output size ({net.output_size}) doesn't match expected size "
                    f"for num_classes={num_classes} (expected {self.output_size}). "
                    f"For binary case (num_classes=1), net should output 1 logit. "
                    f"For multi-class case, net should output num_classes+1 logits."
                )

            self.net = net
        else:
            assert dim is not None
            self.net = Dense(input_size=dim, output_size=self.output_size)

        # Set up class weights: [class_0, class_1, ..., class_N, null_class]
        loss_weights = torch.ones(self.num_classes + 1, dtype=torch.float32)
        if class_weights is not None:
            if len(class_weights) != self.num_classes:
                raise ValueError(f"Length of class_weights ({len(class_weights)}) does not match number of classes ({self.num_classes})")
            loss_weights[: self.num_classes] = torch.tensor(class_weights, dtype=torch.float32)
        loss_weights[-1] = null_weight  # Last class is the null class
        self.register_buffer("loss_weights", loss_weights)

        # Define semantic output keys as properties
        self.logits_key = output_object + "_logit"
        self.probs_key = output_object + "_class_prob"
        self.inputs = [input_object + "_embed"]
        self.outputs = [self.logits_key, self.probs_key]

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Output both logits and class probabilities
        x_logits = self.net(x[self.input_object + "_embed"])

        # Handle both binary and multi-class cases
        if self.num_classes == 1:
            # Convert single logit to 2-class probabilities [valid_prob, null_prob]
            x_logits = x_logits.squeeze(-1)
            x_sigmoid = torch.sigmoid(x_logits)
            x_probs = torch.stack([x_sigmoid, 1 - x_sigmoid], dim=-1)
        else:
            x_probs = torch.softmax(x_logits, dim=-1)

        return {
            self.logits_key: x_logits,
            self.probs_key: x_probs,
        }

    def predict(self, outputs: dict[str, Tensor], threshold: float = 0.5) -> dict[str, Tensor]:
        class_probs = outputs[self.output_object + "_class_prob"].detach()

        # The null class is always the LAST class (index = num_classes)
        # Valid classes are indices 0, 1, ..., num_classes-1
        # Null class index == num_classes
        valid_prob = 1 - class_probs[..., -1]
        classes = class_probs.argmax(-1)
        valid = classes < self.num_classes

        return {
            f"{self.output_object}_class": classes,
            f"{self.output_object}_valid_prob": valid_prob,
            f"{self.output_object}_valid": valid,
        }

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        costs = {}

        if self.num_classes == 1:
            # Binary detection case
            output = outputs[self.output_object + "_logit"].detach().to(torch.float32)
            target = targets[self.target_object + "_valid"].to(torch.float32)
        else:
            # Multi-class detection case
            output = outputs[self.output_object + "_class_prob"].detach().to(torch.float32)
            target = targets[self.target_object + "_class"].long()

        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses = {}

        if self.num_classes == 1:
            # Binary detection case - use logits for loss computation
            output = outputs[self.output_object + "_logit"]
            target = targets[self.target_object + "_valid"].float()
            sample_weight = target + self.loss_weights[-1] * (1 - target)

            for loss_fn, loss_weight in self.losses.items():
                losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, sample_weight=sample_weight)
        else:
            # Multi-class detection case - use logits for loss computation
            output = outputs[self.output_object + "_logit"]
            target = targets[self.target_object + "_class"].long()

            for loss_fn, loss_weight in self.losses.items():
                losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=None, weight=self.loss_weights)

        return losses

    def query_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> Tensor | None:
        if not self.mask_queries:
            return None

        class_probs = outputs[self.output_object + "_class_prob"].detach()
        return class_probs[..., -1] <= (1 - threshold)


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
        has_intermediate_loss: bool = True,
    ):
        """Task used for classifying whether constituents belong to reconstructable objects or not.

        Args:
            name: Name of the task.
            input_object: Name of the constituent type.
            target_field: Name of the target field to predict.
            dim: Embedding dimension.
            threshold: Threshold for classification.
            mask_keys: Whether to mask keys.
            loss_fn: Loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)

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

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        probs = outputs[f"{self.input_object}_logit"].sigmoid()
        return {
            f"{self.input_object}_{self.target_field}_prob": probs,
            f"{self.input_object}_{self.target_field}": probs >= self.threshold,
        }

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        # Pick out the field that denotes whether a hit is on a reconstructable object or not
        output = outputs[f"{self.input_object}_logit"]
        target = targets[f"{self.input_object}_{self.target_field}"].type_as(output)

        # Calculate the BCE loss with class weighting
        if self.loss_fn == "bce":
            pos_weight = 1 / target.float().mean()
            loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "focal":
            loss = mask_focal_loss(output, target)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "both":
            pos_weight = 1 / target.float().mean()
            bce_loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
            focal_loss_value = mask_focal_loss(output, target)
            return {
                f"{self.input_object}_bce": bce_loss,
                f"{self.input_object}_focal": focal_loss_value,
            }
        raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def key_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> dict[str, Tensor]:
        if not self.mask_keys:
            return {}

        return {self.input_object: outputs[f"{self.input_object}_logit"].detach().sigmoid() >= threshold}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        expected_key = f"{self.input_object}_{self.target_field}"
        pred = preds[expected_key]
        true = targets[expected_key]

        tp = (pred * true).sum()
        tn = ((~pred) * (~true)).sum()

        metrics = {
            # Log quantities based on the number of hits
            "nh_total_pre": float(pred.shape[1]),
            "nh_total_post": float(pred.sum()),
            "nh_pred_true": pred.float().sum(),
            "nh_pred_false": (~pred).float().sum(),
            "nh_valid_pre": true.float().sum(),
            "nh_valid_post": (pred & true).float().sum(),
            "nh_noise_pre": (~true).float().sum(),
            "nh_noise_post": (pred & ~true).float().sum(),
            # Standard binary classification metrics
            "acc": (pred == true).half().mean(),
            "valid_recall": tp / true.sum(),
            "valid_precision": tp / pred.sum(),
            "noise_recall": tn / (~true).sum(),
            "noise_precision": tn / (~pred).sum(),
        }

        return metrics


class ObjectHitMaskTask(Task):
    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        object_net: nn.Module | None = None,
        constituent_net: nn.Module | None = None,
        null_weight: float = 1.0,
        mask_attn: bool = True,
        target_field: str = "valid",
        logit_scale: float = 1.0,
        pred_threshold: float = 0.5,
        mask_attention_threshold: float | None = None,
        predict_iou: bool = False,
        iou_loss_weight: float = 1.0,
        has_intermediate_loss: bool = True,
    ):
        """Task for predicting associations between objects and hits.

        Args:
            name: Name of the task.
            input_constituent: Name of the input constituent type (traditionally hits in tracking).
                For unified decoding, use "key" to access merged embeddings.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            losses: Loss functions and their weights.
            costs: Cost functions and their weights.
            dim: Embedding dimension.
            object_net: Get mask tokens from object embeddings
            constituent_net: Get constituent mask tokens from constituent embeddings.
                This is NOT RECOMMENDED - whatever you do, don't use an output activation.
            null_weight: Weight for null class.
            mask_attn: Whether to mask attention.
            target_field: Target field name.
            logit_scale: Scale for logits.
            pred_threshold: Prediction threshold.
            mask_attention_threshold: Threshold for attention masking. Defaults to pred_threshold if None.
            predict_iou: Whether to predict the IoU of the predicted mask.
            iou_loss_weight: Weight for the IoU loss.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.target_field = target_field

        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.constituent_net = constituent_net
        self.object_net = object_net or Dense(dim, dim)
        self.null_weight = null_weight
        self.mask_attn = mask_attn
        self.logit_scale = logit_scale
        self.pred_threshold = pred_threshold
        self.mask_attention_threshold = mask_attention_threshold if mask_attention_threshold is not None else pred_threshold
        self.predict_iou = predict_iou
        self.iou_loss_weight = iou_loss_weight
        self.has_intermediate_loss = mask_attn or predict_iou

        if self.predict_iou:
            self.iou_net = Dense(dim, 1)

        self.output_object_hit = output_object + "_" + input_constituent
        self.target_object_hit = target_object + "_" + input_constituent

        self.inputs = [input_object + "_embed", input_constituent + "_embed"]
        self.outputs = [self.output_object_hit + "_logit"]
        if self.predict_iou:
            self.outputs.append(self.output_object + "_iou_logit")

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Produce mask tokens
        mask_tokens = self.object_net(x[self.input_object + "_embed"])
        xs = x[self.input_constituent + "_embed"]
        if self.constituent_net:
            xs = self.constituent_net(xs)

        # Object-hit probability is the dot product between the hit and object embedding
        object_hit_logit = self.logit_scale * torch.einsum("bnc,bmc->bnm", mask_tokens, xs)

        # Zero out entries for any padded input constituents
        if (valid_mask := x[f"{self.input_constituent}_valid"]) is not None:
            valid_mask = valid_mask.unsqueeze(-2).expand_as(object_hit_logit)
            object_hit_logit[~valid_mask] = torch.finfo(object_hit_logit.dtype).min

        outputs = {self.output_object_hit + "_logit": object_hit_logit}

        if self.predict_iou:
            outputs[self.output_object + "_iou_logit"] = self.iou_net(x[self.input_object + "_embed"]).squeeze(-1)

        return outputs

    def attn_mask(self, outputs: dict[str, Tensor], threshold: float | None = None) -> dict[str, Tensor]:
        if not self.mask_attn:
            return {}

        thresh = threshold if threshold is not None else self.mask_attention_threshold
        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= thresh
        return {self.input_constituent: attn_mask}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        output = {}
        probs = outputs[self.output_object_hit + "_logit"].sigmoid().detach()
        output[self.output_object_hit + "_valid_prob"] = probs
        output[self.output_object_hit + "_valid"] = probs >= self.pred_threshold

        if self.predict_iou:
            output[self.output_object + "_iou"] = outputs[self.output_object + "_iou_logit"].detach().sigmoid()

        return output

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object_hit + "_logit"].detach().to(torch.float32)
        target = targets[self.target_object_hit + "_" + self.target_field].detach().to(output.dtype)

        hit_pad = targets[self.input_constituent + "_valid"]

        costs = {}
        # sample_weight = target + self.null_weight * (1 - target)
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target, input_pad_mask=hit_pad)
        return costs

    def calculate_iou(self, pred_probs: Tensor, target: Tensor) -> Tensor:
        """Calculate IoU between predicted probabilities and target mask.

        Args:
            pred_probs: Predicted probabilities (B, N, M)
            target: Target mask (B, N, M)

        Returns:
            IoU values (B, N)
        """
        intersection = (pred_probs * target).sum(dim=-1)
        union = pred_probs.sum(dim=-1) + target.sum(dim=-1) - intersection

        # Avoid division by zero
        return intersection / (union + 1e-6)

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_" + self.target_field].type_as(output)

        hit_pad = targets[self.input_constituent + "_valid"]
        object_pad = targets[self.target_object + "_valid"]

        sample_weight = target + self.null_weight * (1 - target)
        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](
                output, target, object_valid_mask=object_pad, input_pad_mask=hit_pad, sample_weight=sample_weight
            )

        if self.predict_iou:
            # Calculate the actual IoU between the predicted mask and the target mask
            # Get the predicted probabilities
            pred_probs = output.sigmoid()

            # Calculate IoU using helper method
            iou_target = self.calculate_iou(pred_probs, target)

            # Get the predicted IoU
            iou_pred = outputs[self.output_object + "_iou_logit"].sigmoid()

            # Only compute loss for valid objects
            if object_pad is not None:
                iou_target = iou_target[object_pad]
                iou_pred = iou_pred[object_pad]

            # Compute MSE loss
            iou_loss = torch.nn.functional.mse_loss(iou_pred, iou_target.detach())
            losses["iou_mse"] = self.iou_loss_weight * iou_loss

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

        # Define semantic output key as property
        self.regression_key = output_object + "_regr"

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        latent = self.latent(x)
        return {self.regression_key: latent}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Split the regression vector into the separate fields
        latent = outputs[self.regression_key]
        return {self.output_object + "_" + field: latent[..., i] for i, field in enumerate(self.fields)}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        output = outputs[self.regression_key]

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
        self.outputs = [self.regression_key]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.regression_key].detach().to(torch.float32)
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


class ClassificationTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        classes: list[str],
        net: nn.Module,
        output_object: str | None = None,
        target_object: str | None = None,
        class_weights: dict[str, float] | None = None,
        loss_weight: float = 1.0,
        threshold: float = 0.5,
        multilabel: bool = False,
        permute_loss: bool = True,
        has_intermediate_loss: bool = True,
    ):
        """Standard classification task for existing objects.

        This task is for scenarios where all input objects are already known to be valid,
        and the goal is to classify them into specific categories. Unlike ObjectClassificationTask,
        this task does NOT handle object detection (valid/invalid) and assumes all inputs
        represent real objects that just need categorization.

        Use cases:
        - Jet flavor tagging (b-jet, c-jet, tau-jet, other)
        - Particle type classification for known particles

        For object detection + classification (where slots can be empty), use ObjectClassificationTask instead.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            classes: List of class names (no null class - all inputs assumed valid).
            net: Network for classification. Should output len(classes) logits.
            class_weights: Weights for each class in the loss function.
            loss_weight: Weight for the loss function.
            threshold: Threshold for classification predictions.
            multilabel: Whether this is a multilabel classification.
            permute_loss: Whether to permute loss.
            has_intermediate_loss: Whether the task has intermediate loss.
            output_object: Name of the output object. Defaults to input_object if empty string.
            target_object: Name of the target object. Defaults to input_object if empty string.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=permute_loss)

        self.name = name
        self.input_object = input_object
        self.output_object = output_object if output_object is not None else input_object
        self.target_object = target_object if target_object is not None else input_object
        self.classes = classes
        self.class_weights = class_weights
        self.loss_weight = loss_weight
        self.threshold = threshold
        self.multilabel = multilabel
        self.net = net

        if self.class_weights is not None:
            self.class_weights_values = torch.tensor([self.class_weights[class_name] for class_name in self.classes])

        self.inputs = [input_object + "_embed"]
        self.outputs = [self.output_object + "_logits"]

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Get class logits from the configurable network
        x = self.net(x[f"{self.input_object}_embed"])
        return {f"{self.output_object}_logits": x}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        logits = outputs[self.output_object + "_logits"].detach()
        result = {}

        if len(self.classes) == 1 and not self.multilabel:
            # Binary classification with single output
            logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits
            probs = torch.nn.functional.sigmoid(logits)
            result[self.output_object + "_" + self.classes[0] + "_prob"] = probs
            result[self.output_object + "_" + self.classes[0]] = probs >= self.threshold
        elif self.multilabel:
            # Multilabel classification
            probs = torch.nn.functional.sigmoid(logits)
            for i, class_name in enumerate(self.classes):
                result[self.output_object + "_" + class_name + "_prob"] = probs[..., i]
                result[self.output_object + "_" + class_name] = probs[..., i] >= self.threshold
        else:
            # Multi-class classification
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.nn.functional.one_hot(torch.argmax(logits, dim=-1), num_classes=len(self.classes))
            for i, class_name in enumerate(self.classes):
                result[self.output_object + "_" + class_name + "_prob"] = probs[..., i]
                result[self.output_object + "_" + class_name] = predictions[..., i]

        return result

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        logits = outputs[f"{self.output_object}_logits"]

        if len(self.classes) == 1 and not self.multilabel:
            # Binary classification with single output
            target = targets[self.target_object + "_" + self.classes[0]].float()
            logits = logits.squeeze(-1) if logits.dim() > target.dim() else logits

            # Apply class weight if specified
            pos_weight = None
            if self.class_weights is not None:
                pos_weight = torch.tensor([self.class_weights[self.classes[0]]], dtype=logits.dtype, device=logits.device)

            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
        elif self.multilabel:
            # Multilabel classification with BCE per class
            target = torch.stack([targets[self.target_object + "_" + class_name] for class_name in self.classes], dim=-1).float()

            # Apply class weights if specified
            pos_weight = None
            if self.class_weights is not None:
                pos_weight = self.class_weights_values.type_as(target)

            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none").mean(dim=-1)
        else:
            # Multi-class classification with cross entropy
            target = torch.stack([targets[self.target_object + "_" + class_name] for class_name in self.classes], dim=-1)

            # Put the class weights into a tensor with the correct dtype
            class_weights = None
            if self.class_weights is not None:
                class_weights = self.class_weights_values.type_as(target)

            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target.view(-1, target.shape[-1]),
                weight=class_weights,
                reduction="none",
            )

        # Only consider valid targets - flatten both losses and mask
        valid_mask = targets[f"{self.target_object}_valid"].view(-1)
        losses = losses.view(-1)[valid_mask]
        return {"bce": self.loss_weight * losses.mean()}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        metrics = {}
        for class_name in self.classes:
            target = targets[f"{self.target_object}_{class_name}"][targets[f"{self.target_object}_valid"]].bool()
            pred = preds[f"{self.output_object}_{class_name}"][targets[f"{self.target_object}_valid"]].bool()

            true_positives = float((target & pred).sum())
            false_positives = float((~target & pred).sum())

            metrics[f"{class_name}_eff"] = true_positives / target.sum()
            metrics[f"{class_name}_pur"] = true_positives / pred.sum()
            metrics[f"{class_name}_tp"] = true_positives
            metrics[f"{class_name}_fp"] = false_positives

        return metrics


class IncidenceRegressionTask(Task):
    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        net: nn.Module,
        node_net: nn.Module | None = None,
        has_intermediate_loss: bool = True,
    ):
        """Incidence regression task.

        Args:
            name: Name of the task.
            input_constituent: Name of the input hit object.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            losses: Loss functions and their weights.
            costs: Cost functions and their weights.
            net: Network for object embedding.
            node_net: Network for node embedding.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)
        self.name = name
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.net = net
        self.node_net = node_net if node_net is not None else nn.Identity()

        self.incidence_key = self.output_object + "_incidence"
        self.inputs = [input_object + "_embed", input_constituent + "_embed"]
        self.outputs = [self.incidence_key]

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_object = self.net(x[self.input_object + "_embed"])
        x_hit = self.node_net(x[self.input_constituent + "_embed"])

        incidence_pred = torch.einsum("bqe,ble->bql", x_object, x_hit)
        incidence_pred = incidence_pred.softmax(dim=1) * x[self.input_constituent + "_valid"].unsqueeze(1).expand_as(incidence_pred)

        return {self.incidence_key: incidence_pred}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return {self.output_object + "_incidence": outputs[self.incidence_key].detach()}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.incidence_key].detach().to(torch.float32)
        target = targets[self.target_object + "_incidence"].to(torch.float32)

        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses = {}
        output = outputs[self.incidence_key]
        target = targets[self.target_object + "_incidence"].type_as(output)

        # Create a mask for valid nodes and objects
        node_mask = targets[self.input_constituent + "_valid"].unsqueeze(1).expand_as(output)
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
        input_constituent: str,
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
        has_intermediate_loss: bool = True,
        mode: str = "offset",
        cost: str = "old",
    ):
        """Construct proxy particles from predicted incidence matrix, and then correct the proxies using a regression.

        Args:
            name: Name of the task.
            input_constituent: Name of the input hit object.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            scale_dict_path: Path to the scale dictionary.
            net: Network for regression.
            loss: Type of loss function to use.
            use_incidence: Whether to use incidence matrix.
            use_nodes: Whether to use node features.
            has_intermediate_loss: Whether the task has intermediate loss.
            mode: Regression mode ('offset' or 'scale').
            cost: Cost mode ('old' or 'new').

        Raises:
            ValueError: If the mode is not 'offset' or 'scale'.
            ValueError: If the cost mode is not 'old' or 'new'.
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
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.scaler = FeatureScaler(scale_dict_path=scale_dict_path)
        self.use_incidence = use_incidence
        self.cost_weight = cost_weight
        self.net = net
        self.use_nodes = use_nodes
        self.inputs = [input_object + "_embed"] + [input_constituent + "_" + field for field in fields]
        self.outputs = [output_object + "_regr", output_object + "_proxy_regr"]
        self.mode = mode
        if mode not in {"offset", "scale"}:
            raise ValueError(f"Invalid mode {mode}, must be 'offset' or 'scale'")
        if cost == "old":
            self.cost = self.old_cost
        elif cost == "new":
            self.cost = self.new_cost
        else:
            raise ValueError(f"Invalid cost mode {cost}")

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # get the predictions
        if self.use_incidence:
            inc = x["incidence"].detach()
            proxy_feats, is_charged = self.get_proxy_feats(inc, x["inputs"], class_probs=x["class_probs"].detach())
            input_data = torch.cat(
                [
                    x[self.input_object + "_embed"],
                    proxy_feats,
                    is_charged.unsqueeze(-1),
                ],
                -1,
            )
            if self.use_nodes:
                valid_mask = x[self.input_constituent + "_valid"].unsqueeze(-1)
                masked_embed = valid_mask * x[self.input_constituent + "_embed"]
                node_feats = torch.bmm(inc, masked_embed)
                input_data = torch.cat([input_data, node_feats], dim=-1)
        else:
            input_data = x[self.input_object + "_embed"]
            proxy_feats = torch.zeros_like(input_data[..., : len(self.fields)])
        if self.mode == "offset":
            preds = self.net(input_data) + proxy_feats
        elif self.mode == "scale":
            preds = self.net(input_data) * proxy_feats
        else:
            raise ValueError(f"Invalid mode {self.mode}")
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

    def old_cost(self, outputs, targets) -> dict[str, Tensor]:
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
            pred_pt = outputs[self.output_object + "_regr"][..., self.pt_pos][:, :, None]
            target_pt = targets[self.target_object + "_pt"][:, None, :]
            pt_cost = (target_pt - pred_pt) ** 2 / (target_pt**2 + 1e-8)
        else:
            pt_cost = 0
        # Compute the cost as the sum of the squared differences
        cost = self.cost_weight * torch.sqrt(pt_cost + torch.pow(dphi, 2) + torch.pow(deta, 2))
        return {"regression": cost}

    def new_cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1).to(torch.float32)
        num_objects = output.shape[1]
        num_targets = target.shape[1]

        # The expand is not necessary but stops a broadcasting warning
        costs = self.loss_fn(
            output.unsqueeze(2).expand(-1, -1, num_objects, -1),
            target.unsqueeze(1).expand(-1, num_targets, -1, -1),
            reduction="none",
        )

        return {f"regr_{self.loss_fn_name}": self.cost_weight * costs.mean(-1)}

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"]
        target = target[mask]
        output = output[mask]

        loss = self.loss_fn(output, target, reduction="mean")
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
            [inputs[self.input_constituent + "_" + field].unsqueeze(-1) for field in self.fields],
            dim=-1,
        )

        charged_inc = incidence * inputs[self.input_constituent + "_is_track"].unsqueeze(1)
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
        inc_e_weighted *= 1 - inputs[self.input_constituent + "_is_track"].unsqueeze(1)
        inc = inc_e_weighted / (inc_e_weighted.sum(dim=-1, keepdim=True) + 1e-6)

        proxy_feats_neutral = torch.einsum("bnf,bpn->bpf", proxy_feats, inc)
        proxy_feats_neutral[..., 0] = inc_e_weighted.sum(-1)
        proxy_feats_neutral[..., 1] = proxy_feats_neutral[..., 0] / torch.cosh(proxy_feats_neutral[..., 2])

        proxy_feats_neutral = self.scale_proxy_feats(proxy_feats_neutral) * (~is_charged).unsqueeze(-1)
        proxy_feats = proxy_feats_charged + proxy_feats_neutral

        return proxy_feats, is_charged
