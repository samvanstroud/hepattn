import operator

import torch
from torch import Tensor

from hepattn.models.matcher import Matcher
from hepattn.utils.metrics import mask_metric_cost, mask_metric_score

# Map string to operator function
ops = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def apply_matching(data: dict[str, Tensor], true: str, pred: str, costs: Tensor, matcher: Matcher) -> dict[str, Tensor]:
    """Applies bipartite matching between predicted and true objects using the provided cost tensor.

    The function uses a Hungarian matcher to compute the optimal matching between predictions and ground truths,
    then permutes the prediction-aligned fields in `data` according to the matching indices.

    Parameters
    ----------
    data : dict[str, Tensor]
        Dictionary containing prediction and ground truth data. Fields should follow the naming convention
        `{pred|true}_{field}` and include a `{true}_valid` mask.
    true : str
        Name for the ground truth object (e.g., "particle").
    pred : str
        Name for the predicted object (e.g., "flow").
    costs : Tensor
        Cost tensor of shape `(batch_size, n_pred, n_true)` representing matching costs.

    Returns:
    -------
    dict[str, Tensor]
        Updated `data` dictionary with predicted object fields permuted to align with matched true objects.
    """
    batch_idxs = torch.arange(data[f"{true}_valid"].shape[0]).unsqueeze(1)

    # Compute the matching, remember costs have shape (batch, pred, true)
    pred_idxs = matcher(costs, data[f"{true}_valid"])

    # Permute all the items that are associated to the predicted objects
    for item_name in data:
        if item_name.split("_")[0] == pred:
            data[item_name] = data[item_name][batch_idxs, pred_idxs]

    # Return the reference even though not strictly necessary since we permuted in place
    return data


def calc_cost(data: dict[str, Tensor], true: str, pred: str, metrics: dict) -> Tensor:
    """Computes the total matching cost between predicted and true objects using a dictionary of metric definitions.

    Each metric defines a constituent, a field, a function, and a weight. The final cost is a weighted sum
    across all constituent-level metric costs.

    Parameters
    ----------
    data : dict[str, Tensor]
        Dictionary containing prediction and ground truth data.
    true : str
        Name for ground truth objects.
    pred : str
        Name for predicted objects.
    metrics : dict
        Dictionary of metric definitions. Each value should be a dictionary with:
        - "field": str, the feature name.
        - "metric": callable or metric identifier.
        - "weight": float, weight applied to this metric's cost.

    Returns:
    -------
    Tensor
        Total cost tensor of shape (batch_size, num_objects, num_objects).
    """
    total_cost = None

    for constituent, metric in metrics.items():
        # Calculate the weighted cost contributed by this constituent and this metric
        cost = metric["weight"] * mask_metric_cost(
            data[f"{pred}_{constituent}_valid"].float(),
            data[f"{true}_{constituent}_valid"].float(),
            input_pad_mask=data[f"{constituent}_{metric['field']}"],
            metric=metric["metric"],
        )

        # Add the cost on to the running total cost
        if total_cost is not None:
            total_cost += cost
        else:
            total_cost = cost

    return total_cost


def calc_binary_reco_metrics(data: dict[str, Tensor], true: str, pred: str, metric_definitions: dict) -> dict[str, Tensor]:
    """Computes binary reconstruction metrics indicating whether each true object is matched by a predicted one.

    For each named metric, a composite condition is evaluated over multiple constituents. The result is a
    boolean tensor indicating per-object satisfaction of the metric.

    Parameters
    ----------
    data : dict[str, Tensor]
        Dictionary containing prediction and ground truth tensors.
    true : str
        Name of ground truth objects.
    pred : str
        Name of predicted object.
    metric_definitions : dict
        Dictionary of metric definitions, where each key is a metric name and each value is a dictionary
        of constituent-level metrics with fields:
        - "field": str, the feature to compare.
        - "metric": callable or identifier for score computation.
        - "thresh": float, the minimum acceptable score.

    Returns:
    -------
    dict[str, Tensor]
        Dictionary with keys `{true}_{pred}_{metric_name}` and boolean tensors indicating which true objects
        are successfully reconstructed under each metric by the pred object collection.
    """
    # Calculates whether the true objects are reconstructed by the pred under some metrics

    metric_evals = {}
    for metric_name, constituent_metrics in metric_definitions.items():
        # Contains a mask which will true if this metric is satisfied
        metric_satisfied = torch.full_like(data[f"{true}_valid"], True)

        # Go through all the conditions needed for the metric to be satisfied
        for metric in constituent_metrics:
            constituent = metric["hit"]
            field = metric["field"]
            scores = mask_metric_score(
                data[f"{pred}_{constituent}_{field}"].float(),
                data[f"{true}_{constituent}_{field}"].float(),
                input_pad_mask=data[f"{constituent}_valid"],
                metric=metric["metric"],
            )

            metric_satisfied &= scores >= metric["thresh"]

        # The true object is reconstructed by the pred object under the metric definition
        metric_evals[f"{true}_{pred}_{metric_name}"] = metric_satisfied

    return metric_evals


def calculate_selections(data: dict[str, Tensor], object_name: str, selection_definitions: dict[str, list[str]]) -> dict[str, Tensor]:
    """Computes selection masks based on logical AND over multiple precomputed boolean masks.

    Each selection is defined as a conjunction of other masks related to a given object type.

    Parameters
    ----------
    data : dict[str, Tensor]
        Dictionary containing boolean mask tensors (e.g., validity, quality criteria).
    object_name : str
        Prefix for the object type (e.g., "particle", "jet") whose selections are being computed.
    selection_definitions : dict[str, list[str]]
        Dictionary mapping selection names to a list of required masks.

    Returns:
    -------
    dict[str, Tensor]
        Dictionary of new selection masks with keys `{object_name}_{selection_name}`.
    """
    # Calculate selections which are just an and of other boolean masks
    for selection_name, selection_requirements in selection_definitions.items():
        selection_mask = torch.full_like(data[f"{object_name}_valid"], True)

        for selection_requirement in selection_requirements:
            split = selection_requirement.split(" ")
            if len(split) == 3:
                selection_mask &= ops[split[1]](data[f"{object_name}_{split[0]}"], float(split[2]))
            else:
                selection_mask &= data[f"{object_name}_{selection_requirement}"].bool()

        data[f"{object_name}_{selection_name}"] = selection_mask

    return data
