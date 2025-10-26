from collections.abc import Mapping

import torch
from torch import Tensor, nn

from hepformer.models import Dense
from hepformer.utils.scaling import RegressionTargetScaler


class Task(nn.Module):
    def __init__(
        self,
        name: str,
        input_type: str,
        label: str,
        net: Dense,
        loss: nn.Module,
        weight: float = 1.0,
    ):
        """Task head.

        A wrapper around a dense network, a loss function, a label and a weight.

        Parameters
        ----------
        name : str
            Name of the task
        input_type : str
            Which type of object is input to the task e.g. jet/track/flow
        label : str
            Label name for the task
        net : Dense
            Dense network for performing the task
        loss : nn.Module
            Task loss
        weight : float
            Weight in the overall loss
        label_map : Mapping
            Remap integer labels for training (e.g. 0,4,5 -> 0,1,2)

        """
        super().__init__()

        self.name = name
        self.input_type = input_type
        self.label = label
        self.net = net
        self.loss = loss
        self.weight = weight

    def input_type_mask(self, masks):
        return torch.cat([torch.ones(m.shape[1]) * (t == self.input_type) for t, m in masks.items()]).bool()


class ClassificationTask(Task):
    def __init__(self, label_map: Mapping | None = None, **kwargs):
        super().__init__(**kwargs)
        self.label_map = label_map

    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        masks: Mapping | None = None,
    ):
        if masks is not None and self.name == "track_origin":
            preds = self.net(x["x"])
            mask = masks
        else:
            preds = self.net(x["pooled"])
            mask = None

        # get labels
        labels = labels_dict[self.label] if labels_dict else None
        if labels is not None and self.label_map is not None:
            for k, v in self.label_map.items():
                labels[labels == k] = v

        # could use ignore_index instead of the mask here
        # TODO remove when https://gitlab.cern.ch/atlas/athena/-/merge_requests/60199
        # is in the samples
        if mask is not None:
            if labels is not None:
                mask = torch.masked_fill(mask, labels == -2, 1)
            preds = preds[~mask]  # type: ignore
            if labels is not None:
                labels = labels[~mask]  # type: ignore

        loss = None
        if labels is not None:
            loss = self.loss(preds.permute(0, 2, 1), labels) if preds.ndim == 3 else self.loss(preds, labels)
            loss *= self.weight

        return {self.name: preds}, {self.name: loss}


class RegressionTask(Task):
    def __init__(self, targets: list[str], scaler: RegressionTargetScaler, add_momentum: bool = False, **kwargs):
        """Regression task without uncertainty prediction.

        Parameters
        ----------
        targets : list
            List of target names
        scaler : RegressionTargetScaler
            Target scaler object
        add_momentum : bool
            Whether to add scalar momentum to the predictions, computed from the px, py, pz predictions
        """
        super().__init__(**kwargs)
        self.targets = targets
        self.scaler = scaler
        self.add_momentum = add_momentum
        if self.add_momentum:
            assert all([t in self.targets for t in ["px", "py", "pz"]])
            self.targets.append("p")
            self.i_px = self.targets.index("px")
            self.i_py = self.targets.index("py")
            self.i_pz = self.targets.index("pz")

    def loss(self, preds: dict, labels: dict, label_masks: dict | None = None):
        if label_masks is None:
            label_masks = {}
        preds = preds[self.name]
        labels = labels[self.name] if labels else None
        valid_idx = ~torch.isnan(labels).all(-1)
        valid_idx &= ~(labels == 0).all(-1)  # type: ignore

        if self.name in label_masks:
            valid_idx = torch.logical_and(valid_idx, label_masks[self.name])

        loss = None
        loss_dict = {}
        # scale and concatenate regression targets for later use in the matcher and get_loss() method
        if labels is not None:
            targets = []
            for tgt in self.targets:
                scaled_tgt = self.scaler.scale(tgt)(labels[tgt])
                targets.append(scaled_tgt.unsqueeze(-1))
            targets = torch.cat(targets, dim=-1)
            labels[self.name] = targets
            loss = self.loss(preds[valid_idx], labels[valid_idx]).mean(-2)
            loss *= self.weight
            for i, t in enumerate(targets):
                loss_dict[f"{t}_loss"] = loss[i]

        return loss_dict

    def forward(self, x: dict):
        # get the predictions
        preds = self.net(x[self.input_type]).squeeze(-1)
        if self.add_momentum:
            preds = self.add_momentum_to_preds(preds)
        return {self.name: preds}

    def add_momentum_to_preds(self, preds: Tensor):
        momenutm = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        preds = torch.cat([preds, momenutm.unsqueeze(-1)], dim=-1)
        return preds


def cosine_loss(pred_x, pred_y, true_x, true_y):
    return 1 - (pred_x * true_x + pred_y * true_y) / (torch.sqrt(pred_x**2 + pred_y**2) * torch.sqrt(true_x**2 + true_y**2))


def l1_regression_loss(pred_x, pred_y, true_x, true_y):
    return torch.abs(pred_x - true_x) + torch.abs(pred_y - true_y)


def another_loss(pred_x, pred_y, true_x, true_y):
    return 0.5 * ((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2).mean(-2)


"""
class TrackRegressionTask(Task):
    def __init__(self, targets: list[str], scaler: RegressionTargetScaler, angular_loss: str, **kwargs):
        super().__init__(**kwargs)
        self.targets = targets
        self.scaler = scaler
        assert targets == ["p", "vz", "theta", "phi"]
        assert self.net.output_size == 6
        if angular_loss == "cosine":
            self.angular_loss = cosine_loss
        elif angular_loss == "l1":
            self.angular_loss = l1_regression_loss
        elif angular_loss == "another":
            self.angular_loss = another_loss

    def get_loss(self, preds: dict, labels: dict, label_masks: dict | None = None):
        if label_masks is None:
            label_masks = {}
        preds = preds[self.name]
        labels = labels[self.name] if labels else None
        assert not torch.isnan(labels).any()

        loss_dict = {}
        if labels is not None:
            for i, t in enumerate(self.targets):
                if t in ["p", "vz"]:
                    loss = self.loss(preds[:, i], labels[:, i]).mean(-1)
                else:
                    pred_x, pred_y = torch.tanh(preds[:, i]), torch.tanh(preds[:, i + 2])
                    true_x, true_y = torch.cos(labels[t]), torch.sin(labels[t])
                    loss = self.angular_loss(pred_x, pred_y, true_x, true_y).mean(-1)
                loss_dict[f"{t}_loss"] = loss * self.weight
        return loss_dict

    def forward(self, x: dict, labels: dict):
        # scale and concatenate regression targets for later use in the matcher and get_loss() method
        if labels is not None:
            scaled = {}
            for tgt in self.targets:
                scaled[tgt] = self.scaler.scale(tgt)(labels[tgt]).unsqueeze(-1)
            # TODO: put the targets back into the labels, or don't, but need to modify matcher
            labels[self.name] = targets
        # get the predictions
        preds = self.net(x[self.input_type]).squeeze(-1)
        return {self.name: preds}
"""
