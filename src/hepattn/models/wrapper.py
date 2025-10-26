from typing import Literal

import torch
import torchmetrics as tm
from lightning import LightningModule
from lion_pytorch import Lion
from torch import nn
from torch.optim import AdamW
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad

from hepattn.utils.masks_hepformer import mask_from_logits


class MaskInference:
    def basic_sigmoid(self, pred):
        """
        Assign hits to tracks if they have a high matching probability.
        Able to assign a hit to more than one track.
        """
        pred = pred.sigmoid() > 0.5
        return pred

    def basic_argmax(self, pred):
        """
        Assign hits to the track with the highest probability.
        Can only assign one hit to one track.
        """
        idx = pred.argmax(-2)
        pred = torch.full_like(pred, False).bool()
        pred[idx, torch.arange(len(idx))] = True
        return pred

    def weighted_argmax(self, pred, class_preds):
        """
        Assign hits to the track with the highest probabilithy, weighted with class pred confidence.
        Can only assign one hit to one track.
        This is used in the Maskformer paper.
        """
        idx = (pred.softmax(-2) * class_preds.max(-1)[0].unsqueeze(-1)).argmax(-2)
        pred = torch.zeros_like(pred).bool()
        pred[idx, torch.arange(len(idx))] = True
        return pred

    def exact_match(self, pred, tgt):
        """Perfect hit to track assignment"""
        if len(tgt) == 0:
            return torch.tensor(torch.nan)
        return (pred == tgt).all(-1).float().mean()

    def eff(self, pred, tgt):
        """Efficiency to assign correct hit to track"""
        return ((pred & tgt).sum(-1) / tgt.sum(-1)).mean()

    def pur(self, pred, tgt):
        """Purity of assigned hits on tracks"""
        return ((pred & tgt).sum(-1) / pred.sum(-1)).mean()


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: Literal["AdamW", "Lion"] = "AdamW",
        mtl: bool = False,
        log_metrics_train=True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        # flavour label metrics
        self.obj_accuracy_micro = tm.classification.MulticlassAccuracy(num_classes=2, average="micro")
        self.obj_accuracy_macro = tm.classification.MulticlassAccuracy(num_classes=2, average="macro")

        self.mtl = mtl

        # If we are doing multi-task-learning, optimisation step must be done manually
        if mtl:
            self.automatic_optimization = False

        self.MI = MaskInference()

        # b hadron efficiency and fake rate
        self.eff = tm.classification.BinaryRecall()
        self.pur = tm.classification.BinaryPrecision()

        self.log_metrics_train = log_metrics_train

    def forward(self, inputs):
        return self.model(inputs)

    def predict(self, outputs):
        return self.model.predict(outputs)

    def log_losses(self, losses, stage):
        total_loss = 0

        for layer_name, loss_value in losses.items():
            self.log(f"{layer_name}", loss_value, sync_dist=True)
            total_loss += loss_value
            self.log(f"{stage}/{layer_name}_loss", loss_value, sync_dist=True)

        self.log(f"{stage}/loss", total_loss, sync_dist=True)
        return total_loss

    def log_custom_metrics(self, pred_valid, true_valid, pred_hit_masks, true_hit_masks, stage):
        # Just log predictions from the final layer

        # First log metrics that depend on outputs from multiple tasks
        # TODO: Make the task names configurable or match task names automatically

        # Calculate the true/false positive rates between the predicted and true masks
        # Number of hits that were correctly assigned to the track
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)

        # Number of predicted hits on the track
        hit_p = pred_hit_masks.sum(-1)

        # True number of hits on the track
        hit_t = true_hit_masks.sum(-1)

        # Calculate the efficiency and purity at differnt matching working points
        for wp in [0.5, 0.75, 1.0]:
            both_valid = true_valid & pred_valid

            effs = (hit_tp / hit_t >= wp) & both_valid
            purs = (hit_tp / hit_p >= wp) & both_valid

            roi_effs = effs.float().sum(-1) / true_valid.float().sum(-1)
            roi_purs = purs.float().sum(-1) / pred_valid.float().sum(-1)

            mean_eff = roi_effs.nanmean()
            mean_pur = roi_purs.nanmean()

            self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

        true_num = true_valid.sum(-1)
        pred_num = pred_valid.sum(-1)

        nh_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()

        self.log(f"{stage}/nh_per_particle", torch.mean(nh_per_true.float()), sync_dist=True)
        self.log(f"{stage}/nh_per_track", torch.mean(nh_per_pred.float()), sync_dist=True)

        self.log(f"{stage}/num_tracks", torch.mean(pred_num.float()), sync_dist=True)
        self.log(f"{stage}/num_particles", torch.mean(true_num.float()), sync_dist=True)

    def mask_metrics(self, preds, labels, pred_masks, truth_masks, pred_valid, truth_valid, stage, **kwargs):
        # include selection on predicted mask that it has >= 3 hits. reduces fake rate
        if pred_valid is not None:  # noqa
            pred_valid = pred_valid & (pred_masks.sum(-1) >= 3)
        else:
            pred_valid = pred_masks.sum(-1) >= 3

        # number of hits per track
        self.log(f"{stage}_nh_per_track", float(pred_masks.sum(-1).float().mean()), **kwargs)
        self.log(f"{stage}_nh_per_validtrack", pred_masks[truth_valid].sum(-1).float().mean(), **kwargs)
        self.log(f"{stage}_nh_per_invalidtrack", pred_masks[~truth_valid].sum(-1).float().mean(), **kwargs)

        # general metrics
        tp_hits = (pred_masks & truth_masks).sum(-1)
        per_track_eff = tp_hits / truth_masks.sum(-1)
        per_track_pur = tp_hits / pred_masks.sum(-1)

        # LHC reco eff/fake metrics
        lhc_track_reco = ((per_track_pur > 0.75) & pred_valid).sum()
        self.log(f"{stage}_p75_eff", lhc_track_reco / truth_valid.sum().float(), **kwargs)
        self.log(f"{stage}_p75_pur", lhc_track_reco / pred_valid.sum().float(), **kwargs)

        # double majority reco eff/fake metrics
        dm_track_reco = ((per_track_eff > 0.5) & (per_track_pur > 0.5) & pred_valid).sum()
        self.log(f"{stage}_ep50_eff", dm_track_reco / truth_valid.sum(), **kwargs)
        self.log(f"{stage}_ep50_pur", dm_track_reco / pred_valid.sum(), **kwargs)

        # print(f"{stage}_ep50_eff", dm_track_reco / truth_valid.sum())
        # print(f"{stage}_ep50_pur", dm_track_reco / pred_valid.sum())

        # double majority reco eff/fake metrics
        perf_track_reco = ((per_track_eff >= 1) & (per_track_pur >= 1) & pred_valid).sum()
        self.log(f"{stage}_ep100_eff", perf_track_reco / truth_valid.sum(), **kwargs)
        self.log(f"{stage}_ep100_pur", perf_track_reco / pred_valid.sum(), **kwargs)

        # mask metrics
        truth_masks = truth_masks[truth_valid]
        pred_masks = pred_masks[truth_valid]
        self.log(f"{stage}_mask_exact_match", self.MI.exact_match(pred_masks, truth_masks), **kwargs)

        # with the hit filter, can get 0 in the denominator here
        # should use the unfiltered target masks or only select targets with at least one hit
        recall_idx = truth_masks.sum(-1) > 0  # get truth masks with at least one hit
        self.log(f"{stage}_mask_recall", self.MI.eff(pred_masks[recall_idx], truth_masks[recall_idx]), **kwargs)
        pur_idx = pred_masks.sum(-1) > 0  # get reco masks with at least one predicted hit
        self.log(f"{stage}_mask_purity", self.MI.pur(pred_masks[pur_idx], truth_masks[pur_idx]), **kwargs)

    def log_metrics(self, preds, labels, stage, pad_mask=None):
        kwargs = {"sync_dist": True, "batch_size": 1}

        # input info
        self.log(f"{stage}_n_hits", float(preds["masks"].shape[-1]), **kwargs)
        self.log(f"{stage}_n_tracks", (labels["particle_valid"]).float().sum(), **kwargs)

        # skip detailed metrics for efficiency
        if stage == "train" and not self.log_metrics_train:
            return

        # get info
        track_class_probs = None
        pred_valid = None
        track_class_labels = labels["particle_valid"].squeeze()
        truth_valid = track_class_labels != 0
        if "class_probs" in preds:
            track_class_probs = preds["class_probs"].squeeze(0)
            track_class_preds = track_class_probs.argmax(-1)

            # object class prediction metrics
            self.obj_accuracy_micro(track_class_preds.view(-1), track_class_labels.view(-1))
            self.log(f"{stage}_obj_class_accuracy_micro", self.obj_accuracy_micro, **kwargs)
            self.obj_accuracy_macro(track_class_preds.view(-1), track_class_labels.view(-1))
            self.log(f"{stage}_obj_class_accuracy_macro", self.obj_accuracy_macro, **kwargs)

            # tracking efficiency and fake rate
            pred_valid = track_class_preds != 0
            self.eff(pred_valid, truth_valid)
            self.pur(pred_valid, truth_valid)

            self.log(f"{stage}_pur", self.pur, **kwargs)
            self.log(f"{stage}_eff", self.eff, **kwargs)

        # reco metrics
        pred_masks = preds["masks"].squeeze()  # remove batch index
        truth_masks = labels["particle_hit_valid"].squeeze()  # remove batch index

        # Disable MI if we are using OC, since we only get hard true/false predictions from it
        pred_masks_sigmoid = mask_from_logits(pred_masks, mode="sigmoid")
        if "class_probs" in preds:
            pred_masks_wargmax = mask_from_logits(pred_masks, mode="weighted_argmax", object_class_preds=track_class_probs)

        self.mask_metrics(preds, labels, pred_masks_sigmoid, truth_masks, pred_valid, truth_valid, f"{stage}", **kwargs)

        pred_hit_masks = pred_masks_sigmoid & pred_valid.unsqueeze(-1)
        true_hit_masks = truth_masks & track_class_labels.unsqueeze(-1)
        self.log_custom_metrics(pred_valid, track_class_labels, pred_hit_masks, true_hit_masks, stage)

        if "class_probs" in preds:
            self.mask_metrics(preds, labels, pred_masks_wargmax, truth_masks, pred_valid, truth_valid, f"{stage}_wargmax", **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # Get the model outputs
        preds = self.model(inputs)
        loss = self.model.loss(preds, targets)

        # Compute and log losses
        total_loss = self.log_losses(loss, "train")

        # Get the predictions from the model
        if batch_idx % self.trainer.log_every_n_steps == 0:  # avoid calling predict if possible
            self.log_metrics(preds, targets, "train")

        return {"loss": total_loss, **preds}

    def validation_step(self, batch):
        inputs, targets = batch

        # Get the raw model outputs
        preds = self.model(inputs, targets)
        loss = self.model.loss(preds, targets)

        # Compute and log losses
        total_loss = self.log_losses(loss, "val")

        # Get the predictions from the model
        self.log_metrics(preds, targets, "val")

        return {"loss": total_loss, **preds}

    def test_step(self, batch):
        inputs, targets = batch
        preds = self.model(inputs, targets)
        return self.model.predict(preds)

    def on_train_start(self):
        # Manually overwride the learning rate in case we are starting
        # from a checkpoint that had a LRS and now we want a flat LR
        if self.lrs_config.get("skip_scheduler"):
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lrs_config["initial"]

    def configure_optimizers(self):
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        opt = optimizer(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            # Configure the learning rate scheduler
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            sch = {"scheduler": sch, "interval": "step"}
            return [opt], [sch]
        print("Skipping learning rate scheduler.")
        return opt

    # def mlt_opt(self, losses, outputs):
    #     opt = self.optimizers()
    #     opt.zero_grad()

    #     for layer_name, layer_losses in losses.items():
    #         # Get a list of the features that are used by all of the tasks
    #         layer_feature_names = set()
    #         for task in self.model.tasks:
    #             layer_feature_names.update(task.inputs)

    #         # Remove any duplicate features that are used by multiple tasks
    #         layer_features = [outputs[layer_name][feature_name] for feature_name in layer_feature_names]

    #         # Perform the backward pass for this layer
    #         # For each layer we sum the losses from each task, so we get one loss per task
    #         layer_losses = [sum(losses[layer_name][task.name].values()) for task in self.model.tasks]

    #         mtl_backward(losses=layer_losses, features=layer_features, aggregator=UPGrad())

    #     opt.step()
