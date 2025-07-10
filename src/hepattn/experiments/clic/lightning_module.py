import torch  # noqa: F401
import torchmetrics as tm
from torch import nn

from hepattn.experiments.clic.metrics import MaskInference
from hepattn.models.wrapper import ModelWrapper


class MPflow(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)
        self.MI = MaskInference

        self.obj_accuracy_micro = tm.classification.MulticlassAccuracy(num_classes=6, average="micro")
        self.obj_accuracy_macro = tm.classification.MulticlassAccuracy(num_classes=6, average="macro")

        self.eff = tm.classification.BinaryRecall()
        self.fake = tm.classification.BinaryRecall()
        self.pur = tm.classification.BinaryPrecision()

    def log_custom_metrics(self, preds, labels, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}

        # skip detailed metrics for efficiency
        if stage == "train":
            return

        # get info
        pred_valid = None
        particle_class_labels = labels["particle_class"].squeeze()
        truth_valid = particle_class_labels < 5
        for key, value in preds.items():
            for task_name, task_value in value.items():
                if task_name == "classification":
                    particle_class_preds = task_value["pflow_class"].squeeze()  # remove batch index

                    # object class prediction metrics
                    self.obj_accuracy_micro(particle_class_preds.view(-1), particle_class_labels.view(-1))
                    self.log(f"{stage}/{key}_obj_class_accuracy_micro", self.obj_accuracy_micro, **kwargs)
                    self.obj_accuracy_macro(particle_class_preds.view(-1), particle_class_labels.view(-1))
                    self.log(f"{stage}/{key}_obj_class_accuracy_macro", self.obj_accuracy_macro, **kwargs)

                    # tracking efficiency and fake rate
                    pred_valid = particle_class_preds < 5
                    self.eff(pred_valid, truth_valid)
                    self.pur(pred_valid, truth_valid)

                    self.log(f"{stage}/{key}_pur", self.pur, **kwargs)
                    self.log(f"{stage}/{key}_eff", self.eff, **kwargs)
                if task_name == "mask":
                    pred_masks = task_value["pflow_node_valid"].squeeze()  # remove batch index
                    truth_masks = labels["particle_node_valid"].squeeze()  # remove batch index

                    self.mask_metrics(pred_masks, truth_masks, pred_valid, truth_valid, f"{stage}/{key}", **kwargs)

        # regression metrics
        # if "regression" not in preds:
        #     return

        # valid_idx = ~torch.isnan(labels["regression"]).all(-1)
        # valid_idx &= ~(labels["regression"] == 0).all(-1)
        # reg_idx = valid_idx & pred_valid
        # for i, t in enumerate(self.reg_tgts):
        #     reg_preds = preds["regression"][reg_idx][..., i]
        #     targets = labels["regression"][reg_idx][..., i]

        #     reg_preds = reg_preds
        #     targets = targets

        #     # compute and log absolute error
        #     mae = nn.functional.l1_loss(reg_preds, targets)
        #     self.log(f"{stage}_{t}_mae", mae, **kwargs)

    def mask_metrics(self, pred_masks, truth_masks, pred_valid, truth_valid, prefix, **kwargs):
        # Valid objects should be connected at least to one node
        if pred_valid is not None:  # noqa
            pred_valid = pred_valid & (pred_masks.sum(-1) >= 1)
        else:
            pred_valid = pred_masks.sum(-1) >= 1

        # number of nodes per track
        # self.log(f"{prefix}_nn_per_particle", float(pred_masks.sum(-1).float().mean()), **kwargs)
        # self.log(f"{prefix}_nn_per_validparticle", pred_masks[truth_valid].sum(-1).float().mean(), **kwargs)
        # self.log(f"{prefix}_nn_per_invalidparticle", pred_masks[~truth_valid].sum(-1).float().mean(), **kwargs)

        # mask metrics
        truth_masks = truth_masks[truth_valid]
        pred_masks = pred_masks[truth_valid]

        self.log(f"{prefix}_mask_exact_match", self.MI.exact_match(pred_masks, truth_masks), **kwargs)

        # with the hit filter, can get 0 in the denominator here
        # should use the unfiltered target masks or only select targets with at least one hit
        recall_idx = truth_masks.sum(-1) > 0  # get truth masks with at least one hit
        self.log(f"{prefix}_mask_recall", self.MI.eff(pred_masks[recall_idx], truth_masks[recall_idx]), **kwargs)
        pur_idx = pred_masks.sum(-1) > 0  # get reco masks with at least one predicted hit
        self.log(f"{prefix}_mask_purity", self.MI.pur(pred_masks[pur_idx], truth_masks[pur_idx]), **kwargs)
