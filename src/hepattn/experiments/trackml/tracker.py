import torch
from torch import nn

from hepattn.models.wrapper import ModelWrapper


class TrackMLTracker(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

    def log_compound_metrics(self, preds, targets, stage):  # noqa: PLR0914
        # Just log predictions from the final layer
        preds = preds["final"]

        # First log metrics that depend on outputs from multiple tasks
        # TODO: Make the task names configurable or match task names automatically
        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        # Set the masks of any track slots that are not used as null
        pred_hit_masks = preds["track_hit_valid"]["track_hit_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets["particle_hit_valid"] & true_valid.unsqueeze(-1)

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

            self.log(f"{stage}/p{wp}_eff", mean_eff)
            self.log(f"{stage}/p{wp}_pur", mean_pur)

        true_num = true_valid.sum(-1)
        pred_num = pred_valid.sum(-1)

        nh_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()

        self.log(f"{stage}/nh_per_particle", torch.mean(nh_per_true.float()))
        self.log(f"{stage}/nh_per_track", torch.mean(nh_per_pred.float()))

        self.log(f"{stage}/num_particles", torch.mean(pred_num.float()))
        self.log(f"{stage}/num_tracks", torch.mean(true_num.float()))
