from torch import nn

from hepattn.models.wrapper import ModelWrapper


class TIDEModel(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

    def log_custom_metrics(self, preds, targets, stage):
        # Just log predictions from the final layer
        preds = preds["final"]

        hits = ["pix", "sct"]

        pred_valid = preds["pred_valid"]["pred_valid"]
        true_valid = targets["sudo_valid"]

        for hit in hits:
            if f"pred_{hit}_assignment" not in preds:
                continue

            # Set the masks of any track slots that are not used as null
            pred_hit_masks = preds[f"pred_{hit}_assignment"][f"pred_{hit}_valid"] & pred_valid.unsqueeze(-1)
            true_hit_masks = targets[f"sudo_{hit}_valid"]

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

                self.log(f"{stage}_p{wp}_{hit}_eff", mean_eff)
                self.log(f"{stage}_p{wp}_{hit}_pur", mean_pur)
