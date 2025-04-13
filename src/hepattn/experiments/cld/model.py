import torch
from torch import nn

from hepattn.models.wrapper import ModelWrapper


class CLDReconstructor(ModelWrapper):
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

        hits = [
            "sihit",
            "ecal",
            "hcal",
        ]

        pred_valid = preds["flow_valid"]["flow_valid"]
        true_valid = targets["particle_valid"]

        for hit in hits:
            # Set the masks of any flow slots that are not used as null
            pred_hit_masks = preds[f"flow_{hit}_assignment"][f"flow_{hit}_valid"] & pred_valid.unsqueeze(-1)
            true_hit_masks = targets[f"particle_{hit}_valid"]

            # Calculate the true/false positive rates between the predicted and true masks
            # Number of hits that were correctly assigned to the flow
            hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)

            # Number of predicted hits on the flow
            hit_p = pred_hit_masks.sum(-1)

            # True number of hits on the flow
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

                pred_num_flows = pred_valid.sum(-1)
                true_num_flows = true_valid.sum(-1)

                pred_num_hits_per_flow = pred_hit_masks.sum(-1).float()[pred_valid].mean()
                true_num_hits_per_flow = true_hit_masks.sum(-1).float()[true_valid].mean()

                pred_num_hits_per_flow_all = pred_hit_masks.sum(-1).float().mean()
                true_num_hits_per_flow_all = true_hit_masks.sum(-1).float().mean()

                self.log(f"{stage}_pred_num_{hit}_per_flow", torch.mean(pred_num_hits_per_flow.float()))
                self.log(f"{stage}_true_num_{hit}_per_flow", torch.mean(true_num_hits_per_flow.float()))

                self.log(f"{stage}_pred_num_{hit}_per_flow_all", torch.mean(pred_num_hits_per_flow_all.float()))
                self.log(f"{stage}_true_num_{hit}_per_flow_all", torch.mean(true_num_hits_per_flow_all.float()))

                self.log(f"{stage}_num_pred_flows", torch.mean(pred_num_flows.float()))
                self.log(f"{stage}_num_true_flows", torch.mean(true_num_flows.float()))
