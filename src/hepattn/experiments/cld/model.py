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

    def log_custom_metrics(self, preds, targets, stage):
        # Just log predictions from the final layer
        preds = preds["final"]

        hits = [
            "vtxd",
            "trkr",
            "sihit",
            "ecal",
            "hcal",
            "vtb",
            "muon",
        ]

        for hit in hits:
            if f"flow_{hit}_assignment" not in preds:
                continue

            # Set the masks of any flow slots that are not used as null
            pred_hit_masks = preds[f"flow_{hit}_assignment"][f"flow_{hit}_valid"]
            true_hit_masks = targets[f"particle_{hit}_valid"]

            pred_valid = preds["flow_valid"]["flow_valid"] & (pred_hit_masks.sum(-1) > 0)
            true_valid = targets["particle_valid"] & (true_hit_masks.sum(-1) > 0)

            # Mask out hits that are not on a valid object slot
            pred_hit_masks &= pred_valid.unsqueeze(-1)
            true_hit_masks &= true_valid.unsqueeze(-1)

            # Calculate the true/false positive rates between the predicted and true masks
            # Number of hits that were correctly assigned to the flow
            hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)

            # Number of predicted hits on the flow
            hit_p = pred_hit_masks.sum(-1)

            # True number of hits on the particle
            hit_t = true_hit_masks.sum(-1)

            # Calculate the efficiency and purity at differnt matching working points
            for wp in [0.5, 0.75, 1.0]:
                both_valid = true_valid & pred_valid

                # Whether a truth object is efficient
                effs = ((hit_tp / hit_t) >= wp) & both_valid

                # Whether a pred object is pure / not fake
                purs = ((hit_tp / hit_p) >= wp) & both_valid

                # Calculate the event efficiency / purity
                eff = effs.float().sum(-1) / true_valid.float().sum(-1)
                pur = purs.float().sum(-1) / pred_valid.float().sum(-1)

                self.log(f"{stage}/p{wp}_{hit}_eff", eff.mean())
                self.log(f"{stage}/p{wp}_{hit}_pur", pur.mean())

                # Log some counting info
                pred_num = pred_valid.sum(-1)
                true_num = true_valid.sum(-1)

                num_hits_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()
                num_hits_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()

                self.log(f"{stage}/num_{hit}_per_flow", torch.mean(num_hits_per_pred.float()))
                self.log(f"{stage}/num_{hit}_per_part", torch.mean(num_hits_per_true.float()))

                self.log(f"{stage}/num_flows", torch.mean(pred_num.float()))
                self.log(f"{stage}/num_parts", torch.mean(true_num.float()))
