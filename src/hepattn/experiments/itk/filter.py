import torch
from torch import nn

from hepattn.models.wrapper import ModelWrapper


class ITkFilter(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
    ):
        super().__init__(name, model, lrs_config, optimizer)

    def log_custom_metrics(self, preds, targets, stage):
        # Get the final predictions from the hit filter task
        # Calculate metrics for the combined hits also

        preds_unpacked = {}
        targets_unpacked = {}

        # Pick out just the final predictions
        for hit in ["pixel", "strip"]:
            if f"{hit}_filter" in preds["final"]:
                preds_unpacked[hit] = preds["final"][f"{hit}_filter"][f"{hit}_on_valid_particle"]
                targets_unpacked[hit] = targets[f"{hit}_on_valid_particle"]

        # If we have both pixel and strip hits, create a new hit collection which combines
        # both of them so we can easilly look at overall hit metrics
        if "pixel" in preds_unpacked and "strip" in preds_unpacked:
            preds_unpacked["hit"] = torch.cat((preds_unpacked["pixel"], preds_unpacked["strip"]), dim=-1)
            targets_unpacked["hit"] = torch.cat([targets_unpacked["pixel"], targets_unpacked["strip"]], dim=-1)

        for hit in preds_unpacked:  # noqa: PLC0206
            pred = preds_unpacked[hit]
            true = targets_unpacked[hit]

            tp = (pred & true).sum()
            tn = ((~pred) & (~true)).sum()

            metrics = {
                # Log quanties based on the number of hits
                "total_pre": float(pred.shape[1]),
                "total_post": float(pred.sum()),
                "pred_true": pred.float().sum(),
                "pred_false": (~pred).float().sum(),
                "valid_pre": true.float().sum(),
                "valid_post": (pred & true).float().sum(),
                "noise_pre": (~true).float().sum(),
                "noise_post": (pred & ~true).float().sum(),
                # Standard binary classification metrics
                "acc": (pred == true).half().mean(),
                "valid_recall": tp / true.sum(),
                "valid_precision": tp / pred.sum(),
                "noise_recall": tn / (~true).sum(),
                "noise_precision": tn / (~pred).sum(),
            }

            # Now actually log the metrics
            for metric_name, metric_value in metrics.items():
                self.log(f"{stage}/{hit}_{metric_name}", metric_value, sync_dist=True, batch_size=1)
