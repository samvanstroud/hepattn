from torch import nn

from hepattn.models.wrapper import ModelWrapper


class TrackMLFilter(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
    ):
        super().__init__(name, model, lrs_config, optimizer)

    def log_compound_metrics(self, preds, targets, stage):
        pred = preds["final"]["hit_filter"]["hit_on_valid_particle"]
        true = targets["hit_on_valid_particle"]

        tp = (pred * true).sum()
        tn = ((~pred) * (~true)).sum()

        metrics = {
            # Log quanties based on the number of hits
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

        # Now actually log the metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"{stage}/{metric_name}", metric_value, sync_dist=True, batch_size=1)
