import torch
from lightning.pytorch.cli import ArgsType
from torch import nn

from hepattn.experiments.trackml.data import TrackMLDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class TrackMLFilter(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
    ):
        super().__init__(name, model, lrs_config, optimizer)

    def log_custom_metrics(self, preds, targets, stage):
        preds_unpacked = {}
        targets_unpacked = {}

        for task in self.model.tasks:
            if not all(hasattr(task, attr) for attr in ("input_object", "target_field")):
                continue
            if task.name not in preds["final"]:
                continue

            input_object = task.input_object
            expected_key = f"{input_object}_{task.target_field}"
            if expected_key not in preds["final"][task.name] or expected_key not in targets:
                continue

            pred = preds["final"][task.name][expected_key]
            true = targets[expected_key]
            preds_unpacked[input_object] = pred
            targets_unpacked[input_object] = true

            metrics = self._filter_metrics(pred, true, targets)
            self._log_filter_metrics(stage, input_object, metrics)

            # Preserve the historical TrackML hit_filter metric names.
            if task.name == "hit_filter":
                self._log_filter_metrics(stage, None, metrics)

        if "pixel" in preds_unpacked and "strip" in preds_unpacked:
            pred = torch.cat((preds_unpacked["pixel"], preds_unpacked["strip"]), dim=-1)
            true = torch.cat((targets_unpacked["pixel"], targets_unpacked["strip"]), dim=-1)
            self._log_filter_metrics(stage, "hit", self._filter_metrics(pred, true, targets))

    def _filter_metrics(self, pred, true, targets):
        tp = (pred & true).sum()
        tn = ((~pred) & (~true)).sum()

        return {
            # Log quantities based on the number of hits
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
            # other things
            "num_particles": targets["particle_valid"].float().sum(),
        }

    def _log_filter_metrics(self, stage, prefix, metrics):
        for metric_name, metric_value in metrics.items():
            if prefix is None:
                log_name = f"{stage}/{metric_name}"
            else:
                log_name = f"{stage}/{prefix}_{metric_name}"
            self.log(log_name, metric_value, sync_dist=True, batch_size=1)


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLFilter,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
