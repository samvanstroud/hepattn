"""Entry point and utilities for running hit-filtering experiments.

This module provides `AtlasMuonFilter`, a thin `ModelWrapper` specialization
that logs classification metrics specific to hit-filtering experiments, and a
`main` function used by the CLI. The `log_custom_metrics` method produces
per-batch metrics (counts, precision/recall, AUC when logits are available)
and logs them using the Lightning logger.
"""

import comet_ml  # noqa: F401
import torch
from lightning.pytorch.cli import ArgsType
from torch import nn
from torchmetrics.functional import auroc

from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class AtlasMuonFilter(ModelWrapper):
    """Wrapper for running hit-filtering experiments with logging helpers.

    The wrapper delegates model behaviour to `ModelWrapper` and implements
    `log_custom_metrics` to compute and record a set of common metrics
    useful for hit-filtering evaluation (counts, precision/recall, AUC).
    """

    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "Lion",
    ):
        super().__init__(name, model, lrs_config, optimizer)

    def log_custom_metrics(self, preds, targets, stage, outputs=None):
        """Compute and log per-batch hit-filtering metrics.

        Args:
            preds: Nested dict of predictions produced by the model wrapper.
            targets: Nested dict of ground-truth tensors.
            stage: String stage name used as a logging prefix (e.g. 'train').
            outputs: Optional raw logits; if present, AUC will be computed.
        """
        batch_size = targets["hit_on_valid_particle"].shape[0]

        # Select only the valid (non-padded) hit positions for this batch.
        pred = preds["final"]["hit_filter"]["hit_on_valid_particle"][targets["hit_valid"]]
        true = targets["hit_on_valid_particle"][targets["hit_valid"]]

        # True/false positives and negatives used for summary metrics.
        tp = (pred * true).sum()
        tn = ((~pred) * (~true)).sum()

        # Optionally compute AUC when raw logits are available in `outputs`.
        auc = None
        if outputs is not None:
            pred_logits = outputs["final"]["hit_filter"]["hit_logit"][targets["hit_valid"]]
            pred_probs = torch.sigmoid(pred_logits)
            auc = auroc(pred_probs.flatten(), true.flatten().long(), task="binary")

        metrics = {
            # Counts and set-based summaries
            "nh_total_pre": float(pred.numel()),
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

        if auc is not None:
            metrics["auc"] = auc

        # Log metrics with Lightning's logger; aggregate per-epoch across batch_size.
        for metric_name, metric_value in metrics.items():
            self.log(
                f"{stage}/{metric_name}",
                metric_value,
                sync_dist=True,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )


def main(args: ArgsType = None) -> None:
    CLI(model_class=AtlasMuonFilter, datamodule_class=AtlasMuonDataModule, args=args, parser_kwargs={"default_env": True}, save_config_callback=None)


if __name__ == "__main__":
    main()
