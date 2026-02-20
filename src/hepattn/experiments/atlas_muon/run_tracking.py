"""Run/training entrypoint for tracking experiments.

This module exposes `TrackMLTracker`, a `ModelWrapper` subclass used for
logging a standard set of tracking metrics (efficiency, purity, per-track
and per-hit statistics) during training/validation/test. The `main`
function hooks this wrapper into the project's CLI.
"""

import comet_ml  # noqa: F401
import torch
from lightning.pytorch.cli import ArgsType
from torch import nn

from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class TrackMLTracker(ModelWrapper):
    """Wrapper for tracking experiments that logs tracking-specific metrics.

    The `log_custom_metrics` method expects the model wrapper to supply
    `preds` and `targets` dictionaries with consistent task keys (e.g.
    `track_valid`, `track_hit_valid`, `particle_valid`, `particle_hit_valid`).
    It computes per-batch and per-track summaries (efficiency, purity, counts)
    and logs them via Lightning's logger.
    """

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
        """Compute and log per-batch tracking metrics.

        Args:
            preds: Nested dict of model predictions (expected key: 'final').
            targets: Nested dict of ground-truth tensors.
            stage: Logging stage prefix (e.g. 'train' or 'val').
        """
        # Extract final-stage predictions and required target masks
        preds = preds["final"]
        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        # Mask predicted hits by predicted-valid track slots and compare to truth
        pred_hit_masks = preds["track_hit_valid"]["track_hit_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets["particle_hit_valid"] & true_valid.unsqueeze(-1)

        # Per-track true positives, predicted counts, and true counts
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)
        hit_p = pred_hit_masks.sum(-1)
        hit_t = true_hit_masks.sum(-1)

        # Compute efficiency and purity at several working points and log
        both_valid = true_valid & pred_valid
        for wp in [0.25, 0.5, 0.75, 1.0]:
            effs = (hit_tp / hit_t >= wp) & both_valid
            purs = (hit_tp / hit_p >= wp) & both_valid

            roi_effs = effs.float().sum(-1) / true_valid.float().sum(-1)
            roi_purs = purs.float().sum(-1) / pred_valid.float().sum(-1)

            mean_eff = roi_effs.nanmean()
            mean_pur = roi_purs.nanmean()

            self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

        # Track-level confusion components (TP/FP/FN/TN) and aggregated metrics
        track_tp = (pred_valid & true_valid).float()
        track_fp = (pred_valid & ~true_valid).float()
        track_fn = (~pred_valid & true_valid).float()
        track_tn = (~pred_valid & ~true_valid).float()

        batch_track_effs = []
        batch_track_fake_rates = []
        for batch_idx in range(true_valid.shape[0]):
            tp_batch = track_tp[batch_idx].sum()
            fp_batch = track_fp[batch_idx].sum()
            fn_batch = track_fn[batch_idx].sum()
            tn_batch = track_tn[batch_idx].sum()

            if (tp_batch + fn_batch) > 0:
                batch_track_effs.append(tp_batch / (tp_batch + fn_batch))
            if (fp_batch + tn_batch) > 0 and fp_batch >= 3:
                batch_track_fake_rates.append(fp_batch / (fp_batch + tn_batch))

        if batch_track_effs:
            self.log(f"{stage}/track_efficiency", torch.stack(batch_track_effs).mean(), sync_dist=True)
        if batch_track_fake_rates:
            self.log(f"{stage}/track_fake_rate", torch.stack(batch_track_fake_rates).mean(), sync_dist=True)

        # Aggregate hit-level statistics and log
        true_pos_hits = (true_hit_masks & pred_hit_masks).sum() / torch.sum(true_hit_masks)
        false_pos_hits = (pred_hit_masks.sum() - (true_hit_masks & pred_hit_masks).sum()) / torch.sum(~true_hit_masks)

        self.log(f"{stage}/true_pos_hits", true_pos_hits, sync_dist=True)
        self.log(f"{stage}/false_pos_hits", false_pos_hits, sync_dist=True)

        nh_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()

        self.log(f"{stage}/nh_per_particle", torch.mean(nh_per_true.float()), sync_dist=True)
        self.log(f"{stage}/nh_per_track", torch.mean(nh_per_pred.float()), sync_dist=True)

        self.log(f"{stage}/num_tracks", torch.mean(pred_valid.sum(-1).float()), sync_dist=True)
        self.log(f"{stage}/num_particles", torch.mean(true_valid.sum(-1).float()), sync_dist=True)


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLTracker,
        datamodule_class=AtlasMuonDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
