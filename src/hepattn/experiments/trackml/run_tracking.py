import comet_ml  # noqa: F401
import torch
from lightning.pytorch.cli import ArgsType
from torch import nn

from hepattn.experiments.trackml.data import TrackMLDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class TrackMLTracker(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
        input_hit: str = "hit",
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)
        self.input_hit = input_hit

    def log_custom_metrics(self, preds, targets, stage):
        # Just log predictions from the final layer
        preds = preds["final"]

        # First log metrics that depend on outputs from multiple tasks
        # TODO: Make the task names configurable or match task names automatically
        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        # Set the masks of any track slots that are not used as null
        pred_hit_masks = preds[f"track_{self.input_hit}_valid"][f"track_{self.input_hit}_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets[f"particle_{self.input_hit}_valid"] & true_valid.unsqueeze(-1)

        # For pixel-only efficiency and purity comparison, filter to only pixel hits
        # This works by identifying which hits correspond to pixel features vs strip features
        # based on the feature structure and hit ordering in the merged dataset

        # Build a per-sample pixel mask (B, total_hits), assuming pixel hits are first in the merge order
        batch_size = pred_hit_masks.shape[0]
        total_hits = pred_hit_masks.shape[-1]
        # Prefer explicit pixel counts if available; fall back to generic hit counts (for pixel-only models)
        if "pixel_valid" in targets:
            pixel_counts = targets["pixel_valid"].sum(dim=-1).to(torch.long)  # (B,)
        elif "hit_valid" in targets:
            pixel_counts = targets["hit_valid"].sum(dim=-1).to(torch.long)  # (B,)
        else:
            # Fallback: treat all hits as non-pixel
            pixel_counts = torch.zeros(batch_size, dtype=torch.long, device=pred_hit_masks.device)

        pixel_hit_mask = torch.zeros(batch_size, total_hits, dtype=torch.bool, device=pred_hit_masks.device)
        for b in range(batch_size):
            count_b = int(pixel_counts[b].item())
            if count_b > 0:
                pixel_hit_mask[b, :count_b] = True

        # Filter hit masks to only include pixel hits
        pred_pixel_masks = pred_hit_masks & pixel_hit_mask.unsqueeze(1)
        true_pixel_masks = true_hit_masks & pixel_hit_mask.unsqueeze(1)

        # Calculate the true/false positive rates between the predicted and true masks
        # Number of hits that were correctly assigned to the track (all hits)
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)

        # Number of predicted hits on the track (all hits)
        hit_p = pred_hit_masks.sum(-1)

        # True number of hits on the track (all hits)
        hit_t = true_hit_masks.sum(-1)

        # Calculate pixel-only metrics for direct comparison between pixel and strip models
        # Number of pixel hits that were correctly assigned to the track
        pixel_hit_tp = (pred_pixel_masks & true_pixel_masks).sum(-1)

        # Number of predicted pixel hits on the track
        pixel_hit_p = pred_pixel_masks.sum(-1)

        # True number of pixel hits on the track
        pixel_hit_t = true_pixel_masks.sum(-1)


        # Calculate the efficiency and purity at different matching working points
        eps = 1e-6
        for wp in [0.1, 0.5, 0.75, 1.0]:
            both_valid = true_valid & pred_valid


            # All hits efficiency and purity (safe division)
            effs = (hit_tp.float() / (hit_t.float() + eps) >= wp) & both_valid
            purs = (hit_tp.float() / (hit_p.float() + eps) >= wp) & both_valid

            # Pixel-only efficiency and purity (safe division)
            pixel_effs = (pixel_hit_tp.float() / (pixel_hit_t.float() + eps) >= wp) & both_valid
            pixel_purs = (pixel_hit_tp.float() / (pixel_hit_p.float() + eps) >= wp) & both_valid
            roi_effs = effs.float().sum(-1) / (true_valid.float().sum(-1) + eps)
            roi_purs = purs.float().sum(-1) / (pred_valid.float().sum(-1) + eps)
            roi_pixel_effs = pixel_effs.float().sum(-1) / (true_valid.float().sum(-1) + eps)
            roi_pixel_purs = pixel_purs.float().sum(-1) / (pred_valid.float().sum(-1) + eps)

            mean_eff = roi_effs.nanmean()
            mean_pur = roi_purs.nanmean()
            mean_pixel_eff = roi_pixel_effs.nanmean()
            mean_pixel_pur = roi_pixel_purs.nanmean()

            # Log all-hits metrics
            self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

            # Log pixel-only metrics for direct comparison between pixel and strip models
            self.log(f"{stage}/p{wp}_pixel_eff", mean_pixel_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pixel_pur", mean_pixel_pur, sync_dist=True)

        true_num = true_valid.sum(-1)
        pred_num = pred_valid.sum(-1)

        nh_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()

        # Pixel-only hit counts
        nh_per_true_pixel = true_pixel_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred_pixel = pred_pixel_masks.sum(-1).float()[pred_valid].mean()

        # Log all-hits metrics
        self.log(f"{stage}/nh_per_particle", torch.mean(nh_per_true.float()), sync_dist=True)
        self.log(f"{stage}/nh_per_track", torch.mean(nh_per_pred.float()), sync_dist=True)

        # Log pixel-only hit count metrics for direct comparison
        self.log(f"{stage}/nh_per_particle_pixel", torch.mean(nh_per_true_pixel.float()), sync_dist=True)
        self.log(f"{stage}/nh_per_track_pixel", torch.mean(nh_per_pred_pixel.float()), sync_dist=True)

        self.log(f"{stage}/num_tracks", torch.mean(pred_num.float()), sync_dist=True)
        self.log(f"{stage}/num_particles", torch.mean(true_num.float()), sync_dist=True)


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLTracker,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
