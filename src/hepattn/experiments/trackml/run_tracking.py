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
        pred_hit_masks = preds[f"track_{self.input_hit}_valid"][f"track{self.input_hit}_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets[f"particle_{self.input_hit}_valid"] & true_valid.unsqueeze(-1)

        # For pixel-only efficiency and purity comparison, filter to only pixel hits
        # This works by identifying which hits correspond to pixel features vs strip features
        # based on the feature structure and hit ordering in the merged dataset

        # Get the number of hits for each feature type
        pixel_hit_count = 0

        # Count hits for each feature type based on the input structure
        for feature_name in ["pixel", "hit"]:  # "hit" is used for pixel-only models
            if f"{feature_name}_valid" in targets:
                pixel_hit_count = targets[f"{feature_name}_valid"].sum().item()
                break

        # Create pixel hit mask based on the hit ordering
        # In the merged dataset, pixel hits come first, then strip hits
        total_hits = pred_hit_masks.shape[-1]
        pixel_hit_mask = torch.zeros(total_hits, dtype=torch.bool, device=pred_hit_masks.device)

        if pixel_hit_count > 0:
            # Pixel hits are the first pixel_hit_count hits in the merged dataset
            pixel_hit_mask[:pixel_hit_count] = True

        # Filter hit masks to only include pixel hits
        pred_pixel_masks = pred_hit_masks & pixel_hit_mask.unsqueeze(0).unsqueeze(0)
        true_pixel_masks = true_hit_masks & pixel_hit_mask.unsqueeze(0).unsqueeze(0)

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

        print("hit_tp", hit_tp)
        print("hit_t", hit_t)
        print("hit_p", hit_p)
        print("pixel_hit_tp", pixel_hit_tp)
        print("pixel_hit_t", pixel_hit_t)
        print("pixel_hit_p", pixel_hit_p)
        print("pixel_hit_count", pixel_hit_count)
        print("total_hits", total_hits)
        print("pixel_hit_mask.sum()", pixel_hit_mask.sum())
        print(true_valid.sum())
        print(pred_valid.sum())
        print((true_valid & pred_valid).sum())

        # Calculate the efficiency and purity at different matching working points
        for wp in [0.5, 0.75, 1.0]:
            both_valid = true_valid & pred_valid

            print("hit_tp / hit_t", hit_tp / hit_t)
            print("hit_tp / hit_p", hit_tp / hit_p)
            print("pixel_hit_tp / pixel_hit_t", pixel_hit_tp / pixel_hit_t)
            print("pixel_hit_tp / pixel_hit_p", pixel_hit_tp / pixel_hit_p)

            # All hits efficiency and purity
            effs = (hit_tp / hit_t >= wp) & both_valid
            purs = (hit_tp / hit_p >= wp) & both_valid

            # Pixel-only efficiency and purity
            pixel_effs = (pixel_hit_tp / pixel_hit_t >= wp) & both_valid
            pixel_purs = (pixel_hit_tp / pixel_hit_p >= wp) & both_valid

            print("effs.float().sum(-1)", effs.float().sum(-1))
            print("true_valid.float().sum(-1)", true_valid.float().sum(-1))
            print("purs.float().sum(-1)", purs.float().sum(-1))
            print("pred_valid.float().sum(-1)", pred_valid.float().sum(-1))
            print("pixel_effs.float().sum(-1)", pixel_effs.float().sum(-1))
            print("pixel_purs.float().sum(-1)", pixel_purs.float().sum(-1))

            roi_effs = effs.float().sum(-1) / true_valid.float().sum(-1)
            roi_purs = purs.float().sum(-1) / pred_valid.float().sum(-1)
            roi_pixel_effs = pixel_effs.float().sum(-1) / true_valid.float().sum(-1)
            roi_pixel_purs = pixel_purs.float().sum(-1) / pred_valid.float().sum(-1)

            print("roi_effs", roi_effs)
            print("roi_purs", roi_purs)
            print("roi_pixel_effs", roi_pixel_effs)
            print("roi_pixel_purs", roi_pixel_purs)

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
