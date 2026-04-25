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
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

    @staticmethod
    def _collect_pred_hit_masks(task_preds: dict[str, torch.Tensor]) -> torch.Tensor | None:
        masks = [value.bool() for key, value in task_preds.items() if key.startswith("track_") and key.endswith("_valid")]
        if not masks:
            return None
        if len(masks) == 1:
            return masks[0]
        return torch.cat(masks, dim=-1)

    @staticmethod
    def _collect_pred_true_hit_masks(
        task_preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pred_masks: list[torch.Tensor] = []
        true_masks: list[torch.Tensor] = []

        for key, pred_mask in task_preds.items():
            if not (key.startswith("track_") and key.endswith("_valid")):
                continue

            constituent = key[len("track_") : -len("_valid")]
            target_key = f"particle_{constituent}_valid"
            if target_key not in targets:
                if "particle_hit_valid" in targets and targets["particle_hit_valid"].shape == pred_mask.shape:
                    target_key = "particle_hit_valid"
                else:
                    continue

            pred_masks.append(pred_mask.bool())
            true_masks.append(targets[target_key].bool())

        if not pred_masks:
            return None, None
        if len(pred_masks) == 1:
            return pred_masks[0], true_masks[0]
        return torch.cat(pred_masks, dim=-1), torch.cat(true_masks, dim=-1)

    def _augment_test_targets_with_event_counts(
        self,
        _inputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        augmented_targets = targets.copy()

        sample_tensor = next(value for value in targets.values() if isinstance(value, torch.Tensor))
        batch_size = sample_tensor.shape[0]
        device = sample_tensor.device

        query_mask = targets.get("query_mask")
        if query_mask is not None:
            augmented_targets["num_initialized_queries"] = query_mask.bool().sum(dim=-1, keepdim=True).to(torch.int32)
        else:
            decoder = getattr(getattr(self, "model", None), "decoder", None)
            num_queries = getattr(decoder, "_num_queries", None)
            if num_queries is not None:
                augmented_targets["num_initialized_queries"] = torch.full(
                    (batch_size, 1),
                    int(num_queries),
                    dtype=torch.int32,
                    device=device,
                )

        particle_valid = targets.get("particle_valid")
        if particle_valid is not None:
            augmented_targets["num_reconstructable_particles"] = particle_valid.bool().sum(dim=-1, keepdim=True).to(torch.int32)

        input_names = list(getattr(getattr(self, "model", None), "input_names", []))
        input_hit_counts: list[torch.Tensor] = []
        for input_name in input_names:
            valid_key = f"{input_name}_valid"
            if valid_key not in targets:
                continue
            hit_count = targets[valid_key].bool().sum(dim=-1, keepdim=True).to(torch.int32)
            augmented_targets[f"num_{input_name}_hits"] = hit_count
            input_hit_counts.append(hit_count)

        if "key_valid" in targets:
            augmented_targets["num_input_hits"] = targets["key_valid"].bool().sum(dim=-1, keepdim=True).to(torch.int32)
        elif input_hit_counts:
            augmented_targets["num_input_hits"] = torch.stack(input_hit_counts, dim=0).sum(dim=0)

        if particle_valid is not None and "particle_key_valid" in targets:
            truth_hits = (
                targets["particle_key_valid"].bool() & particle_valid.bool().unsqueeze(-1)
            ).sum(dim=(-1, -2), keepdim=False)
            augmented_targets["num_truth_hits_on_reconstructable_tracks"] = truth_hits.unsqueeze(-1).to(torch.int32)

        return augmented_targets

    def log_custom_metrics(self, preds, targets, stage):
        query_mask = targets.get("query_mask")
        if query_mask is not None:
            query_mask = query_mask.bool()

        # log intermediate layer mask predictions
        for layer_name, layer_preds in preds.items():
            # Skip layers that don't have track_hit_valid task (e.g., encoder layer)
            if "track_hit_valid" not in layer_preds:
                continue
            mask = self._collect_pred_hit_masks(layer_preds["track_hit_valid"])
            if mask is not None:
                num_valid = mask.sum(-1).float()
                frac_valid = num_valid / mask.shape[-1]

                if query_mask is None:
                    self.log(f"{stage}/{layer_name}_avg_num_valid_hits", torch.mean(num_valid), sync_dist=True)
                    self.log(f"{stage}/{layer_name}_avg_frac_valid_hits", torch.mean(frac_valid), sync_dist=True)
                else:
                    valid_num_valid = num_valid[query_mask]
                    valid_frac_valid = frac_valid[query_mask]
                    if valid_num_valid.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_num_valid_hits", valid_num_valid.mean(), sync_dist=True)
                    if valid_frac_valid.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_frac_valid_hits", valid_frac_valid.mean(), sync_dist=True)

        # Just log predictions from the final layer
        preds = preds["final"]

        # First log metrics that depend on outputs from multiple tasks
        # TODO: Make the task names configurable or match task names automatically
        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        if query_mask is not None:
            pred_valid = pred_valid & query_mask

        # Set the masks of any track slots that are not used as null
        pred_mask, true_mask = self._collect_pred_true_hit_masks(preds["track_hit_valid"], targets)
        if pred_mask is None or true_mask is None:
            return
        pred_hit_masks = pred_mask & pred_valid.unsqueeze(-1)
        true_hit_masks = true_mask & true_valid.unsqueeze(-1)

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

            self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

        pred_num = pred_valid.sum(-1)

        nh_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()

        self.log(f"{stage}/nh_per_particle", torch.mean(nh_per_true.float()), sync_dist=True)
        self.log(f"{stage}/nh_per_track", torch.mean(nh_per_pred.float()), sync_dist=True)

        self.log(f"{stage}/num_tracks", torch.mean(pred_num.float()), sync_dist=True)
        true_num = true_valid.sum(-1)
        self.log(f"{stage}/num_particles", torch.mean(true_num.float()), sync_dist=True)

        num_hits_total = float(pred_hit_masks.shape[-1])
        num_hits_valid = true_hit_masks.sum().float().item()
        num_hits_noise = num_hits_total - num_hits_valid
        self.log(f"{stage}/num_hits", num_hits_total, sync_dist=True)
        self.log(f"{stage}/num_hits_valid", num_hits_valid, sync_dist=True)
        self.log(f"{stage}/num_hits_noise", num_hits_noise, sync_dist=True)

    def test_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss to also run matching and expose query_mask/sorted targets.
        outputs, targets, losses = self.model.loss(outputs, targets)

        preds = self.model.predict(outputs)
        augmented_targets = self._augment_test_targets_with_event_counts(inputs, targets)

        return outputs, preds, losses, augmented_targets


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLTracker,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
