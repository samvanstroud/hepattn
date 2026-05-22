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
    def _track_hit_mask_pred_map(layer_preds: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pred_map: dict[str, torch.Tensor] = {}
        for task_preds in layer_preds.values():
            if not isinstance(task_preds, dict):
                continue
            for key, value in task_preds.items():
                if key == "track_valid":
                    continue
                if key.startswith("track_") and key.endswith("_valid"):
                    constituent = key[len("track_") : -len("_valid")]
                    pred_map[constituent] = value

        if len(pred_map) > 1 and "key" in pred_map:
            pred_map.pop("key")

        return pred_map

    @classmethod
    def _collect_pred_hit_masks(cls, layer_preds: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor | None:
        masks = [value.bool() for value in cls._track_hit_mask_pred_map(layer_preds).values()]
        if not masks:
            return None
        if len(masks) == 1:
            return masks[0]
        return torch.cat(masks, dim=-1)

    @classmethod
    def _pred_true_hit_mask_map(
        cls,
        layer_preds: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        mask_map: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        for constituent, pred_mask in cls._track_hit_mask_pred_map(layer_preds).items():
            target_key = None
            candidate_keys = [f"particle_{constituent}_hit_valid", f"particle_{constituent}_valid", "particle_hit_valid"]
            for key in candidate_keys:
                target = targets.get(key)
                if target is not None and target.shape == pred_mask.shape:
                    target_key = key
                    break

            if target_key is None:
                continue

            mask_map[constituent] = (pred_mask.bool(), targets[target_key].bool())

        return mask_map

    @classmethod
    def _collect_pred_true_hit_masks(
        cls,
        layer_preds: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pred_masks: list[torch.Tensor] = []
        true_masks: list[torch.Tensor] = []

        for pred_mask, true_mask in cls._pred_true_hit_mask_map(layer_preds, targets).values():
            pred_masks.append(pred_mask)
            true_masks.append(true_mask)

        if not pred_masks:
            return None, None
        if len(pred_masks) == 1:
            return pred_masks[0], true_masks[0]
        return torch.cat(pred_masks, dim=-1), torch.cat(true_masks, dim=-1)

    def _log_avg_num_valid_hits(
        self,
        stage: str,
        metric_prefix: str,
        mask: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> None:
        num_valid = mask.sum(-1).float()
        frac_valid = num_valid / mask.shape[-1]

        if query_mask is None:
            self.log(f"{stage}/{metric_prefix}avg_num_valid_hits", num_valid.mean(), sync_dist=True)
            self.log(f"{stage}/{metric_prefix}avg_frac_valid_hits", frac_valid.mean(), sync_dist=True)
            return

        valid_num_valid = num_valid[query_mask]
        valid_frac_valid = frac_valid[query_mask]
        if valid_num_valid.numel() > 0:
            self.log(f"{stage}/{metric_prefix}avg_num_valid_hits", valid_num_valid.mean(), sync_dist=True)
        if valid_frac_valid.numel() > 0:
            self.log(f"{stage}/{metric_prefix}avg_frac_valid_hits", valid_frac_valid.mean(), sync_dist=True)

    def _log_hit_mask_performance(
        self,
        pred_hit_masks: torch.Tensor,
        true_hit_masks: torch.Tensor,
        pred_valid: torch.Tensor,
        true_valid: torch.Tensor,
        stage: str,
        metric_prefix: str = "",
    ) -> None:
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1).float()
        hit_p = pred_hit_masks.sum(-1).float()
        hit_t = true_hit_masks.sum(-1).float()

        true_relevant = true_valid & (hit_t > 0)
        pred_relevant = pred_valid & (hit_p > 0)

        hit_eff = torch.where(hit_t > 0, hit_tp / hit_t, torch.zeros_like(hit_tp))
        hit_pur = torch.where(hit_p > 0, hit_tp / hit_p, torch.zeros_like(hit_tp))

        for wp in [0.5, 0.75, 1.0]:
            effs = (hit_eff >= wp) & pred_valid & true_relevant
            purs = (hit_pur >= wp) & true_valid & pred_relevant

            true_counts = true_relevant.float().sum(-1)
            pred_counts = pred_relevant.float().sum(-1)

            roi_effs = torch.where(true_counts > 0, effs.float().sum(-1) / true_counts, torch.nan)
            roi_purs = torch.where(pred_counts > 0, purs.float().sum(-1) / pred_counts, torch.nan)

            valid_roi_effs = roi_effs[~roi_effs.isnan()]
            valid_roi_purs = roi_purs[~roi_purs.isnan()]

            if valid_roi_effs.numel() > 0:
                self.log(f"{stage}/{metric_prefix}p{wp}_eff", valid_roi_effs.mean(), sync_dist=True)
            if valid_roi_purs.numel() > 0:
                self.log(f"{stage}/{metric_prefix}p{wp}_pur", valid_roi_purs.mean(), sync_dist=True)

        nh_per_true = true_hit_masks.sum(-1).float()[true_relevant]
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_relevant]

        if nh_per_true.numel() > 0:
            self.log(f"{stage}/{metric_prefix}nh_per_particle", nh_per_true.mean(), sync_dist=True)
        if nh_per_pred.numel() > 0:
            self.log(f"{stage}/{metric_prefix}nh_per_track", nh_per_pred.mean(), sync_dist=True)

        num_hits_total = float(pred_hit_masks.shape[-1])
        num_hits_valid = true_hit_masks.sum().float().item()
        num_hits_noise = num_hits_total - num_hits_valid
        self.log(f"{stage}/{metric_prefix}num_hits", num_hits_total, sync_dist=True)
        self.log(f"{stage}/{metric_prefix}num_hits_valid", num_hits_valid, sync_dist=True)
        self.log(f"{stage}/{metric_prefix}num_hits_noise", num_hits_noise, sync_dist=True)

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
            mask = self._collect_pred_hit_masks(layer_preds)
            if mask is not None:
                self._log_avg_num_valid_hits(stage, f"{layer_name}_", mask, query_mask)

            for constituent, constituent_mask in self._track_hit_mask_pred_map(layer_preds).items():
                self._log_avg_num_valid_hits(stage, f"{layer_name}_{constituent}_", constituent_mask.bool(), query_mask)

        # Just log predictions from the final layer
        preds = preds["final"]

        # First log metrics that depend on outputs from multiple tasks
        # TODO: Make the task names configurable or match task names automatically
        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        if query_mask is not None:
            pred_valid = pred_valid & query_mask

        # Set the masks of any track slots that are not used as null
        pred_mask, true_mask = self._collect_pred_true_hit_masks(preds, targets)
        if pred_mask is None or true_mask is None:
            return
        pred_hit_masks = pred_mask & pred_valid.unsqueeze(-1)
        true_hit_masks = true_mask & true_valid.unsqueeze(-1)
        self._log_hit_mask_performance(pred_hit_masks, true_hit_masks, pred_valid, true_valid, stage)

        for constituent, (constituent_pred_mask, constituent_true_mask) in self._pred_true_hit_mask_map(preds, targets).items():
            constituent_pred_hit_masks = constituent_pred_mask & pred_valid.unsqueeze(-1)
            constituent_true_hit_masks = constituent_true_mask & true_valid.unsqueeze(-1)
            self._log_hit_mask_performance(
                constituent_pred_hit_masks,
                constituent_true_hit_masks,
                pred_valid,
                true_valid,
                stage,
                metric_prefix=f"{constituent}_",
            )

        pred_num = pred_valid.sum(-1)

        self.log(f"{stage}/num_tracks", torch.mean(pred_num.float()), sync_dist=True)
        true_num = true_valid.sum(-1)
        self.log(f"{stage}/num_particles", torch.mean(true_num.float()), sync_dist=True)

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
