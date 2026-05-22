from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor

from hepattn.utils.tensor_utils import tensor_to_numpy


def _numpy_sample(value, idx):
    return tensor_to_numpy(value[idx])


def _lookup_nested(mapping, *keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _paper_track_iou_values(final_preds, idx):
    track_iou = _lookup_nested(final_preds, "track_iou", "query_iou")
    if track_iou is None:
        return None
    return _numpy_sample(track_iou, idx).astype(np.float32)


def _paper_named_track_iou_values(final_preds, idx, task_name: str):
    track_iou = _lookup_nested(final_preds, task_name, "query_iou")
    if track_iou is None:
        return None
    return _numpy_sample(track_iou, idx).astype(np.float32)


def _paper_track_valid_prob_values(final_preds, idx, task_name: str = "track_valid"):
    valid_prob = _lookup_nested(final_preds, task_name, "track_valid_prob")
    if valid_prob is None:
        valid_mask = _lookup_nested(final_preds, task_name, "track_valid")
        if valid_mask is None:
            return None
        return _numpy_sample(valid_mask, idx).astype(np.float32)
    return _numpy_sample(valid_prob, idx).astype(np.float32)


def _paper_class_scores(final_preds, idx):
    valid_prob_np = _paper_track_valid_prob_values(final_preds, idx)
    if valid_prob_np is None:
        return None
    return np.stack([valid_prob_np, 1.0 - valid_prob_np], axis=-1)


def _paper_mask_values(final_outputs, final_preds, targets, idx):
    mask_logits = _lookup_nested(final_outputs, "track_hit_valid", "track_key_logit")
    if mask_logits is None:
        mask_logits = _lookup_nested(final_outputs, "track_hit_valid", "track_hit_logit")
    if mask_logits is not None:
        return _numpy_sample(mask_logits, idx).astype(np.float32)

    mask_values = _lookup_nested(final_preds, "track_hit_valid", "track_key_valid")
    if mask_values is None:
        mask_values = _lookup_nested(final_preds, "track_hit_valid", "track_hit_valid")
    if mask_values is not None:
        return _numpy_sample(mask_values, idx).astype(np.float32)

    # Fallback for separate pixel+strip decoders: merge onto the paper hit axis
    # using the sorted key-hit provenance masks when available.
    pixel_logit = _lookup_nested(final_outputs, "track_pixel_assignment", "track_pixel_logit")
    strip_logit = _lookup_nested(final_outputs, "track_strip_assignment", "track_strip_logit")
    if pixel_logit is not None and strip_logit is not None:
        pixel_logit_np = _numpy_sample(pixel_logit, idx).astype(np.float32)
        strip_logit_np = _numpy_sample(strip_logit, idx).astype(np.float32)

        paper_hit_is_pixel = targets.get("paper_hit_is_pixel")
        paper_hit_is_strip = targets.get("paper_hit_is_strip")
        if paper_hit_is_pixel is not None and paper_hit_is_strip is not None:
            is_pixel = _numpy_sample(paper_hit_is_pixel, idx).astype(bool)
            is_strip = _numpy_sample(paper_hit_is_strip, idx).astype(bool)
            if pixel_logit_np.shape[1] != int(is_pixel.sum()):
                raise ValueError(
                    "track_pixel_assignment width does not match the number of paper pixel hits."
                )
            if strip_logit_np.shape[1] != int(is_strip.sum()):
                raise ValueError(
                    "track_strip_assignment width does not match the number of paper strip hits."
                )

            merged = np.full(
                (pixel_logit_np.shape[0], is_pixel.shape[0]),
                fill_value=np.finfo(np.float32).min,
                dtype=np.float32,
            )
            merged[:, is_pixel] = pixel_logit_np
            merged[:, is_strip] = strip_logit_np
            return merged

        return _numpy_sample(torch.cat([pixel_logit, strip_logit], dim=-1), idx).astype(np.float32)
    if pixel_logit is not None:
        return _numpy_sample(pixel_logit, idx).astype(np.float32)
    if strip_logit is not None:
        return _numpy_sample(strip_logit, idx).astype(np.float32)

    return None


def _paper_regression_values(final_preds, idx, num_queries):
    regression = {}
    track_regr = _lookup_nested(final_preds, "track_regr")
    if isinstance(track_regr, dict):
        for key, value in track_regr.items():
            old_key = key.removeprefix("track_")
            regression[old_key] = _numpy_sample(value, idx).astype(np.float32)

    has_cartesian = all(key in regression for key in ("px", "py", "pz"))
    if not has_cartesian and all(key in regression for key in ("pt", "eta", "phi")):
        pt = regression["pt"]
        eta = regression["eta"]
        phi = regression["phi"]
        regression["px"] = (pt * np.cos(phi)).astype(np.float32)
        regression["py"] = (pt * np.sin(phi)).astype(np.float32)
        regression["pz"] = (pt * np.sinh(eta)).astype(np.float32)
        has_cartesian = True

    if not has_cartesian:
        missing = np.full((num_queries,), np.nan, dtype=np.float32)
        regression["px"] = missing
        regression["py"] = missing.copy()
        regression["pz"] = missing.copy()
        return regression, True

    return regression, False


class PredictionWriter(Callback):
    def __init__(
        self,
        write_inputs: bool,
        write_outputs: bool,
        write_preds: bool,
        write_targets: bool,
        write_losses: bool,
        write_layers: list[str] | None = None,
        write_paper_compatible_test: bool = False,
        paper_track_valid_threshold: float = 0.5,
        paper_iou_threshold: float = 0.0,
    ):
        if write_layers is None:
            write_layers = ["final"]
        super().__init__()

        self.write_inputs = write_inputs
        self.write_outputs = write_outputs
        self.write_preds = write_preds
        self.write_targets = write_targets
        self.write_losses = write_losses
        self.write_layers = write_layers
        self.write_paper_compatible_test = write_paper_compatible_test
        self.paper_track_valid_threshold = float(paper_track_valid_threshold)
        self.paper_iou_threshold = float(paper_iou_threshold)

        self.file = None
        self.paper_file = None
        self.num_queries: int | None = None
        self.input_sort_field: str | None = None
        self.paper_event_index = 0
        self.paper_missing_regression_warned = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)

        self.trainer = trainer
        self.dataset = trainer.datamodule.test_dataloader().dataset

        self.num_queries = self._resolve_num_queries(pl_module)
        sorter = getattr(getattr(pl_module, "model", None), "sorter", None)
        self.input_sort_field = getattr(sorter, "input_sort_field", None)

        # Open the handle for writing to the file
        self.file = h5py.File(self.output_path, "w")
        if self.input_sort_field is not None:
            self.file.attrs["input_sort_field"] = self.input_sort_field
        if self.write_paper_compatible_test:
            self.paper_file = h5py.File(self.paper_output_path, "w")

    def _resolve_num_queries(self, pl_module: LightningModule) -> int | None:
        # Some models (e.g. HitFilter) have no decoder.
        decoder = getattr(getattr(pl_module, "model", None), "decoder", None)
        if decoder is None:
            return None
        num_queries = getattr(decoder, "_num_queries", None)
        return int(num_queries) if num_queries is not None else None

    @property
    def output_path(self) -> Path:
        # The output dataset will be saved in the same directory as the checkpoint
        split = Path(self.dataset.dirpath).name
        return Path(self.trainer.ckpt_dir / f"{self.trainer.ckpt_name}_{split}_eval.h5")

    @property
    def paper_output_path(self) -> Path:
        return Path(self.trainer.ckpt_dir / f"{self.trainer.ckpt_name}__test.h5")

    def on_test_batch_end(self, trainer, pl_module, test_step_outputs, batch, batch_idx):
        inputs, targets = batch
        sorted_targets = None

        if len(test_step_outputs) == 4:
            outputs, preds, losses, sorted_targets = test_step_outputs
        else:
            outputs, preds, losses = test_step_outputs

        if "targets_model_aligned" not in self.file.attrs:
            self.file.attrs["targets_model_aligned"] = bool(sorted_targets is not None)

        targets_to_write = sorted_targets if sorted_targets is not None else targets

        # handle batched case
        if "sample_id" in targets_to_write:
            # Get all of the sample IDs in the batch, this is what will be used to retrieve the samples
            sample_ids = targets_to_write["sample_id"]

            # Iterate through all of the samples in the batch
            for idx, sample_id in enumerate(sample_ids):
                self.write_sample(sample_id, inputs, targets_to_write, outputs, preds, losses, idx)
                if self.paper_file is not None:
                    self.write_paper_sample(sample_id, targets_to_write, outputs, preds, idx)

        # handle unbatched case
        else:
            self.write_sample(batch_idx, inputs, targets_to_write, outputs, preds, losses, 0)
            if self.paper_file is not None:
                self.write_paper_sample(batch_idx, targets_to_write, outputs, preds, 0)

    def write_sample(self, sample_id, inputs, targets, outputs, preds, losses, idx):
        """Write a single sample to the output file."""
        # create a group for thie sample_id
        if isinstance(sample_id, Tensor):
            sample_id = sample_id.item()
        sample_group = self.file.create_group(str(sample_id))

        # Write inputs and targets
        if self.write_inputs:
            self.write_items(sample_group, "inputs", inputs, idx)

        if self.write_targets:
            self.write_items(sample_group, "targets", targets, idx)

        # Items produced by model have layer/task structure
        if self.write_outputs:
            self.write_layer_task_items(sample_group, "outputs", outputs, idx)

        if self.write_preds:
            self.write_layer_task_items(sample_group, "preds", preds, idx)

        if self.write_losses:
            self.write_layer_task_items(sample_group, "losses", losses, idx)

    def write_paper_sample(self, sample_id, targets, outputs, preds, idx):
        if isinstance(sample_id, Tensor):
            sample_id = sample_id.item()

        sample_group = self.paper_file.create_group(f"event_{self.paper_event_index}")
        sample_group.attrs["sample_id"] = sample_id
        self.paper_event_index += 1

        required_target_keys = (
            "paper_truth_particle_id",
            "paper_truth_hit_id",
            "paper_truth_weight",
            "paper_hit_particle_id",
            "paper_hit_id",
            "paper_hit_weight",
            "paper_all_particle_id",
            "paper_all_particle_pt",
            "paper_all_particle_eta",
            "paper_all_particle_phi",
            "paper_all_particle_vz",
        )
        missing = [key for key in required_target_keys if key not in targets]
        if missing:
            msg = (
                "Paper-compatible TrackML output requested, but test targets are missing "
                f"{missing}. Set `data.paper_compatible_test_output=true` for the test run."
            )
            raise ValueError(msg)

        final_preds = preds.get("final", {})
        final_outputs = outputs.get("final", {})

        class_scores = _paper_class_scores(final_preds, idx)
        if class_scores is None:
            raise ValueError("Paper-compatible output requires `preds/final/track_valid` or `track_valid_prob`.")
        valid_prob = _paper_track_valid_prob_values(final_preds, idx)
        valid_prob_pixel = _paper_track_valid_prob_values(final_preds, idx, task_name="track_pixel_valid")
        valid_prob_strip = _paper_track_valid_prob_values(final_preds, idx, task_name="track_strip_valid")
        pred_iou = _paper_track_iou_values(final_preds, idx)
        pred_iou_pixel = _paper_named_track_iou_values(final_preds, idx, task_name="track_pixel_iou")
        pred_iou_strip = _paper_named_track_iou_values(final_preds, idx, task_name="track_strip_iou")

        masks = _paper_mask_values(final_outputs, final_preds, targets, idx)
        if masks is None:
            raise ValueError("Paper-compatible output requires merged `track_hit_valid` predictions.")

        regression, used_placeholder_regression = _paper_regression_values(final_preds, idx, num_queries=masks.shape[0])
        if used_placeholder_regression and not self.paper_missing_regression_warned:
            print(
                "PredictionWriter: no track regression outputs were found; paper-compatible "
                "file is writing NaN `px/py/pz` placeholders. Old eval fake-rate "
                "or efficiency numbers that depend on predicted track kinematics will not "
                "be directly comparable for this checkpoint."
            )
            self.paper_missing_regression_warned = True

        truth_group = sample_group.create_group("truth")
        hits_group = sample_group.create_group("hits")
        parts_group = sample_group.create_group("parts")
        prefilter_parts_group = None
        if all(key in targets for key in ("prefilter_all_particle_id", "prefilter_all_particle_pt", "prefilter_all_particle_eta", "prefilter_all_particle_phi")):
            prefilter_parts_group = sample_group.create_group("prefilter_parts")
        preds_group = sample_group.create_group("preds")
        regression_group = preds_group.create_group("regression")

        self.create_dataset(truth_group, "particle_id", _numpy_sample(targets["paper_truth_particle_id"], idx))
        self.create_dataset(truth_group, "hit_id", _numpy_sample(targets["paper_truth_hit_id"], idx))
        self.create_dataset(truth_group, "weight", _numpy_sample(targets["paper_truth_weight"], idx))

        self.create_dataset(hits_group, "pids", _numpy_sample(targets["paper_hit_particle_id"], idx))
        self.create_dataset(hits_group, "hids", _numpy_sample(targets["paper_hit_id"], idx))
        self.create_dataset(hits_group, "weight", _numpy_sample(targets["paper_hit_weight"], idx))
        if "paper_hit_is_pixel" in targets:
            self.create_dataset(hits_group, "is_pixel", _numpy_sample(targets["paper_hit_is_pixel"], idx))
        if "paper_hit_is_strip" in targets:
            self.create_dataset(hits_group, "is_strip", _numpy_sample(targets["paper_hit_is_strip"], idx))

        self.create_dataset(parts_group, "pids", _numpy_sample(targets["paper_all_particle_id"], idx))
        self.create_dataset(parts_group, "pts", _numpy_sample(targets["paper_all_particle_pt"], idx))
        self.create_dataset(parts_group, "etas", _numpy_sample(targets["paper_all_particle_eta"], idx))
        self.create_dataset(parts_group, "phis", _numpy_sample(targets["paper_all_particle_phi"], idx))
        self.create_dataset(parts_group, "vzs", _numpy_sample(targets["paper_all_particle_vz"], idx))
        if "paper_all_particle_num_hits" in targets:
            self.create_dataset(parts_group, "nhits", _numpy_sample(targets["paper_all_particle_num_hits"], idx))
        if "paper_all_particle_num_pixel_hits" in targets:
            self.create_dataset(parts_group, "n_pixel_hits", _numpy_sample(targets["paper_all_particle_num_pixel_hits"], idx))
        if "paper_all_particle_num_strip_hits" in targets:
            self.create_dataset(parts_group, "n_strip_hits", _numpy_sample(targets["paper_all_particle_num_strip_hits"], idx))
        if prefilter_parts_group is not None:
            self.create_dataset(prefilter_parts_group, "pids", _numpy_sample(targets["prefilter_all_particle_id"], idx))
            self.create_dataset(prefilter_parts_group, "pts", _numpy_sample(targets["prefilter_all_particle_pt"], idx))
            self.create_dataset(prefilter_parts_group, "etas", _numpy_sample(targets["prefilter_all_particle_eta"], idx))
            self.create_dataset(prefilter_parts_group, "phis", _numpy_sample(targets["prefilter_all_particle_phi"], idx))
            if "prefilter_all_particle_vz" in targets:
                self.create_dataset(prefilter_parts_group, "vzs", _numpy_sample(targets["prefilter_all_particle_vz"], idx))
            if "prefilter_all_particle_num_hits" in targets:
                self.create_dataset(prefilter_parts_group, "nhits", _numpy_sample(targets["prefilter_all_particle_num_hits"], idx))
            if "prefilter_all_particle_num_pixel_hits" in targets:
                self.create_dataset(
                    prefilter_parts_group,
                    "n_pixel_hits",
                    _numpy_sample(targets["prefilter_all_particle_num_pixel_hits"], idx),
                )
            if "prefilter_all_particle_num_strip_hits" in targets:
                self.create_dataset(
                    prefilter_parts_group,
                    "n_strip_hits",
                    _numpy_sample(targets["prefilter_all_particle_num_strip_hits"], idx),
                )

        self.create_dataset(preds_group, "class_preds", class_scores)
        if valid_prob is not None:
            self.create_dataset(preds_group, "track_valid_prob", valid_prob)
        if valid_prob_pixel is not None:
            self.create_dataset(preds_group, "track_valid_prob_pixel", valid_prob_pixel)
        if valid_prob_strip is not None:
            self.create_dataset(preds_group, "track_valid_prob_strip", valid_prob_strip)
        self.create_dataset(preds_group, "masks", masks)
        if pred_iou is not None:
            self.create_dataset(preds_group, "query_iou", pred_iou)
        if pred_iou_pixel is not None:
            self.create_dataset(preds_group, "query_iou_pixel", pred_iou_pixel)
        if pred_iou_strip is not None:
            self.create_dataset(preds_group, "query_iou_strip", pred_iou_strip)
        for name, value in regression.items():
            self.create_dataset(regression_group, name, value)

    def write_items(self, sample_group, item_name, items, idx):
        # This will write out a dict of items that has the structure
        # sample/item/value, e.g.
        # sample_id/inputs/pixel_x
        items_group = sample_group.create_group(item_name)
        for name, value in items.items():
            self.create_dataset(items_group, name, value[idx][None, ...])

    def write_layer_task_items(self, sample_group, item_name, items, idx):
        items_group = sample_group.create_group(item_name)
        # This will write out a dict of items that has the structure
        # sample/item/layer/task/value, e.g.
        # sample_id/preds/final/track_regression/track_phi
        for layer_name, layer_items in items.items():
            # Only write items fow the specified layers
            if layer_name not in self.write_layers:
                continue
            layer_group = items_group.create_group(layer_name)
            for task_name, task_items in layer_items.items():
                task_group = layer_group.create_group(task_name)
                for name, value in task_items.items():
                    self.create_dataset(task_group, name, value[idx][None, ...])

    def create_dataset(self, group, name, value):
        # Shouldn't need to detach as we are testing
        if isinstance(value, np.ndarray):
            value_np = value
        else:
            value_np = tensor_to_numpy(value)

        # Write the data to the file
        dataset_kwargs = {}
        if np.ndim(value_np) != 0:
            dataset_kwargs["compression"] = "lzf"
        group.create_dataset(name, data=value_np, **dataset_kwargs)

    def teardown(self, trainer, module, stage):
        # Close the file handle now we are done
        if stage == "test":
            if self.file is not None:
                self.file.close()
            if self.paper_file is not None:
                self.paper_file.close()
            print("-" * 80)
            print("Created output file", self.output_path)
            if self.write_paper_compatible_test:
                print("Created paper-compatible output file", self.paper_output_path)
            print("-" * 80)
