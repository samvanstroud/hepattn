from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot
from ftag.hdf5 import H5Writer
from lightning import Callback, LightningModule, Trainer
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from hepattn.utils.array_utils import join_structured_arrays, maybe_pad


def load_convert_h5(filepath):
    with h5py.File(filepath, "r") as f:
        # if "Bin" in filepath:
        #     pflow_class_probs_raw = np.stack([f["object_class"][f"class_probs_{i}"][:] for i in range(2)], axis=-1)
        #     pflow_class_probs = pflow_class_probs_raw
        #     pflow_class = np.argmax(pflow_class_probs, axis=-1)
        # else:
        #     class_probs_vars = [f"class_probs_{i}" for i in range(6)]
        #     pflow_class_probs = s2u(f["object_class"].fields(class_probs_vars)[:])
        #     pflow_class = np.argmax(pflow_class_probs, axis=-1)
        pflow_class = f["object_class"]["pflow_class"][:]

        pflow_vars = [f"pred_{el}" for el in ["e", "pt", "eta", "sinphi", "cosphi"]]
        proxy_vars = [f"proxy_{el}" for el in ["e", "pt", "eta", "sinphi", "cosphi"]]
        pflow_data = s2u(f["regression"].fields(pflow_vars)[:])
        proxy_data = s2u(f["regression"].fields(proxy_vars)[:])

        pflow_ptetaphi = np.stack(
            [
                pflow_data[..., 1],
                pflow_data[..., 2],
                np.arctan2(pflow_data[..., 3], pflow_data[..., 4]),
            ],
            axis=-1,
        )
        proxy_ptetaphi = np.stack(
            [
                proxy_data[..., 1],
                proxy_data[..., 2],
                np.arctan2(proxy_data[..., 3], proxy_data[..., 4]),
            ],
            axis=-1,
        )

        pflow_indicator = pflow_class < 1 if "Bin" in filepath else (pflow_class < 5) & (np.abs(pflow_ptetaphi[..., 1]) < 4)

        neutral_mask = (pflow_class < 5) & (pflow_class > 2)
        pflow_ptetaphi[neutral_mask][..., 0] = pflow_data[neutral_mask][..., 0] / np.cosh(pflow_ptetaphi[neutral_mask][..., 1])

        event_number = f["events"]["event_number"][:]

        return (
            event_number,
            pflow_class,
            pflow_ptetaphi,
            proxy_ptetaphi,
            pflow_indicator,
        )


class PflowPredictionWriter(Callback):
    def __init__(self) -> None:
        super().__init__()

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        self.writer = None

        # basic properties
        self.trainer = trainer
        self.batch_size = trainer.datamodule.batch_size
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.test_suff = trainer.datamodule.test_suff
        self.num_events = len(self.ds)

        self.var_transform = self.ds.scaler.transforms

    @property
    def output_path(self) -> Path:
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        suffix = f"_{self.test_suff}" if self.test_suff else ""
        return Path(out_dir / f"{out_basename}__test{suffix}.h5")

    def _write_batch_outputs(self, batch_outputs, pad_masks, batch_idx):
        to_write = {}
        # blow = batch_idx * self.batch_size
        # bhigh = (batch_idx + 1) * self.batch_size
        for input_name, outputs in batch_outputs.items():
            this_outputs = []
            name = input_name
            inputs = None

            for preds in outputs.values():
                if inputs is not None:
                    this_outputs.append(maybe_pad(preds, inputs))
                else:
                    this_outputs.append(preds)

            # add mask if present
            if name in pad_masks:
                pad_mask = pad_masks[name].cpu()
                pad_mask = u2s(np.expand_dims(pad_mask, -1), dtype=np.dtype([("mask", "?")]))
                this_outputs.append(maybe_pad(pad_mask, inputs))
            to_write[name] = join_structured_arrays(this_outputs)

        # If the writer hasn't been created yet, create it now that we have the dtypes and shapes
        if self.writer is None:
            dtypes = {k: v.dtype for k, v in to_write.items()}
            shapes = {k: (self.num_events,) + v.shape[1:] for k, v in to_write.items()}
            self.writer = H5Writer(
                jets_name="events",
                dst=self.output_path,
                dtypes=dtypes,
                shapes=shapes,
                shuffle=False,
                precision="full",
            )
        self.writer.write(to_write)

    def on_test_batch_end(self, trainer, module, test_step_outputs, batch, batch_idx):
        _inputs, targets = batch
        outputs, preds, _losses = test_step_outputs
        outputs = outputs["final"]
        preds = preds["final"]
        to_write = {}
        # Event numbers
        to_write["events"] = {
            "event_number": u2s(
                targets["event_number"].cpu().numpy().astype(np.int64).reshape(-1, 1),
                dtype=np.dtype([("event_number", "i8")]),
            )
        }
        # object class
        to_write["object_class"] = {}
        to_write["object_class"]["targets"] = u2s(
            targets["particle_class"].cpu().unsqueeze(-1).numpy(),
            dtype=np.dtype([("object_class", "i8")]),
        )
        if "classification" in preds:
            to_write["object_class"]["preds"] = u2s(
                preds["classification"]["pflow_class"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("pflow_class", "i8")]),
            )

        # masks
        to_write["object_masks"] = {}
        to_write["object_masks"]["targets"] = u2s(
            targets["particle_node_valid"].cpu().unsqueeze(-1).numpy(),
            dtype=np.dtype([("truth_masks", "i8")]),
        )
        # to_write["object_masks"]["preds"] = u2s(
        #     preds['mask']["pflow_node_valid"].cpu().unsqueeze(-1).float().numpy(),
        #     dtype=np.dtype([("mask_logits", np.float32)]),
        # )
        to_write["object_masks"]["preds"] = u2s(
            outputs["mask"]["pflow_node_logit"].cpu().unsqueeze(-1).float().numpy(),
            dtype=np.dtype([("mask_logits", np.float32)]),
        )

        if "particle_incidence" in targets and "incidence" in preds:
            to_write["incidence"] = {}
            to_write["incidence"]["targets"] = u2s(
                targets["particle_incidence"].cpu().unsqueeze(-1).numpy(),
                dtype=np.dtype([("truth_incidence", np.float32)]),
            )
            to_write["incidence"]["preds"] = u2s(
                preds["incidence"]["pflow_incidence"].cpu().unsqueeze(-1).float().numpy(),
                dtype=np.dtype([("pred_incidence", np.float32)]),
            )

        # regression
        to_write["regression"] = {}
        for _, t in enumerate(["e", "pt", "eta", "sinphi", "cosphi"]):
            truth_data = targets[f"particle_{t}"].cpu().float().unsqueeze(-1)
            pred_data = preds["regression"][f"pflow_{t}"].cpu().float().unsqueeze(-1)
            proxy_data = None
            if f"pflow_proxy_{t}" in preds["regression"]:
                proxy_data = preds["regression"][f"pflow_proxy_{t}"].cpu().float().unsqueeze(-1)
            if t in self.var_transform:
                truth_data = self.var_transform[t].inverse_transform(truth_data)
                pred_data = self.var_transform[t].inverse_transform(pred_data)
                if proxy_data is not None:
                    proxy_data = self.var_transform[t].inverse_transform(proxy_data)

            to_write["regression"][f"truth_{t}"] = u2s(
                truth_data.numpy(),
                dtype=np.dtype([(f"truth_{t}", np.float32)]),
            )
            to_write["regression"][f"pred_{t}"] = u2s(
                pred_data.numpy(),
                dtype=np.dtype([(f"pred_{t}", np.float32)]),
            )
            if proxy_data is not None:
                to_write["regression"][f"proxy_{t}"] = u2s(
                    proxy_data.numpy(),
                    dtype=np.dtype([(f"proxy_{t}", np.float32)]),
                )
        # for key, value in to_write.items():
        #     if isinstance(value, dict):
        #         for subkey, subvalue in value.items():
        #             print(f"{key}/{subkey}: {subvalue.shape}")
        #     else:
        #         print(f"{key}: {value.shape}")
        self._write_batch_outputs(to_write, {}, batch_idx)

    def on_test_end(self, trainer, module):
        self.output_path: Path
        if self.writer is not None:
            print(f"Wrote predictions to {self.output_path}")
            self.writer.close()
        print("Loading predictions...")
        event_number, pflow_class, pflow_ptetaphi, proxy_ptetaphi, pflow_indicator = load_convert_h5(self.output_path.as_posix())
        root_path = self.output_path.with_suffix(".root").as_posix()
        print("Writing to ROOT file")
        with uproot.recreate(root_path) as f:
            f["event_tree"] = {
                "mpflow": {
                    "pt": ak.Array(pflow_ptetaphi[..., 0]),
                    "eta": ak.Array(pflow_ptetaphi[..., 1]),
                    "phi": ak.Array(pflow_ptetaphi[..., 2]),
                    "class": ak.Array(pflow_class),
                },
                "proxy": {
                    "pt": ak.Array(proxy_ptetaphi[..., 0]),
                    "eta": ak.Array(proxy_ptetaphi[..., 1]),
                    "phi": ak.Array(proxy_ptetaphi[..., 2]),
                },
                "pred_ind": ak.Array(pflow_indicator),
                "event_number": ak.Array(event_number)[: len(pflow_indicator)],
            }
        print(f"Wrote ROOT file to {root_path}")
