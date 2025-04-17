from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer


class HitPredictionWriter(Callback):
    def __init__(self) -> None:
        super().__init__()

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:  # noqa: ARG002
        if stage != "test":
            return

        self.trainer = trainer

        # get test dataset
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.num_events = len(self.ds)

    @property
    def output_path(self) -> Path:
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        return Path(out_dir / f"{out_basename}__test.h5")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        self.file = h5py.File(self.output_path, "w")

    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx):  # noqa: ARG002
        preds = outputs
        targets = batch[1]

        # create a group for each event using the event id
        event_id = self.ds.get_event_id_from_batch_idx(batch_idx)
        g = self.file.create_group(event_id)

        self.create_dataset(g, preds["hit_pred"], "hit_pred")
        for i in targets:
            if isinstance(targets[i], dict):
                for j in targets[i]:
                    self.create_dataset(g, targets[i][j], f"targets/{i}/{j}")
            else:
                self.create_dataset(g, targets[i], "targets/" + i)

    def on_test_epoch_end(self, trainer, module):  # noqa: ARG002
        self.file.close()
        print("Created output file", self.output_path)
        print("-" * 100, "\n")

    def create_dataset(self, f, a, name, half_precision=True):
        a = a.squeeze()

        if isinstance(a, torch.Tensor):
            a = a.float().cpu().numpy()

        # convert down to float16
        if half_precision:
            t = np.dtype(a.dtype)
            if t.kind == "f" and t.itemsize == 2:
                a = a.astype(np.float16)

        # write
        f.create_dataset(name, data=a, compression="lzf")
