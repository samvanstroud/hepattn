from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer


class TestEvalWriter(Callback):
    def __init__(
        self,
        write_inputs: bool,
        write_outputs: bool,
        write_preds: bool,
        write_targets: bool,
        write_losses: bool,
        write_layers: list[str] | None = None,
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

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)

        self.trainer = trainer
        self.dataset = trainer.datamodule.test_dataloader().dataset

    @property
    def output_path(self) -> Path:
        # The output dataset will be saved in the same direcory as the checkpoint
        out_dir = Path(self.trainer.ckpt_path).parent
        out_basename = str(Path(self.trainer.ckpt_path).stem)
        split = self.dataset.dirpath.name
        return Path(out_dir / f"{out_basename}_{split}_eval.h5")

    def on_test_start(self, trainer: Trainer, module: LightningModule) -> None:  # noqa: ARG002
        # Open the handle for writing to the file
        self.file = h5py.File(self.output_path, "w")

    def on_test_batch_end(self, trainer, pl_module, test_step_outputs, batch, batch_idx):  # noqa: ARG002
        inputs, targets = batch
        outputs, preds, losses = test_step_outputs

        # TODO: Standardise this somehow for dataloders that have a
        # batched input / do not have event names
        event_name = self.dataset.event_names[batch_idx]
        sample_group = self.file.create_group(event_name)

        # Write inputs and targets
        if self.write_inputs:
            self.write_items(sample_group, "inputs", inputs)

        if self.write_targets:
            self.write_items(sample_group, "targets", targets)

        # Items produced by model have layer/task structure
        if self.write_outputs:
            self.write_layer_task_items(sample_group, "outputs", outputs)

        if self.write_preds:
            self.write_layer_task_items(sample_group, "preds", preds)

        if self.write_losses:
            self.write_layer_task_items(sample_group, "losses", losses)

    def write_items(self, sample_group, item_name, items):
        # This will write out a dict of items that has the structure
        # sample/item/layer/task/value, e.g.
        # event_name/inputs/pixel_x
        items_group = sample_group.create_group(item_name)
        for name, value in items.items():
            self.create_dataset(items_group, name, value)

    def write_layer_task_items(self, sample_group, item_name, items):
        items_group = sample_group.create_group(item_name)
        # This will write out a dict of items that has the structure
        # sample/item/layer/task/value, e.g.
        # event_name/preds/final/track_regression/track_phi
        for layer_name, layer_items in items.items():
            # Only write items fow the specified layers
            if layer_name not in self.write_layers:
                continue
            layer_group = items_group.create_group(layer_name)
            for task_name, task_items in layer_items.items():
                task_group = layer_group.create_group(task_name)
                for name, value in task_items.items():
                    self.create_dataset(task_group, name, value)

    def on_test_epoch_end(self, trainer, module):  # noqa: ARG002
        # Close the file handle now we are done
        self.file.close()
        print("Created output file", self.output_path)

    def create_dataset(self, group, name, value):
        # Shouldn't need to detach as we are testing
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()

        # Handle half precison values
        value_type = np.dtype(value.dtype)
        if value_type.kind == "f" and value_type.element_size() == 2:
            value = value.astype(np.float16)

        # Write the data to the file
        group.create_dataset(name, data=value, compression="lzf")
