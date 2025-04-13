import socket
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CometLogger


class Metadata(Callback):
    def __init__(self) -> None:
        super().__init__()

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:  # noqa: ARG002
        if trainer.is_global_zero:
            print("-" * 80)
            print(f"log dir: {trainer.log_dir!r}")
            print("-" * 80)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.trainer = trainer
        if not trainer.is_global_zero:
            return
        if not trainer.log_dir:
            return

        log_dir = Path(trainer.log_dir)
        self.save_metadata(log_dir, pl_module)
        if isinstance(self.trainer.logger, CometLogger):
            for file in log_dir.glob("*.yaml"):
                self.trainer.logger.experiment.log_asset(file)

    def save_metadata(self, log_dir: Path, pl_module: LightningModule) -> None:
        trainer = self.trainer
        logger = trainer.logger
        datamodule = trainer.datamodule

        meta = {
            "num_train": len(datamodule.train_dataloader().dataset),
            "num_val": len(datamodule.val_dataloader().dataset),
            "batch_size": datamodule.train_dataloader().batch_size,
            "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            "num_gpus": trainer.num_devices,
            "gpu_ids": trainer.device_ids,
            "num_workers": datamodule.train_dataloader().num_workers,
            "torch_version": str(torch.__version__),
            "lightning_version": str(L.__version__),
            "cuda_version": torch.version.cuda,
            "hostname": socket.gethostname(),
            "out_dir": logger.save_dir,
            "log_url": getattr(logger.experiment, "url", None),
        }
        if hasattr(trainer, "timestamp"):
            meta["timestamp"] = trainer.timestamp

        if logger := pl_module.logger:
            logger.log_hyperparams(meta)

        meta_path = log_dir / "metadata.yaml"
        with meta_path.open("w") as file:
            yaml.dump(meta, file, sort_keys=False)
