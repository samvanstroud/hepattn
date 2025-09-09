import os
import socket
from pathlib import Path

import lightning
import torch
import yaml
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CometLogger


class SaveConfig(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.already_saved = False

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.already_saved or trainer.fast_dev_run:
            return

        log_dir = trainer.log_dir  # this broadcasts the directory
        assert log_dir is not None

        if trainer.is_global_zero:
            print("-" * 80)
            print(f"log dir: {log_dir!r}")
            print("-" * 80)

            if log_dir:
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                self.save_metadata(trainer, log_dir, pl_module)

            if isinstance(trainer.logger, CometLogger):
                for file in log_dir.glob("*.yaml"):
                    trainer.logger.experiment.log_asset(file)
                base_dir = Path(__file__).parents[3]
                for file in (base_dir / "src").glob("**/*.py"):
                    trainer.logger.experiment.log_code(file)

            self.already_saved = True

        self.already_saved = trainer.strategy.broadcast(self.already_saved, src=0)

    def save_metadata(self, trainer, log_dir: Path, pl_module: LightningModule) -> None:
        logger = trainer.logger
        datamodule = trainer.datamodule

        # Metadata that we want to log
        meta = {
            "num_train": len(datamodule.train_dataloader().dataset),
            "num_val": len(datamodule.val_dataloader().dataset),
            "batch_size": datamodule.train_dataloader().batch_size,
            "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            "num_gpus": trainer.num_devices,
            "gpu_ids": trainer.device_ids,
            "num_workers": datamodule.train_dataloader().num_workers,
            "torch_version": str(torch.__version__),
            "lightning_version": str(lightning.__version__),
            "cuda_version": torch.version.cuda,
            "hostname": socket.gethostname(),
            "out_dir": logger.save_dir if logger else log_dir,
            "log_url": getattr(logger.experiment, "url", None) if logger else None,
        }

        if hasattr(trainer, "timestamp"):
            meta["timestamp"] = trainer.timestamp

        # Log the SLURM info if its present
        if "SLURM_JOB_ID" in os.environ:
            meta["slurm_job_id"] = "SLURM_" + str(os.environ["SLURM_JOB_ID"])

        # Log the metadata to the logger if we are using one
        logger = pl_module.logger
        if logger:
            logger.log_hyperparams(meta)

        # Save the metadata locally to a YAML file
        meta_path = log_dir / "metadata.yaml"
        with meta_path.open("w") as file:
            yaml.dump(meta, file, sort_keys=False)
