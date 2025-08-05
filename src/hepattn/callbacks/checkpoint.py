from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    def __init__(self, monitor: str = "val/loss", log_model: bool = True, **kwargs) -> None:
        filename = "epoch={epoch:03d}-" + monitor.replace("/", "_") + "={" + monitor + ":.5f}"
        super().__init__(save_top_k=-1, monitor=monitor, filename=filename, auto_insert_metric_name=False, **kwargs)
        self.log_model = log_model

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
        self.save_last = False
        self.name = pl_module.name
        if stage == "fit":
            if trainer.fast_dev_run:
                return
            log_dir = Path(trainer.log_dir)
            self.dirpath = str(log_dir / "ckpts")

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        metadata = {"epoch": trainer.current_epoch, "dirpath": self.dirpath}
        if self.log_model:
            trainer.logger.experiment.log_model(name=self.name, file_or_folder=filepath, metadata=metadata)
