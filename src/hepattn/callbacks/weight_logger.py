from lightning import Callback, LightningModule, Trainer


class WeightLoggerCallback(Callback):
    def __init__(self, log_every_n_steps=50):
        """Callback to log model weights and biases during training.

        Args:
            log_every_n_steps (int): Frequency of logging gradients. Logs every `n` steps.
        """
        self.log_every_n_steps = log_every_n_steps

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        kwargs = {"sync_dist": len(trainer.device_ids) > 1}

        def log(metrics, stage):
            for t, loss_value in metrics.items():
                n = f"{stage}_{t}"
                module.log(n, loss_value, **kwargs)

        self.log = log

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: ARG002
        """Called after each training batch ends.
        Logs the mean and std of weights and biases in each layer.
        """
        if trainer.global_step % self.log_every_n_steps == 0:
            for name, param in pl_module.named_parameters():
                if "weight" in name or "bias" in name:
                    # Log weight statistics
                    weight_mean = param.data.mean().item()
                    weight_std = param.data.std().item()
                    self.log(
                        {
                            f"weights/{name}_mean": weight_mean,
                            f"weights/{name}_std": weight_std,
                        },
                        "weights",
                    )
