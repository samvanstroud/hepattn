from lightning import Callback, LightningModule, Trainer


class GradientLoggerCallback(Callback):
    def __init__(self, log_every_n_steps=50):
        """Callback to log model gradients during training.

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

    def on_after_backward(self, trainer, pl_module):
        """Called after the backward pass in training.
        Logs the gradients of the model's parameters.
        """
        # Check if logging should happen at this step
        if trainer.global_step % self.log_every_n_steps == 0:
            total_grad_magnitude = 0.0
            total_params = 0
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_magnitude = param.grad.norm().item()
                    total_grad_magnitude += grad_magnitude
                    total_params += param.grad.numel()
                    # Log gradient statistics
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    self.log(
                        {f"grad/{name}_mean": grad_mean, f"grad/{name}_std": grad_std}, "gradients"
                    )

                else:
                    self.log({f"grad/{name}_mean": None, f"grad/{name}_std": None}, "gradients")

            # Log total gradient magnitude or average magnitude
            if total_params > 0:
                avg_grad_magnitude = total_grad_magnitude / total_params
                self.log(
                    {
                        "total_magnitude": total_grad_magnitude,
                        "average_magnitude": avg_grad_magnitude,
                    },
                    "gradient",
                )
