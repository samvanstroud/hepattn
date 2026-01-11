from typing import Literal

import torch
from lightning import LightningModule
from lion_pytorch import Lion
from torch import Tensor, nn
from torch._functorch import config as functorch_config  # noqa: PLC2701
from torch.optim import AdamW
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: Literal["AdamW", "Lion"] = "AdamW",
        mtl: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        self.mtl = mtl

        if mtl:
            # Donated buffers can cause issues with graph retention needed for MTL
            functorch_config.donated_buffer = False
            # If we are doing multi-task-learning, optimisation step must be done manually
            self.automatic_optimization = False
            # MTL does not currently support intermediate losses
            assert all(task.has_intermediate_loss is False for task in self.model.tasks)

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model.predict(outputs)

    def aggregate_losses(self, losses: dict, stage: str | None = None) -> Tensor:
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)

        for layer_name, layer_losses in losses.items():
            layer_loss = 0
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    self.log(f"{stage}/{layer_name}_{task_name}_{loss_name}", loss_value, sync_dist=True)
                    total_loss += loss_value
                    layer_loss += loss_value

            # Log the total loss from the layer
            self.log(f"{stage}/{layer_name}_loss", layer_loss, sync_dist=True)

        # Log the total loss
        self.log(f"{stage}/loss", total_loss, sync_dist=True)
        return total_loss

    @staticmethod
    def _align_predictions_to_full_targets(
        preds: dict[str, dict[str, dict[str, Tensor]]],
        query_particle_idx: Tensor,
        num_full_particles: int,
    ) -> dict[str, dict[str, dict[str, Tensor]]]:
        """Align query-sized predictions to full particle dimension using query-to-particle mapping.

        Uses scatter_reduce with 'amax' to handle duplicate queries mapping to the same particle.
        This ensures that valid (non-zero) predictions are not overwritten by zeroed-out
        duplicate query values.

        Args:
            preds: Predictions dict with structure {layer: {task: {field: tensor}}}
            query_particle_idx: (B, N_queries) mapping from query index to particle index
            num_full_particles: Number of particles in full targets

        Returns:
            Aligned predictions dict with particle dimension expanded to full size
        """
        batch_size = query_particle_idx.shape[0]
        num_queries = query_particle_idx.shape[1]
        device = query_particle_idx.device
        idx = query_particle_idx.detach().long()

        def scatter_align(value: Tensor) -> Tensor:
            """Scatter query-sized tensor to full particle dimension using amax reduction."""
            original_dtype = value.dtype
            value = value.detach().float()

            # Build output shape: replace query dim with num_full_particles
            out_shape = list(value.shape)
            out_shape[1] = num_full_particles
            aligned = torch.zeros(out_shape, dtype=torch.float32, device=device)

            # Expand index to match value shape for scatter_reduce
            idx_expanded = idx.view(batch_size, num_queries, *([1] * (value.dim() - 2))).expand_as(value)

            aligned.scatter_reduce_(dim=1, index=idx_expanded, src=value, reduce="amax", include_self=True)

            # Convert back to original dtype
            if original_dtype == torch.bool:
                return aligned > 0.5
            return aligned.to(original_dtype)

        aligned_preds = {}
        for layer_name, layer_preds in preds.items():
            aligned_preds[layer_name] = {}
            for task_name, task_preds in layer_preds.items():
                aligned_preds[layer_name][task_name] = {}
                for field_name, field_value in task_preds.items():
                    # Only align tensors with query dimension
                    if isinstance(field_value, Tensor) and field_value.shape[1] == num_queries:
                        aligned_preds[layer_name][task_name][field_name] = scatter_align(field_value)
                    else:
                        aligned_preds[layer_name][task_name][field_name] = field_value

        return aligned_preds

    def log_task_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        # Log any task specific metrics
        for layer_name in preds:
            for task in self.model.tasks:
                if task.name not in preds[layer_name]:
                    continue

                task_metrics = task.metrics(preds[layer_name][task.name], targets)
                if task_metrics:
                    self.log_dict({f"{stage}/{layer_name}_{task.name}_{k}": v for k, v in task_metrics.items()}, sync_dist=True)

    def log_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        # Check if dynamic queries are active and we need to align predictions
        if "query_particle_idx" in targets and "particle_valid_full" in targets:
            # Align predictions to full particle dimension for metrics computation
            num_full_particles = targets["particle_valid_full"].shape[1]
            aligned_preds = self._align_predictions_to_full_targets(preds, targets["query_particle_idx"], num_full_particles)

            # Create aligned targets dict with full validity masks
            aligned_targets = targets.copy()
            aligned_targets["particle_valid"] = targets["particle_valid_full"]
            aligned_targets["particle_hit_valid"] = targets["particle_hit_valid_full"]

            # Log task metrics with aligned predictions and full targets
            self.log_task_metrics(aligned_preds, aligned_targets, stage)

            # Log custom metrics with aligned predictions and full targets
            if hasattr(self, "log_custom_metrics"):
                self.log_custom_metrics(aligned_preds, aligned_targets, stage)
        else:
            # No dynamic queries, use standard metrics computation
            self.log_task_metrics(preds, targets, stage)

            if hasattr(self, "log_custom_metrics"):
                self.log_custom_metrics(preds, targets, stage)

    def training_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]], batch_idx: int) -> dict[str, Tensor] | None:
        inputs, targets = batch

        # Get the model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        losses, targets = self.model.loss(outputs, targets)

        # Get the predictions from the model, avoid calling predict if possible
        if batch_idx % self.trainer.log_every_n_steps == 0:
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")

        if self.mtl:
            self.mlt_opt(losses, outputs)
            return None
        total_loss = self.aggregate_losses(losses, stage="train")

        return {"loss": total_loss}

    def validation_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> dict[str, Tensor]:
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Compute losses then aggregate and log them
        losses, targets = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")

        return {"loss": total_loss}

    def test_step(
        self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss to also run matching
        losses, targets = self.model.loss(outputs, targets)

        # Get the predictions from the model
        preds = self.model.predict(outputs)

        return outputs, preds, losses, targets

    def on_train_start(self) -> None:
        # Manually overwride the learning rate in case we are starting
        # from a checkpoint that had a LRS and now we want a flat LR
        if self.lrs_config.get("skip_scheduler"):
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lrs_config["initial"]

    def configure_optimizers(self):
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.opt_config['opt']}")

        opt = optimizer(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            # Configure the learning rate scheduler
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            sch = {"scheduler": sch, "interval": "step"}
            return [opt], [sch]

        print("Skipping learning rate scheduler.")
        return opt

    def mlt_opt(self, losses: dict[str, Tensor], outputs: dict[str, Tensor]) -> None:
        opt = self.optimizers()
        opt.zero_grad()

        # TODO: Make this not hard coded?
        feature_names = ["query_embed", "key_embed"]

        # Remove any duplicate features that are used by multiple tasks
        features = [outputs["final"][feature_name] for feature_name in feature_names]

        # TODO: Figure out if we can set retain_graph to false somehow, since it uses a lot of memory
        task_losses = [sum(losses["final"][task.name].values()) for task in self.model.tasks]
        mtl_backward(losses=task_losses, features=features, aggregator=UPGrad(), retain_graph=True)

        # Manually perform the optimizer step
        opt.step()
