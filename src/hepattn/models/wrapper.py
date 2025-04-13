import torch
from lightning import LightningModule
from torch import nn
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        self.mtl = mtl

        # If we are doing multi-task-learning, optimisation step must be done manually
        if mtl:
            self.automatic_optimization = False

    def forward(self, inputs):
        return self.model(inputs)

    def predict(self, outputs):
        return self.model.predict(outputs)

    def log_losses(self, losses, stage):
        total_loss = 0

        # Log the losses from each task from each layer
        for layer_name, layer_losses in losses.items():
            for task_name, task_losses in layer_losses.items():
                task_losses = layer_losses[task_name]
                for loss_name, loss_value in task_losses.items():
                    self.log(f"{stage}/{layer_name}_{task_name}_{loss_name}", loss_value)
                total_loss += loss_value

        # Log the total loss
        self.log(f"{stage}/loss", total_loss)

    def log_task_metrics(self, preds, targets, stage):
        # Log any task specific metrics
        for task in self.model.tasks:
            # Check that the task actually has some metrics to log
            if not hasattr(task, "metrics"):
                continue

            # Just log the predictions from the final layer for now
            task_metrics = task.metrics(preds["final"][task.name], targets)

            # If the task returned a non-empty metrics dict, log it
            if task_metrics:
                self.log_dict({f"{stage}/final_{task.name}_{k}": v for k, v in task_metrics.items()})

    def log_metrics(self, preds, targets, stage):
        # First log any task metrics
        self.log_task_metrics(preds, targets, stage)

        # If the superclass has implemented some compound metrics that
        # depend on the outputs of multiple tasks, log those too
        # TODO: Probably a better way of doing this will callbacks or something
        if hasattr(self, "log_compound_metrics"):
            self.log_compound_metrics(preds, targets, stage)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # Get the model outputs
        outputs = self.model(inputs)

        # Get the losses from all of the tasks
        losses = self.model.loss(outputs, targets)

        self.log_losses(losses, "train")

        if batch_idx % self.trainer.log_every_n_steps == 0:
            # If we want to log metrics we go ahead and compute the actual predictions
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")

        # Here we choose which strategy to use to weight/handle the different layers and tasks
        # Use Jacobian Descent for Multi Task Learning https://arxiv.org/abs/2406.16232
        if self.mtl:
            opt = self.optimizers()
            opt.zero_grad()

            for layer_name, layer_losses in losses.items():
                # Get a list of the features that are used by all of the tasks
                layer_features = []
                for task in self.model.tasks:
                    layer_features.extend(outputs[layer_name][input_feature] for input_feature in task.input_features)

                # Remove any duplicate features that are used by multiple tasks
                layer_features = list(set(layer_features))

                # Perform the backward pass for this layer
                layer_losses = [sum(losses[layer_name][task.name]) for task in self.model.tasks]
                mtl_backward(losses=layer_losses, features=layer_features, aggregator=UPGrad())

            opt.step()

        # Default multi-task strategy is to just add up all the losses
        else:
            losses_flat = {}

            # Unpack the nested losses into a single flat dictionary
            for layer_name, layer_losses in losses.items():
                for task_name, task_losses in layer_losses.items():
                    for loss_name, loss in task_losses.items():
                        losses_flat[f"{layer_name}_{task_name}_{loss_name}"] = loss

            total_loss = sum(losses_flat.values())

            return total_loss
        return None

    def validation_step(self, batch):
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Note we have to compute the losses for the validation step also so
        # the metrics are correct as the loss calculation is where we do the
        # permutation of the predictions to match the targets
        losses = self.model.loss(outputs, targets)
        self.log_losses(losses, "val")

        # Use the outputs to produce actual usable predictions
        preds = self.model.predict(outputs)

        # Evaluate the predictions on the
        self.log_metrics(preds, targets, "val")

        return preds

    def test_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Like in validation step, we have to calculate the losses first
        losses = self.model.loss(outputs, targets)
        preds = self.model.predict(outputs)

        return outputs, preds, losses

    def on_train_start(self):
        # Manually overwride the learning rate in case we are starting
        # from a checkpoint that had a LRS and now we want a flat LR
        if self.lrs_config["skip_scheduler"]:
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lrs_config["initial"]

    def configure_optimizers(self):
        # Pick which optimizer we want to use
        if self.optimizer.lower() == "adamw":
            from torch.optim import AdamW  # noqa: PLC0415

            opt = AdamW(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])
        # https://arxiv.org/abs/2302.06675
        elif self.optimizer.lower() == "lion":
            from lion_pytorch import Lion  # noqa: PLC0415

            opt = Lion(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config["skip_scheduler"]:
            # Configure the learning rate scheduler
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches * 2,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            sch = {"scheduler": sch, "interval": "step"}
            return [opt], [sch]
        print("Skipping learning rate scheduler.")
        return opt
