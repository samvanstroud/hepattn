from copy import deepcopy

import torch
from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch.callbacks import ThroughputMonitor


class MyThroughputMonitor(ThroughputMonitor):
    def __init__(self):
        super().__init__(batch_size_fn=lambda x: 1)  # noqa: ARG005

    def setup(self, trainer, pl_module, stage):
        super().setup(trainer, pl_module, stage)
        with torch.device("meta"):
            model = deepcopy(pl_module).to(device="meta")

            def dummy_forward():
                batch = {"hit": torch.randn(1, int(1e4), pl_module.init.input_size, device="meta")}
                batch["phi"] = torch.randn(1, int(1e4), device="meta")
                batch["theta"] = torch.randn(1, int(1e4), device="meta")
                batch["r"] = torch.randn(1, int(1e4), device="meta")
                return model(batch)["hit_pred"]

            pl_module.flops_per_batch = measure_flops(model, dummy_forward, loss_fn=torch.Tensor.sum)
