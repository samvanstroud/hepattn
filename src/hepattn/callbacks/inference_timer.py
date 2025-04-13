from pathlib import Path

import numpy as np
import torch
from lightning import Callback

from hepattn.utils.cuda_timer import cuda_timer


class InferenceTimer(Callback):
    def __init__(self):
        super().__init__()
        self.times = []
        self.dims = []
        self.n_warm_start = 10

    def on_test_start(self, trainer, pl_module):  # noqa: ARG002
        self.old_forward = pl_module.model.forward

        def new_forward(*args, **kwargs):
            self.dims.append(sum(v.shape[1] for v in args[0].values()))
            with cuda_timer(self.times):
                return self.old_forward(*args, **kwargs)

        pl_module.model.forward = new_forward

    def on_test_end(self, trainer, pl_module):
        pl_module.forward = self.old_forward
        self.times = self.times[5:]  # ensure warm start
        self.times = torch.tensor(self.times)

        if len(self.times):
            self.mean_time = self.times.mean().item()
            self.std_time = self.times.std().item()
        else:
            raise ValueError("No times recorded.")

        self.times_path = Path(trainer.log_dir) / "times"
        self.times_path.mkdir(parents=True, exist_ok=True)

        np.save(self.times_path / f"{pl_module.name}_times.npy", self.times)
        np.save(self.times_path / f"{pl_module.name}_dims.npy", self.dims)

    def teardown(self, trainer, pl_module, stage):  # noqa: ARG002
        if len(self.times):
            print("-" * 80)
            print(f"Mean inference time: {self.mean_time:.2f} ± {self.std_time:.2f} ms")
            print(f"Saved timing info to {self.times_path!r}")
            print("-" * 80)
