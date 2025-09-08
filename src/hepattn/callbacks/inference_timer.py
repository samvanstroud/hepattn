import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from lightning import Callback

from hepattn.utils.cuda_timer import cuda_timer


class InferenceTimer(Callback):
    def __init__(self):
        super().__init__()
        self.times = []
        self.dims = defaultdict(list)
        self.n_warm_start = 10
        self._tmp_dims = None

    def on_test_start(self, trainer, pl_module):
        assert trainer.global_rank == 0, "InferenceTimer should only be used with a single process."
        model = pl_module
        if hasattr(model, "model"):
            model = model.model
        self.old_forward = model.forward

        def new_forward(*args, **kwargs):
            # Only record valid field sizes, args[0] gets the model inputs
            self._tmp_dims = {k: v.shape for k, v in args[0].items() if "valid" in k}

            with cuda_timer(self.times):
                return self.old_forward(*args, **kwargs)

        model.forward = new_forward

        matmul_precision = torch.get_float32_matmul_precision()
        if matmul_precision in {"high", "highest"}:
            warnings.warn(
                f"""The current float32 matmul precision is set to {matmul_precision},
            which may impact inference times. Consider if `low` or `medium` matmul
            precision can be used instead.""",
                UserWarning,
            )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._tmp_dims is not None:
            for k, v in self._tmp_dims.items():
                self.dims[k].append(v)
            self._tmp_dims = None

    def on_test_end(self, trainer, pl_module):
        pl_module.forward = self.old_forward

        if not len(self.times):
            raise ValueError("No times recorded.")

        # ensure warm start
        self.times = self.times[self.n_warm_start :]
        self.dims = self.dims[self.n_warm_start :]

        if not len(self.times):
            print("Not enough steps to obtain timing information")
            return

        self.times = torch.tensor(self.times)
        self.dims = {k: torch.tensor(v) for k, v in self.dims.items()}

        self.times_path = Path(trainer.log_dir) / "times"
        self.times_path.mkdir(parents=True, exist_ok=True)

        np.save(self.times_path / f"{pl_module.name}_times.npy", self.times)
        np.save(self.times_path / f"{pl_module.name}_dims.npy", self.dims)

    def teardown(self, trainer, pl_module, stage):
        if len(self.times):
            times = torch.tensor(self.times)
            warm_times = torch.tensor(self.times[self.n_warm_start :])
            print("-" * 80)
            print(f"Mean inference time: {times.mean().item():.2f} ± {times.std().item():.2f} ms")
            print(f"Mean inference time (warm start): {warm_times.mean().item():.2f} ± {warm_times.std().item():.2f} ms")
            print(f"Saved timing info to {self.times_path}")
            print("-" * 80)
