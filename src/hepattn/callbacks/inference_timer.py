import warnings
from pathlib import Path

import numpy as np
import torch
from lightning import Callback


class InferenceTimer(Callback):
    def __init__(self, do_compile=True):
        super().__init__()
        self.times = []
        self.dims = []
        self.n_warm_start = 10
        self._tmp_dims = None
        self.compile = do_compile

    def on_test_start(self, trainer, pl_module):
        assert trainer.global_rank == 0, "InferenceTimer should only be used with a single process."
        model = pl_module
        if hasattr(model, "model"):
            model = model.model
        self.old_forward = model.forward
        if self.compile:
            compiled_forward = torch.compile(self.old_forward)

        @torch._dynamo.disable(recursive=False)
        def new_forward(*args, **kwargs):
            self._tmp_dims = sum(v.shape[1] for v in args[0].values())
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = compiled_forward(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            self.times.append(start.elapsed_time(end))
            return out

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
            self.dims.append(self._tmp_dims)
            self._tmp_dims = None

    def on_test_end(self, trainer, pl_module):
        pl_module.forward = self.old_forward
        self.times = self.times[self.n_warm_start :]  # ensure warm start
        self.dims = self.dims[self.n_warm_start :]

        if not len(self.times):
            raise ValueError("No times recorded.")

        self.times = torch.tensor(self.times)
        self.mean_time = self.times.mean().item()
        self.std_time = self.times.std().item()

        self.times_path = Path(trainer.log_dir) / "times"
        self.times_path.mkdir(parents=True, exist_ok=True)

        np.save(self.times_path / f"{pl_module.name}_times.npy", self.times)
        np.save(self.times_path / f"{pl_module.name}_dims.npy", self.dims)

    def teardown(self, trainer, pl_module, stage):
        if len(self.times):
            print("-" * 80)
            print(f"Mean inference time: {self.mean_time:.2f} ± {self.std_time:.2f} ms")
            print(f"Saved timing info to {self.times_path}")
            print("-" * 80)
