import warnings
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
        self.query_counts = []
        self.n_warm_start = 20
        self._tmp_dims = None
        self._wrapped_module = None

        self.peak_allocated = []
        self.peak_reserved = []

    def _extract_input_dims(self, inputs):
        dim_summary = {}
        total_valid = 0

        for key, value in inputs.items():
            if not key.endswith("_valid"):
                continue

            valid_count = int(value.to(dtype=torch.int64).sum().item())
            dim_summary[key] = valid_count
            total_valid += valid_count

        if dim_summary:
            dim_summary["total_valid"] = total_valid

        return dim_summary

    def on_test_start(self, trainer, pl_module):
        assert trainer.global_rank == 0, "InferenceTimer should only be used with a single process."
        model = pl_module
        if hasattr(model, "model"):
            model = model.model
        self._wrapped_module = model
        self.old_forward = model.forward

        # Pick device for mem stats
        self._cuda = torch.cuda.is_available()
        self._device = getattr(pl_module, "device", None)
        # If device isn't a CUDA device (e.g. cpu), fall back to current device
        if self._cuda and (self._device is None or self._device.type != "cuda"):
            self._device = torch.device("cuda", torch.cuda.current_device())

        def new_forward(*args, **kwargs):
            self._tmp_dims = self._extract_input_dims(args[0])

            if self._cuda:
                # Reset peak counters so "max_*" corresponds to this forward pass
                torch.cuda.synchronize(self._device)
                torch.cuda.reset_peak_memory_stats(self._device)
                base_alloc = torch.cuda.memory_allocated(self._device)
                base_rsvd = torch.cuda.memory_reserved(self._device)

            with cuda_timer(self.times):
                out = self.old_forward(*args, **kwargs)

            if self._cuda:
                # Make sure kernels are done before reading stats
                torch.cuda.synchronize(self._device)
                peak_alloc = torch.cuda.max_memory_allocated(self._device) - base_alloc
                peak_rsvd = torch.cuda.max_memory_reserved(self._device) - base_rsvd
                self.peak_allocated.append(peak_alloc)
                self.peak_reserved.append(peak_rsvd)

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

    def _extract_query_count(self, pl_module, test_step_outputs):
        model_outputs = test_step_outputs[0] if isinstance(test_step_outputs, (tuple, list)) else test_step_outputs

        if isinstance(model_outputs, dict):
            encoder_outputs = model_outputs.get("encoder", {})
            query_mask = encoder_outputs.get("query_mask")
            if query_mask is not None:
                return int(query_mask.to(dtype=torch.int64).sum().item())

        model = getattr(pl_module, "model", pl_module)
        decoder = getattr(model, "decoder", None)
        static_num_queries = getattr(decoder, "_num_queries", None)
        if static_num_queries is None:
            return None
        return int(static_num_queries)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._tmp_dims is not None:
            self.dims.append(self._tmp_dims)
            self._tmp_dims = None

        query_count = self._extract_query_count(pl_module, outputs)
        if query_count is not None:
            self.query_counts.append(query_count)

    def on_test_end(self, trainer, pl_module):
        if self._wrapped_module is not None:
            self._wrapped_module.forward = self.old_forward

        if not len(self.times):
            raise ValueError("No times recorded.")

        # ensure warm start
        self.times = self.times[self.n_warm_start :]
        self.dims = self.dims[self.n_warm_start :]
        self.query_counts = self.query_counts[self.n_warm_start :]

        if self._cuda:
            self.peak_allocated = self.peak_allocated[self.n_warm_start :]
            self.peak_reserved = self.peak_reserved[self.n_warm_start :]

        if not len(self.times):
            print("Not enough steps to obtain timing information")
            return

        self.times = torch.tensor(self.times)
        self.mean_time = self.times.mean().item()
        self.std_time = self.times.std().item()

        if self._cuda and len(self.peak_allocated):
            alloc = torch.tensor(self.peak_allocated, dtype=torch.float64)
            rsvd = torch.tensor(self.peak_reserved, dtype=torch.float64)
            self.mean_peak_alloc_mb = (alloc.mean().item()) / (1024**2)
            self.max_peak_alloc_mb = (alloc.max().item()) / (1024**2)
            self.mean_peak_rsvd_mb = (rsvd.mean().item()) / (1024**2)
            self.max_peak_rsvd_mb = (rsvd.max().item()) / (1024**2)

        self.times_path = Path(trainer.log_dir) / "times"
        self.times_path.mkdir(parents=True, exist_ok=True)

        dims_by_key = {}
        if self.dims:
            for key in sorted({key for dim in self.dims for key in dim}):
                dims_by_key[key] = np.asarray([dim[key] for dim in self.dims], dtype=np.int64)

        np.save(self.times_path / f"{pl_module.name}_times.npy", self.times.cpu().numpy())
        np.save(self.times_path / f"{pl_module.name}_dims.npy", dims_by_key)
        np.save(self.times_path / f"{pl_module.name}_query_counts.npy", np.asarray(self.query_counts))

        if self._cuda and len(self.peak_allocated):
            np.save(self.times_path / f"{pl_module.name}_peak_allocated_bytes.npy", np.asarray(self.peak_allocated))
            np.save(self.times_path / f"{pl_module.name}_peak_reserved_bytes.npy", np.asarray(self.peak_reserved))

    def teardown(self, trainer, pl_module, stage):
        if len(self.times):
            print("-" * 80)
            print(f"Mean inference time: {self.mean_time:.2f} ± {self.std_time:.2f} ms")

            if getattr(self, "_cuda", False) and len(getattr(self, "peak_allocated", [])):
                print(f"Peak GPU mem per step (allocated): mean {self.mean_peak_alloc_mb:.1f} MB, max {self.max_peak_alloc_mb:.1f} MB")
                print(f"Peak GPU mem per step (reserved):  mean {self.mean_peak_rsvd_mb:.1f} MB, max {self.max_peak_rsvd_mb:.1f} MB")

            print(f"Saved timing info to {self.times_path}")
            print("-" * 80)
