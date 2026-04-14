import json
import re
from pathlib import Path

import torch
import torch.distributed as dist
from lightning import Callback, LightningModule, Trainer


class MemoryStats(Callback):
    """Record peak CUDA memory usage for fit/test runs."""

    def __init__(self, save_dirname: str = "memory") -> None:
        super().__init__()
        self.save_dirname = save_dirname
        self._saved_stages: set[str] = set()

    def _get_cuda_device(self, trainer: Trainer) -> torch.device | None:
        device = getattr(trainer.strategy, "root_device", None)
        if device is None or device.type != "cuda":
            return None
        return device

    def _reset_peaks(self, trainer: Trainer) -> None:
        device = self._get_cuda_device(trainer)
        if device is None:
            return
        torch.cuda.reset_peak_memory_stats(device)

    def _sanitize_name(self, value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
        return sanitized or "run"

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._saved_stages.discard("fit")
        self._reset_peaks(trainer)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._saved_stages.discard("test")
        self._reset_peaks(trainer)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._saved_stages.discard("predict")
        self._reset_peaks(trainer)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._save_stats(trainer, pl_module, stage="fit")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._save_stats(trainer, pl_module, stage="test")

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._save_stats(trainer, pl_module, stage="predict")

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        stage = self._infer_stage(trainer)
        self._save_stats(trainer, pl_module, stage=stage, interrupted=True, exception=exception)

    def _infer_stage(self, trainer: Trainer) -> str:
        if getattr(trainer, "testing", False):
            return "test"
        if getattr(trainer, "predicting", False):
            return "predict"
        return "fit"

    def _save_stats(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
        interrupted: bool = False,
        exception: BaseException | None = None,
    ) -> None:
        if stage in self._saved_stages:
            return

        device = self._get_cuda_device(trainer)
        if device is None:
            return

        sync_error = None
        try:
            torch.cuda.synchronize(device)
        except RuntimeError as err:
            sync_error = repr(err)

        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        current_allocated = int(torch.cuda.memory_allocated(device))
        current_reserved = int(torch.cuda.memory_reserved(device))
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)

        local_summary = {
            "stage": stage,
            "run_name": pl_module.name,
            "global_rank": int(trainer.global_rank),
            "world_size": int(trainer.world_size),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "total_device_memory_bytes": total_memory,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "current_allocated_bytes": current_allocated,
            "current_reserved_bytes": current_reserved,
            "peak_allocated_gb": peak_allocated / 1024**3,
            "peak_reserved_gb": peak_reserved / 1024**3,
            "current_allocated_gb": current_allocated / 1024**3,
            "current_reserved_gb": current_reserved / 1024**3,
            "peak_allocated_fraction_of_device": peak_allocated / total_memory,
            "peak_reserved_fraction_of_device": peak_reserved / total_memory,
            "interrupted": interrupted,
        }
        if exception is not None:
            local_summary["exception_type"] = type(exception).__name__
            local_summary["exception_message"] = str(exception)
        if sync_error is not None:
            local_summary["synchronize_error"] = sync_error

        if dist.is_available() and dist.is_initialized():
            local_tensor = torch.tensor([peak_allocated, peak_reserved], device=device, dtype=torch.int64)
            gathered = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local_tensor)
            gathered_cpu = torch.stack(gathered).cpu()
            rank_summaries = [
                {
                    "global_rank": rank,
                    "peak_allocated_bytes": int(values[0].item()),
                    "peak_reserved_bytes": int(values[1].item()),
                    "peak_allocated_gb": float(values[0].item() / 1024**3),
                    "peak_reserved_gb": float(values[1].item() / 1024**3),
                }
                for rank, values in enumerate(gathered_cpu)
            ]
            global_peak_allocated = max(item["peak_allocated_bytes"] for item in rank_summaries)
            global_peak_reserved = max(item["peak_reserved_bytes"] for item in rank_summaries)
        else:
            rank_summaries = [
                {
                    "global_rank": int(trainer.global_rank),
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "peak_allocated_gb": peak_allocated / 1024**3,
                    "peak_reserved_gb": peak_reserved / 1024**3,
                }
            ]
            global_peak_allocated = peak_allocated
            global_peak_reserved = peak_reserved

        if trainer.is_global_zero:
            log_dir = trainer.log_dir or trainer.default_root_dir
            assert log_dir is not None
            out_dir = Path(log_dir) / self.save_dirname
            out_dir.mkdir(parents=True, exist_ok=True)
            run_name = self._sanitize_name(pl_module.name)
            output_path = out_dir / f"{run_name}_{stage}_memory_summary.json"

            summary = {
                **local_summary,
                "global_peak_allocated_bytes": global_peak_allocated,
                "global_peak_reserved_bytes": global_peak_reserved,
                "global_peak_allocated_gb": global_peak_allocated / 1024**3,
                "global_peak_reserved_gb": global_peak_reserved / 1024**3,
                "rank_summaries": rank_summaries,
            }

            with output_path.open("w") as f:
                json.dump(summary, f, indent=2, sort_keys=False)

            print("-" * 80)
            print(f"{stage} peak allocated memory: {summary['global_peak_allocated_gb']:.2f} GB")
            print(f"{stage} peak reserved memory: {summary['global_peak_reserved_gb']:.2f} GB")
            if interrupted:
                print(f"{stage} run ended early with {summary.get('exception_type', 'an exception')}")
            print(f"Saved memory summary to {output_path!s}")
            print("-" * 80)

        self._saved_stages.add(stage)
