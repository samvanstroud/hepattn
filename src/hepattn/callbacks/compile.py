import torch
from lightning import Callback


class Compile(Callback):
    def __init__(
        self,
        dynamic: bool = True,
        mode: str | None = None,
        skip_dynamic_cudagraphs: bool = False,
    ):
        super().__init__()
        self.dynamic = dynamic
        self.mode = mode
        if skip_dynamic_cudagraphs:
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True  # noqa: SLF001

    def setup(self, trainer, pl_module, stage):
        self.trainer = trainer
        self.compile(pl_module)

    def compile(self, module):
        if self.trainer.is_global_zero:
            print("-" * 80)
            print("compiling model...")
        for name, submodule in module.named_children():
            if submodule.__class__.__module__.startswith("torchmetrics"):
                continue
            if self.trainer.is_global_zero:
                print(f" -> compiling {name}...")
            submodule.compile(dynamic=self.dynamic, mode=self.mode)
        if self.trainer.is_global_zero:
            print("-" * 80, "\n")
