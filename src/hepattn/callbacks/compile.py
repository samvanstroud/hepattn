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
        self._compiled = False
        if skip_dynamic_cudagraphs:
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True  # noqa: SLF001

    def setup(self, trainer, pl_module, stage):
        self.trainer = trainer

    def on_train_start(self, trainer, pl_module) -> None:
        # Compile after Lightning's sanity check so the first traced graph is
        # created under the training grad-mode, avoiding immediate recompiles.
        if self._compiled:
            return
        self.trainer = trainer
        self.compile(pl_module)
        self._compiled = True

    def on_test_start(self, trainer, pl_module) -> None:
        if self._compiled:
            return
        self.trainer = trainer
        self.compile(pl_module)
        self._compiled = True

    def compile(self, module):
        model = dict(module.named_children()).get("model")
        if model is None:
            return
        if self.trainer.is_global_zero:
            print("-" * 80)
            print("compiling model...")
        for name, submodule in model.named_children():
            if name in {"encoder", "decoder"} and isinstance(submodule, torch.nn.Module):
                if self.trainer.is_global_zero:
                    print(f" -> compiling model.{name}...")
                submodule.compile(dynamic=self.dynamic, mode=self.mode)
        if self.trainer.is_global_zero:
            print("-" * 80, "\n")
