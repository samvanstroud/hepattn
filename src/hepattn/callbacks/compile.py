from lightning import Callback


class Compile(Callback):
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
            submodule.compile(dynamic=True)
        if self.trainer.is_global_zero:
            print("-" * 80, "\n")
