from lightning import Callback


class Compile(Callback):
    def setup(self, trainer, pl_module, stage):
        self.compile(pl_module)

    def compile(self, module):
        print("-" * 80)
        print("compiling model...")
        for name, submodule in module.named_children():
            print(f" -> compiling {name}...")
            submodule.compile(dynamic=True)
        print("-" * 80, "\n")
