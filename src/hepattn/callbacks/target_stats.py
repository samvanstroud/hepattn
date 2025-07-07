from collections import defaultdict

from lightning import Callback


class TargetStats(Callback):
    def __init__(self):
        super().__init__()
        self.stats = defaultdict(lambda: {"pos": 0, "neg": 0})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, targets = batch
        # calculate running means for each target
        for target, values in targets.items():
            if not target.endswith("_valid"):
                continue

            self.stats[target]["pos"] += values.bool().sum().item()
            self.stats[target]["neg"] += (~values.bool()).sum().item()

            pos_frac = self.stats[target]["pos"] / (self.stats[target]["pos"] + self.stats[target]["neg"])
            pl_module.log(f"train/{target}_pos_frac", pos_frac, sync_dist=True)
