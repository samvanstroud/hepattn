from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from hepattn.experiments.odd.data import ODDEventDataModule, ODDEventDataset
from hepattn.experiments.odd.event_display import plot_odd_event


config_path = Path("src/hepattn/experiments/odd/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0

datamodule = ODDEventDataModule(**config)
datamodule.setup(stage="test")
test_dataloader = datamodule.test_dataloader()
data_iter = iter(test_dataloader)


inputs, targets = next(data_iter)

inputs, targets = next(data_iter)

for k, v in inputs.items():
    print("input:  ", k, v.shape)

for k, v in targets.items():
    print("target: ", k, v.shape)

ax_spec = [
    {"x": "x", "y": "y", "input_names": ["sihit", "calohit"], "xlabel": r"Global $x$", "ylabel": r"Global $y$"},
    {"x": "z", "y": "y", "input_names": ["sihit", "calohit"], "xlabel": r"Global $z$", "ylabel": r"Global $y$"}
]
data = inputs | targets

fig = plot_odd_event(
    data,
    ax_spec,
    "particle",
)

fig.savefig("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/odd/display.png")