from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from hepattn.experiments.odd.data import ODDEventDataModule, ODDEventDataset
from hepattn.experiments.odd.event_display import plot_odd_event


# Setup the dataloader
config_path = Path("src/hepattn/experiments/odd/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0

datamodule = ODDEventDataModule(**config)
datamodule.setup(stage="test")
test_dataloader = datamodule.test_dataloader()
data_iter = iter(test_dataloader)

# Read from the dataset
for _i in range(2):
    inputs, targets = next(data_iter)

data = inputs | targets

for k, v in data.items():
    print(k, v.shape)

# Define plotting config
plot_save_dir = Path("src/hepattn/experiments/odd/plots/event_displays/")
plot_save_dir.mkdir(exist_ok=True, parents=True)

# Plot full detector for particles
fig = plot_odd_event(
    data,
    [
        {"x": "x", "y": "y", "input_names": ["sihit", "calohit"], "xlabel": r"Global $x$", "ylabel": r"Global $y$"},
        {"x": "z", "y": "y", "input_names": ["sihit", "calohit"], "xlabel": r"Global $z$", "ylabel": r"Global $y$"}
    ],
    "particle",
)

fig.savefig(plot_save_dir / Path("particles_event_display.png"))

# Plot tracker for particles
fig = plot_odd_event(
    data,
    [
        {"x": "x", "y": "y", "input_names": ["sihit"], "xlabel": r"Global $x$", "ylabel": r"Global $y$"},
        {"x": "z", "y": "y", "input_names": ["sihit"], "xlabel": r"Global $z$", "ylabel": r"Global $y$"}
    ],
    "particle",
)

fig.savefig(plot_save_dir / Path("particles_event_display_tracker.png"))

# Plot tracker for tracks
fig = plot_odd_event(
    data,
    [
        {"x": "x", "y": "y", "input_names": ["sihit"], "xlabel": r"Global $x$", "ylabel": r"Global $y$"},
        {"x": "z", "y": "y", "input_names": ["sihit"], "xlabel": r"Global $z$", "ylabel": r"Global $y$"}
    ],
    "track",
)

fig.savefig(plot_save_dir / Path("tracks_event_display_tracker.png"))