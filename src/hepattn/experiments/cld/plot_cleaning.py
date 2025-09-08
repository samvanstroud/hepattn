from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.plot_event import plot_cld_event

plt.rcParams["figure.dpi"] = 300

plot_save_dir = Path("src/hepattn/experiments/cld/eval_plots/")

config_path = Path("src/hepattn/experiments/cld/configs/uncleaned.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 10
config["batch_size"] = 10


sample_id = 1226276301090181

datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")
dataset = datamodule.test_dataloader().dataset

# data_iterator = iter(datamodule.test_dataloader())
# for i in range(1000):
#     inputs, targets = next(data_iterator)
#     print(i)
#     for j in range(10):
#         if targets["particle_muon_valid"][j].sum(-1).sum(-1) >= 24:
#             print(targets["sample_id"][j], targets["particle_valid"][j].float().sum(-1))

sample = dataset.load_sample(int(sample_id))
inputs, targets = dataset.prep_sample(sample)
data = inputs | targets


hits = ["vtxd", "trkr", "ecal", "hcal", "muon"]

gridspec_kw = {"width_ratios": [1, 1.6]}

# Spec for event displays
event_display_cfg = [
    {"x": "pos.x", "y": "pos.y", "px": "mom.x", "py": "mom.y", "input_names": hits, "xlabel": r"Global $x$", "ylabel": r"Global $y$"},
    {"x": "pos.z", "y": "pos.y", "px": "mom.z", "py": "mom.y", "input_names": hits, "xlabel": r"Global $z$", "ylabel": r"Global $y$"},
]

fig = plot_cld_event(data, event_display_cfg, "particle", gridspec_kw=gridspec_kw)
fig.tight_layout()
fig.savefig(plot_save_dir / Path("event_displays/cleaning_pre.png"))

config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 10
config["batch_size"] = 10

datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")
dataset = datamodule.test_dataloader().dataset

sample = dataset.load_sample(int(sample_id))
inputs, targets = dataset.prep_sample(sample)
data = inputs | targets

fig = plot_cld_event(data, event_display_cfg, "particle", gridspec_kw=gridspec_kw)
fig.tight_layout()
fig.savefig(plot_save_dir / Path("event_displays/cleaning_post.png"))
