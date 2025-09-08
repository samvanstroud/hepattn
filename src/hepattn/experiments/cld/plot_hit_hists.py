from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.utils.histogram import CountingHistogram
from hepattn.utils.plot import plot_hist_to_ax

plt.rcParams["figure.dpi"] = 300


aliases = {
    "pos.x": "Position $x$ [m]",
    "pos.y": "Position $y$ [m]",
    "pos.z": "Position $z$ [m]",
}

scales = {
    "pos.x": "linear",
    "pos.y": "linear",
    "pos.z": "linear",
}

bins = {
    "pos.x": np.linspace(-5, 5, 32),
    "pos.y": np.linspace(-5, 5, 32),
    "pos.z": np.linspace(-5, 5, 32),
}

hit_aliases = {
    "vtxd": "VTXD",
    "trkr": "Tracker",
    "ecal": "ECAL",
    "hcal": "HCAL",
    "muon": "Muon",
}

hit_colours = {
    "vtxd": "tab:blue",
    "trkr": "tab:orange",
    "ecal": "tab:green",
    "hcal": "tab:red",
    "muon": "tab:purple",
}


hists = {field: {hit: CountingHistogram(bins) for hit in hit_aliases} for field, bins in bins.items()}

# Setup the dataset
config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 10
config["batch_size"] = 10
config["num_test"] = 10000

datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")
dataloader = datamodule.test_dataloader()
data_iterator = iter(dataloader)

# Iterate through the dataset
for _i in tqdm(range(10)):
    inputs, targets = next(data_iterator)

    for field, hits in hists.items():
        for hit, hist in hits.items():
            values = inputs[f"{hit}_{field}"][inputs[f"{hit}_valid"].bool()]
            hist.fill(values)

plots = {
    "hit_xyz": [
        "pos.x",
        "pos.y",
        "pos.z",
    ],
}

for plot_name, fields in plots.items():
    fig, ax = plt.subplots(1, len(fields))
    fig.set_size_inches(12, 3)

    for ax_idx, field in enumerate(fields):
        for hit, hist in hists[field].items():
            plot_hist_to_ax(
                ax[ax_idx],
                hist.counts,
                hist.bins,
                label=hit_aliases[hit],
                color=hit_colours[hit],
                vertical_lines=True,
            )

        ax[ax_idx].set_yscale("log")
        ax[ax_idx].set_xscale(scales[field])
        ax[ax_idx].set_xlabel(f" {aliases[field]}")
        ax[ax_idx].set_ylabel("Count")
        ax[ax_idx].grid(zorder=0, alpha=0.25, linestyle="--")

    ax[0].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(Path(f"src/hepattn/experiments/cld/plots/data/{plot_name}.png"))
