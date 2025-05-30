from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from hepattn.experiments.tide.data import ROIDataModule

plt.rcParams["figure.dpi"] = 300


def plot_roi(inputs, targets):
    track = "sudo"

    # batch_idx = torch.argmax(targets[f"{track}_valid"].sum(-1))
    batch_idx = torch.argmax(inputs["sct_valid"].sum(-1))

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)

    colormap = plt.cm.tab10
    cycler = [colormap(i) for i in range(colormap.N)]

    for i, hit in enumerate(["pix", "sct"]):
        hit_x = inputs[f"{hit}_r"][batch_idx]
        hit_y = inputs[f"{hit}_dphi"][batch_idx]
        hit_valid = inputs[f"{hit}_valid"][batch_idx]

        mask = targets[f"{track}_{hit}_valid"][batch_idx]

        ax[i].scatter(hit_x[hit_valid], hit_y[hit_valid], s=16.0, marker="s", fc="none", ec="black")

        track_valid = targets[f"{track}_valid"][batch_idx]

        for track_idx in range(track_valid.shape[-1]):
            if not track_valid[track_idx]:
                continue

            color = cycler[track_idx % len(cycler)]

            track_hit_x = hit_x[mask[track_idx]]
            track_hit_y = hit_y[mask[track_idx]]

            sort_idx = torch.argsort(track_hit_x)

            ax[i].plot(track_hit_x[sort_idx], track_hit_y[sort_idx], color=color)

    num_pix = inputs["pix_valid"][batch_idx].sum()
    num_sct = inputs["sct_valid"][batch_idx].sum()

    roi_id = targets["sample_id"][batch_idx].item()

    ax[0].set_title(f"ROI {roi_id}: {num_pix} pixel hits, {num_sct} SCT hits")

    fig.tight_layout()
    fig.savefig(Path("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/plots/roi_display.png"))


config_path = Path("src/hepattn/experiments/tide/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0
config["batch_size"] = 100
config["num_test"] = 10000

datamodule = ROIDataModule(**config)
datamodule.setup(stage="test")
dataloader = datamodule.test_dataloader()
dataset = dataloader.dataset
data_iterator = iter(dataloader)
inputs, targets = next(data_iterator)

plot_roi(inputs, targets)
