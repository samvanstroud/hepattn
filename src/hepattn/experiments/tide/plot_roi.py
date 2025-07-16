from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from hepattn.experiments.tide.data import ROIDataModule

plt.rcParams["figure.dpi"] = 300


def plot_roi(inputs, targets):
    track = "sudo"

    # batch_idx = torch.argmax(targets[f"{track}_valid"].sum(-1))
    # batch_idx = torch.argmax(inputs["sct_valid"].sum(-1))
    batch_idx = torch.argmax(torch.max(targets["sudo_pix_valid"].sum(-2), dim=-1)[0])

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
    fig.savefig(Path("src/hepattn/experiments/tide/plots/roi_display.png"))

    fig, ax = plt.subplots(4, 4)
    fig.set_size_inches(8, 4)

    ax = ax.flatten()

    sudo_valid = targets["sudo_valid"][batch_idx]
    sudo_pix_valid = targets["sudo_pix_valid"][batch_idx]

    sudo_pix_x = targets["sudo_pix_loc_x"][batch_idx]
    sudo_pix_y = targets["sudo_pix_loc_y"][batch_idx]

    pix_num_sudo = sudo_pix_valid.sum(-2)
    pix_idxs = torch.argsort(pix_num_sudo, descending=True)
    pix_charge_matrices = inputs["pix_log_charge_matrix"][batch_idx]

    for ax_idx in range(len(ax)):
        pix_idx = pix_idxs[ax_idx]

        pix_charge_matrix = torch.flipud(pix_charge_matrices[pix_idx].reshape(7, 7).T)
        pix_charge_matrix[pix_charge_matrix == 0] = torch.nan
        pix_charge_matrix = torch.exp(pix_charge_matrix)

        extent = 3.5

        ax[ax_idx].set_xticks(np.linspace(-extent, extent, int(2 * extent + 1)))
        ax[ax_idx].set_yticks(np.linspace(-extent, extent, int(2 * extent + 1)))

        ax[ax_idx].grid(alpha=0.5, zorder=-100)
        ax[ax_idx].grid(alpha=0.5, zorder=-100)

        ax[ax_idx].set_xticklabels([])
        ax[ax_idx].set_yticklabels([])

        ax[ax_idx].imshow(
            pix_charge_matrix,
            extent=(-3.5, 3.5, -3.5, 3.5),
        )

        for track_idx in range(sudo_valid.shape[-1]):
            if not sudo_pix_valid[track_idx, pix_idx]:
                continue

            ax[ax_idx].scatter(
                sudo_pix_x[track_idx, pix_idx],
                sudo_pix_y[track_idx, pix_idx],
                color=cycler[track_idx % len(cycler)],
                s=32.0,
                ec="black",
                linewidths=1,
            )

    fig.tight_layout()
    fig.savefig(Path("src/hepattn/experiments/tide/plots/roi_pixel_display.png"), transparent=True)


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
