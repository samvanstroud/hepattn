# ruff: noqa
"""Plot TIDE ROI event displays with ATLAS-style formatting.

Usage:
    pixi run python src/reconstruct_anything/experiments/tide/scripts/plot_roi.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from atlasify import atlasify
from reconstruct_anything.experiments.tide.data.data import ROIDataModule

# ── Plot style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex": False,
    "figure.dpi": 300,
    "font.size": 16,
    "figure.constrained_layout.use": True,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
})

SAVE_FORMAT = "png"
SUB_FONTSIZE = 16
PLOT_DIR = Path(__file__).resolve().parent.parent / "plots" / "roi_displays"


def iqr_limits(values, margin=2.0, hard_limit=0.05):
    """Compute axis limits based on IQR to exclude outliers.

    Returns (lo, hi) covering Q1 - margin*IQR to Q3 + margin*IQR,
    clamped to [-hard_limit, +hard_limit].
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo = max(q1 - margin * iqr, -hard_limit)
    hi = min(q3 + margin * iqr, hard_limit)
    return lo, hi


def plot_roi_display(inputs, targets, batch_idx, plot_dir):
    """Plot an ROI event display showing pixel hits coloured by track assignment."""
    track = "sudo"

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    colormap = plt.cm.tab10
    cycler = [colormap(i) for i in range(colormap.N)]

    hit_valid = inputs["pix_valid"][batch_idx]
    mask = targets[f"{track}_pix_valid"][batch_idx]
    track_valid = targets[f"{track}_valid"][batch_idx]

    # Left: r vs dphi
    hit_r = inputs["pix_r"][batch_idx]
    hit_dphi = inputs["pix_dphi"][batch_idx]
    hit_deta = inputs["pix_deta"][batch_idx]

    # Compute IQR-based limits to focus on the ROI core
    dphi_vals = hit_dphi[hit_valid].numpy()
    deta_vals = hit_deta[hit_valid].numpy()
    dphi_lo, dphi_hi = iqr_limits(dphi_vals)
    deta_lo, deta_hi = iqr_limits(deta_vals)

    ax[0].scatter(hit_r[hit_valid], hit_dphi[hit_valid], s=16.0, marker="s", fc="none", ec="black", linewidths=0.5)
    for track_idx in range(track_valid.shape[-1]):
        if not track_valid[track_idx]:
            continue
        color = cycler[track_idx % len(cycler)]
        tx = hit_r[mask[track_idx]]
        ty = hit_dphi[mask[track_idx]]
        sort_idx = torch.argsort(tx)
        ax[0].plot(tx[sort_idx], ty[sort_idx], color=color, linewidth=1.5)

    ax[0].set_xlabel(r"Pixel $r$ [mm]")
    ax[0].set_ylabel(r"Pixel $\Delta\phi$")
    ax[0].set_ylim(dphi_lo, dphi_hi)
    ax[0].grid(zorder=0, alpha=0.25, linestyle="--")

    # Right: deta vs dphi
    ax[1].scatter(hit_deta[hit_valid], hit_dphi[hit_valid], s=16.0, marker="s", fc="none", ec="black", linewidths=0.5)
    for track_idx in range(track_valid.shape[-1]):
        if not track_valid[track_idx]:
            continue
        color = cycler[track_idx % len(cycler)]
        tx = hit_deta[mask[track_idx]]
        ty = hit_dphi[mask[track_idx]]
        sort_idx = torch.argsort(tx)
        ax[1].plot(tx[sort_idx], ty[sort_idx], color=color, linewidth=1.5)

    ax[1].set_xlabel(r"Pixel $\Delta\eta$")
    ax[1].set_ylabel(r"Pixel $\Delta\phi$")
    ax[1].set_xlim(deta_lo, deta_hi)
    ax[1].set_ylim(dphi_lo, dphi_hi)
    ax[1].grid(zorder=0, alpha=0.25, linestyle="--")

    num_pix = hit_valid.sum().item()
    num_tracks = track_valid.sum().item()
    roi_id = targets["sample_id"][batch_idx].item()

    atlasify(
        "Simulation Internal",
        rf"$\sqrt{{s}} = 13\,\mathrm{{TeV}},\; Z'\!\rightarrow q\bar{{q}}$" + f"\nROI {roi_id}: {num_pix} pixel hits, {num_tracks} tracks",
        sub_font_size=SUB_FONTSIZE,
    )

    fig.savefig(plot_dir / f"roi_display.{SAVE_FORMAT}")
    plt.close(fig)


def main():
    torch.manual_seed(42)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    config = yaml.safe_load(config_path.read_text())["data"]
    config["num_workers"] = 0
    config["batch_size"] = 100
    config["num_test"] = 1000
    config["test_dir"] = "/share/rcifdata/maxhart/data/ambi/train/"

    datamodule = ROIDataModule(**config)
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    inputs, targets = next(iter(dataloader))

    # Find the ROI containing the most-shared pixel hit
    # sudo_pix_valid is [B, num_tracks, num_pix] — count tracks per pixel
    tracks_per_pix = targets["sudo_pix_valid"].sum(dim=-2)  # [B, num_pix]
    max_sharing_per_roi = tracks_per_pix.max(dim=-1).values  # [B]
    batch_idx = torch.argmax(max_sharing_per_roi)
    max_shared = max_sharing_per_roi[batch_idx].item()

    print(
        f"Plotting ROI {targets['sample_id'][batch_idx].item()} "
        f"(most-shared pixel has {max_shared} tracks, "
        f"{targets['sudo_valid'][batch_idx].sum().item()} total tracks, "
        f"{inputs['pix_valid'][batch_idx].sum().item()} pixel hits)"
    )

    plot_roi_display(inputs, targets, batch_idx, PLOT_DIR)
    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
