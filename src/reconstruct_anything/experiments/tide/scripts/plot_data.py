# ruff: noqa
"""Plot TIDE data distributions with ATLAS-style formatting.

Usage:
    pixi run python src/reconstruct_anything/experiments/tide/scripts/plot_data.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from atlasify import atlasify
from tqdm import tqdm

from reconstruct_anything.experiments.tide.data.data import ROIDataModule

SUB_FONTSIZE = 16
ATLAS_LABEL = r"$\sqrt{s} = 13\,\mathrm{TeV},\; Z'\!\rightarrow q\bar{q}$"

# ── Plot style (matching eval plots) ─────────────────────────────────────────

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

# ── Configuration ────────────────────────────────────────────────────────────

SAVE_FORMAT = "png"  # "png" or "pdf"
NUM_ROIS = 1000
BATCH_SIZE = 250

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots" / "data"

HIT_COLOR = "cornflowerblue"

# ── Data loading ─────────────────────────────────────────────────────────────


PIX_FIELDS = ["pix_x", "pix_y", "pix_z", "pix_r", "pix_eta", "pix_phi", "pix_deta", "pix_dphi"]
TRACK_FIELDS = ["sudo_pt", "sudo_eta", "sudo_phi"]


def load_data():
    """Load TIDE ROI data, returning flat 1D arrays of valid hits and tracks."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    config = yaml.safe_load(config_path.read_text())["data"]
    config["num_workers"] = 0
    config["batch_size"] = BATCH_SIZE
    config["num_test"] = NUM_ROIS
    config["test_dir"] = "/share/rcifdata/maxhart/data/ambi/train/"

    datamodule = ROIDataModule(**config)
    datamodule.setup(stage="test")

    pix_data = {f: [] for f in PIX_FIELDS}
    track_data = {f: [] for f in TRACK_FIELDS}
    n_rois = 0

    for inputs, targets in tqdm(datamodule.test_dataloader(), desc="Loading ROIs"):
        n_rois += inputs["pix_valid"].shape[0]
        pix_valid = inputs["pix_valid"]
        sudo_valid = targets["sudo_valid"]

        for f in PIX_FIELDS:
            pix_data[f].append(inputs[f][pix_valid])
        for f in TRACK_FIELDS:
            track_data[f].append(targets[f][sudo_valid])

    pix = {f: torch.cat(v) for f, v in pix_data.items()}
    tracks = {f: torch.cat(v) for f, v in track_data.items()}
    return pix, tracks, n_rois


# ── Plotting helpers ─────────────────────────────────────────────────────────


def plot_1d_distributions(data, fields, field_labels, title, filename, ylabel="Hits", bins=64):
    """Plot 1D distributions for a set of fields (data already masked/flat)."""
    n_fields = len(fields)
    fig, axes = plt.subplots(1, n_fields, figsize=(4 * n_fields, 4))
    if n_fields == 1:
        axes = [axes]

    for ax, field, label in zip(axes, fields, field_labels):
        values = data[field].numpy()
        ax.hist(values, bins=bins, histtype="step", color=HIT_COLOR, linewidth=1.5)
        ax.set_xlabel(label)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(zorder=0, alpha=0.25, linestyle="--")

    atlasify("Simulation Internal", ATLAS_LABEL, sub_font_size=SUB_FONTSIZE)
    fig.savefig(PLOT_DIR / f"{filename}.{SAVE_FORMAT}")
    plt.close(fig)


def plot_2d_scatter(x_data, y_data, xlabel, ylabel, title, filename, alpha=0.1, s=0.5):
    """Plot a 2D scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_data, y_data, alpha=alpha, s=s, color=HIT_COLOR, rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    atlasify("Simulation Internal", ATLAS_LABEL, sub_font_size=SUB_FONTSIZE)
    fig.savefig(PLOT_DIR / f"{filename}.{SAVE_FORMAT}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    torch.manual_seed(42)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Loading {NUM_ROIS} ROIs...")
    pix, tracks, n_rois = load_data()
    print(f"Loaded {n_rois} ROIs, {len(pix['pix_x'])} pixel hits, {len(tracks['sudo_pt'])} tracks")

    # ── Hit coordinate distributions ─────────────────────────────────────

    plot_1d_distributions(
        pix,
        ["pix_x", "pix_y", "pix_z"],
        [r"Pixel $x$ [mm]", r"Pixel $y$ [mm]", r"Pixel $z$ [mm]"],
        f"Pixel Hit Cartesian Coordinates ({n_rois} ROIs)",
        "pix_cartesian",
    )

    plot_1d_distributions(
        pix,
        ["pix_r", "pix_eta", "pix_phi"],
        [r"Pixel $r$ [mm]", r"Pixel $\eta$", r"Pixel $\phi$"],
        f"Pixel Hit Cylindrical Coordinates ({n_rois} ROIs)",
        "pix_cylindrical",
    )

    plot_1d_distributions(
        pix,
        ["pix_deta", "pix_dphi"],
        [r"Pixel $\Delta\eta$", r"Pixel $\Delta\phi$"],
        f"Pixel Hit ROI-Frame Coordinates ({n_rois} ROIs)",
        "pix_roi_frame",
    )

    # ── 2D hit maps ──────────────────────────────────────────────────────

    plot_2d_scatter(
        pix["pix_x"].numpy(),
        pix["pix_y"].numpy(),
        r"Pixel $x$ [mm]",
        r"Pixel $y$ [mm]",
        f"Pixel Hits $x$-$y$ ({n_rois} ROIs)",
        "pix_xy",
    )

    plot_2d_scatter(
        pix["pix_z"].numpy(),
        pix["pix_r"].numpy(),
        r"Pixel $z$ [mm]",
        r"Pixel $r$ [mm]",
        f"Pixel Hits $z$-$r$ ({n_rois} ROIs)",
        "pix_zr",
    )

    # ── Track kinematics ─────────────────────────────────────────────────

    plot_1d_distributions(
        tracks,
        ["sudo_pt", "sudo_eta", "sudo_phi"],
        [r"Track $p_\mathrm{T}$ [GeV]", r"Track $\eta$", r"Track $\phi$"],
        f"Track Kinematics ({n_rois} ROIs)",
        "track_kinematics",
        ylabel="Tracks",
    )

    pt = tracks["sudo_pt"].numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pt, bins=np.geomspace(1, 1000, 64), histtype="step", color=HIT_COLOR, linewidth=1.5)
    ax.set_xlabel(r"Track $p_\mathrm{T}$ [GeV]")
    ax.set_ylabel("Tracks")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    atlasify("Simulation Internal", ATLAS_LABEL, sub_font_size=SUB_FONTSIZE)
    fig.savefig(PLOT_DIR / f"track_pt_log.{SAVE_FORMAT}")
    plt.close(fig)

    print(f"\nPlots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
