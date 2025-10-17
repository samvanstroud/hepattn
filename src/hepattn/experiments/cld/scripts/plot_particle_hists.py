import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.utils.histogram import CountingHistogram
from hepattn.utils.plotting import plot_hist_to_ax

plt.rcParams["figure.dpi"] = 300


aliases = {
    "calib_energy_ecal": "Total Calibrated ECAL Energy [GeV]",
    "calib_energy_hcal": "Total Calibrated HCAL Energy [GeV]",
    "mom.r": r"$p_T$",
    "mom.eta": r"$\eta$",
    "mom.phi": r"$\phi$",
    "vtx.r": r"Vertex $r$",
    "isolation": r"Particle Angular Isolation",
    "num_vtxd": "Num. VTXD Hits",
    "num_trkr": "Num. Tracker Hits",
    "num_sihit": "Num. Si Hits",
    "num_ecal": "Num. ECAL Hits",
    "num_hcal": "Num. HCAL Hits",
    "num_muon": "Num. Muon Hits",
    "mass": "Mass [GeV]",
    "energy": "Energy [GeV]",
}

scales = {
    "calib_energy_ecal": "log",
    "calib_energy_hcal": "log",
    "mom.r": "log",
    "mom.eta": "linear",
    "mom.phi": "linear",
    "vtx.r": "linear",
    "isolation": "log",
    "num_vtxd": "linear",
    "num_trkr": "linear",
    "num_sihit": "linear",
    "num_ecal": "log",
    "num_hcal": "log",
    "num_muon": "linear",
    "mass": "log",
    "energy": "log",
}

bins = {
    "calib_energy_ecal": np.logspace(-3, 2, 32),
    "calib_energy_hcal": np.logspace(-3, 2, 32),
    "mom.r": np.geomspace(0.01, 365, 32),
    "mom.eta": np.linspace(-3, 3, 32),
    "mom.phi": np.linspace(-np.pi, np.pi, 32),
    "vtx.r": np.linspace(0, 500, 32),
    "isolation": np.geomspace(1e-4, math.pi, 32),
    "num_vtxd": np.arange(-1, 12) + 0.5,
    "num_trkr": np.arange(-1, 12) + 0.5,
    "num_sihit": np.arange(-1, 24) + 0.5,
    "num_ecal": np.geomspace(1, 1000, 32),
    "num_hcal": np.geomspace(1, 500, 32),
    "num_muon": np.arange(-1, 14) + 0.5,
    "mass": np.logspace(-4, 3, 32),
    "energy": np.logspace(-3, 3, 32),
}

object_name = "particle"

selection_aliases = {
    "valid": "All",
    "is_charged": "Charged",
    "is_charged_hadron": "Charged Hadrons",
    "is_neutral_hadron": "Neutral Hadrons",
    "is_electron": "Electrons",
    "is_photon": "Photons",
    "is_muon": "Muons",
}

selection_colours = {
    "valid": "tab:blue",
    "is_charged": "tab:orange",
    "is_charged_hadron": "tab:green",
    "is_neutral_hadron": "tab:red",
    "is_electron": "tab:purple",
    "is_photon": "tab:brown",
    "is_muon": "tab:pink",
}


hists = {field: {selection: CountingHistogram(bins) for selection in selection_aliases} for field, bins in bins.items()}

# Setup the dataset
config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 10
config["batch_size"] = 10
config["num_test"] = 10000
config["test_dir"] = "/share/rcif2/maxhart/data/cld/test/prepped/"

datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")
dataloader = datamodule.test_dataloader()
data_iterator = iter(dataloader)

# Iterate through the dataset
for _i in tqdm(range(25)):
    inputs, targets = next(data_iterator)

    for field, selections in hists.items():
        for selection, hist in selections.items():
            values = targets[f"particle_{field}"][targets[f"particle_{selection}"].bool()]
            hist.fill(values)

# Define which plots we want here
# Each dict entry is a plot, they key is the plot name, and the value is a
# list which specifies which fields will be subplots in the plot
plots = {
    "particle_pt_eta_phi": ["mom.r", "mom.eta", "mom.phi"],
    "particle_mass_energy": [
        "mass",
        "energy",
    ],
    "particle_calo_energy": [
        "calib_energy_ecal",
        "calib_energy_hcal",
    ],
    "particle_iso_d0": [
        "vtx.r",
        "isolation",
    ],
    "particle_num_sihits": [
        "num_vtxd",
        "num_trkr",
        "num_sihit",
    ],
    "particle_hits": [
        "num_ecal",
        "num_hcal",
        "num_muon",
    ],
}

for plot_name, fields in plots.items():
    fig, ax = plt.subplots(1, len(fields))
    fig.set_size_inches(12, 3)

    for ax_idx, field in enumerate(fields):
        for selection, hist in hists[field].items():
            plot_hist_to_ax(
                ax[ax_idx],
                hist.counts,
                hist.bins,
                label=selection_aliases[selection],
                color=selection_colours[selection],
                vertical_lines=True,
            )

        ax[ax_idx].set_yscale("log")
        ax[ax_idx].set_xscale(scales[field])
        ax[ax_idx].set_xlabel(f"Particle {aliases[field]}")
        ax[ax_idx].set_ylabel("Count")
        ax[ax_idx].grid(zorder=0, alpha=0.25, linestyle="--")

    ax[0].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(Path(f"src/hepattn/experiments/cld/plots/data/{plot_name}.png"))
