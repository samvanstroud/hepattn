from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from hepattn.experiments.cld.data import CLDDataModule

plt.rcParams["figure.dpi"] = 300


config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0
config["batch_size"] = 25

datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")

dataloader = datamodule.train_dataloader()
data_iterator = iter(dataloader)

inputs, targets = next(data_iterator)

print("Read data from dataloader")


# Plot the particle fields

fields = [
    ("calib_energy_ecal", "Total Calibrated ECAL Energy [GeV]", "log", np.logspace(-3, 2, 32)),
    ("calib_energy_hcal", "Total Calibrated HCAL Energy [GeV]", "log", np.logspace(-3, 2, 32)),
]

selection_masks = {
    "All": targets["particle_valid"],
    "Charged": targets["particle_isCharged"],
    "Neutral": targets["particle_isNeutral"],
}

fig, ax = plt.subplots(1, len(fields))
fig.set_size_inches(12, 3)

for j, (field, alias, scale, bins) in enumerate(fields):
    for selection_name, selection_mask in selection_masks.items():
        ax[j].hist(targets[f"particle_{field}"][selection_mask.bool()], bins=bins, histtype="step", label=selection_name)
        ax[j].set_yscale("log")
        ax[j].set_xscale(scale)
        ax[j].set_xlabel(f"Particle {alias}")
        ax[j].set_ylabel("Count")
        ax[j].grid(zorder=0, alpha=0.25, linestyle="--")
        ax[j].legend(fontsize=8)

fig.tight_layout()
fig.savefig(Path("src/hepattn/experiments/cld/plots/data/cld_particle_calo_energy.png"))


fields = [
    ("mom.eta", r"$\eta$", "linear", np.linspace(-4, 4, 32)),
    ("mom.r", r"$p_T$", "log", np.logspace(-2, 2, 32)),
]

selection_masks = {
    "All": targets["particle_valid"],
    "Charged": targets["particle_isCharged"],
    "Neutral": targets["particle_isNeutral"],
}

fig, ax = plt.subplots(1, len(fields))
fig.set_size_inches(12, 3)

for j, (field, alias, scale, bins) in enumerate(fields):
    for selection_name, selection_mask in selection_masks.items():
        ax[j].hist(targets[f"particle_{field}"][selection_mask.bool()], bins=bins, histtype="step", label=selection_name)
        ax[j].set_yscale("log")
        ax[j].set_xscale(scale)
        ax[j].set_xlabel(f"Particle {alias}")
        ax[j].set_ylabel("Count")
        ax[j].grid(zorder=0, alpha=0.25, linestyle="--")
        ax[j].legend(fontsize=8)

fig.tight_layout()
fig.savefig(Path("src/hepattn/experiments/cld/plots/data/cld_particle_pt_eta.png"))


fields = [
    ("num_vtxd", "Num. VTXD Hits", "linear", np.arange(-1, 12) + 0.5),
    ("num_trkr", "Num. Tracker Hits", "linear", np.arange(-1, 12) + 0.5),
    ("num_sihit", "Num. VTXD + Tracker Hits", "linear", np.arange(-1, 20) + 0.5),
]

selection_masks = {
    "All": targets["particle_valid"],
    "Charged": targets["particle_isCharged"],
    "Neutral": targets["particle_isNeutral"],
}

fig, ax = plt.subplots(1, len(fields))
fig.set_size_inches(12, 3)

for j, (field, alias, scale, bins) in enumerate(fields):
    for selection_name, selection_mask in selection_masks.items():
        ax[j].hist(targets[f"particle_{field}"][selection_mask.bool()], bins=bins, histtype="step", label=selection_name)
        ax[j].set_yscale("log")
        ax[j].set_xscale(scale)
        ax[j].set_xlabel(f"Particle {alias}")
        ax[j].set_ylabel("Count")
        ax[j].grid(zorder=0, alpha=0.25, linestyle="--")
        ax[j].legend(fontsize=8)

fig.tight_layout()
fig.savefig(Path("src/hepattn/experiments/cld/plots/data/cld_particle_sihit_counts.png"))


fields = [
    ("num_ecal", "Num. ECAL Hits", "log", np.geomspace(1, 1000, 32)),
    ("num_hcal", "Num. HCAL Hits", "log", np.geomspace(1, 400, 32)),
]

selection_masks = {
    "All": targets["particle_valid"],
    "Charged": targets["particle_isCharged"],
    "Neutral": targets["particle_isNeutral"],
}

fig, ax = plt.subplots(1, len(fields))
fig.set_size_inches(12, 3)

for j, (field, alias, scale, bins) in enumerate(fields):
    for selection_name, selection_mask in selection_masks.items():
        ax[j].hist(targets[f"particle_{field}"][selection_mask.bool()], bins=bins, histtype="step", label=selection_name)
        ax[j].set_yscale("log")
        ax[j].set_xscale(scale)
        ax[j].set_xlabel(f"Particle {alias}")
        ax[j].set_ylabel("Count")
        ax[j].grid(zorder=0, alpha=0.25, linestyle="--")
        ax[j].legend(fontsize=8)

fig.tight_layout()
fig.savefig(Path("src/hepattn/experiments/cld/plots/data/cld_particle_calohit_counts.png"))
