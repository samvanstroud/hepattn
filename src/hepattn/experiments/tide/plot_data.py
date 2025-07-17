from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from hepattn.experiments.tide.data import ROIDataModule

plt.rcParams["figure.dpi"] = 300

torch.manual_seed(42)

config_path = Path("src/hepattn/experiments/tide/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0
config["batch_size"] = 250
config["num_test"] = 10000

datamodule = ROIDataModule(**config)
datamodule.setup(stage="test")


dataloader = datamodule.test_dataloader()
data_iterator = iter(dataloader)

output_dir = Path("src/hepattn/experiments/tide/plots/data")

inputs, targets = next(data_iterator)

for k, v in targets.items():
    print(k, v.shape)

print("ROI has a track with this origin:")
for origin_class in ["b", "c", "tau", "other"]:
    print(origin_class, round(100 * torch.mean(targets[f"sudo_from_{origin_class}"][targets["sudo_valid"]].float()).item(), 3))

print("ROI labelled with this origin:")
for origin_class in ["b", "c", "tau", "other"]:
    frac = torch.mean(targets[f"roi_is_{origin_class}"].float()).item()
    print(f"{origin_class}, {frac:3f}, {(1 / frac):.3f}")

# Plot the global hit cluster coordinates

hits = ["pix", "sct"]
hit_aliases = {"pix": "Pixel", "sct": "SCT"}
field_aliases = {
    "x": r"$x$",
    "y": r"$y$",
    "z": r"$z$",
    "r": r"$r$",
    "theta": r"$\theta$",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "dtheta": r"$\Delta \theta$",
    "deta": r"$\Delta \eta$",
    "dphi": r"$\Delta \phi$",
}

fields = ["x", "y", "z"]

fig, ax = plt.subplots(len(hits), len(fields))
fig.set_size_inches(12, 4)

for i, hit in enumerate(hits):
    for j, field in enumerate(fields):
        ax[i, j].hist(inputs[f"{hit}_{field}"][inputs[f"{hit}_valid"]], bins=32, histtype="step")
        ax[i, j].set_xlabel(rf"{hit_aliases[hit]} {field_aliases[field]}")
        ax[i, j].set_ylabel("Count")
        ax[i, j].set_yscale("log")

fig.tight_layout()
fig.savefig(output_dir / "tide_hit_global_coords_cartesian.png")

fields = ["r", "theta", "eta", "phi"]

fig, ax = plt.subplots(len(hits), len(fields))
fig.set_size_inches(12, 4)

for i, hit in enumerate(hits):
    for j, field in enumerate(fields):
        ax[i, j].hist(inputs[f"{hit}_{field}"][inputs[f"{hit}_valid"]], bins=32, histtype="step")
        ax[i, j].set_xlabel(rf"{hit_aliases[hit]} {field_aliases[field]}")
        ax[i, j].set_ylabel("Count")
        ax[i, j].set_yscale("log")

fig.tight_layout()
fig.savefig(output_dir / "tide_hit_global_coords_cylindrical.png")
