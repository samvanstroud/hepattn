from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from hepattn.experiments.pixel.data import PixelClusterDataModule

plt.rcParams["figure.dpi"] = 300

config_path = Path("src/hepattn/experiments/pixel/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0
config["batch_size"] = 10000

datamodule = PixelClusterDataModule(**config)
datamodule.setup(stage="test")


dataloader = datamodule.test_dataloader()

inputs, targets = next(iter(dataloader))


print("Average query occupancy:", targets["particle_valid"].float().mean())

particle_masks = {
    "All": targets["particle_valid"],
    "Primary": targets["particle_primary"].to(torch.bool) & targets["particle_valid"],
    "Secondary": targets["particle_secondary"].to(torch.bool) & targets["particle_valid"],
    "No Truth": targets["particle_notruth"].to(torch.bool) & targets["particle_valid"],
    "Primary or Secondary": (targets["particle_primary"].to(torch.bool) | targets["particle_secondary"].to(torch.bool)) & targets["particle_valid"],
}

print(f"\nCluster multiplicities ({targets['cluster_valid'].float().sum()} clusters\n")

multiplicities, multiplicity_counts = np.unique(targets["cluster_multiplicity"], return_counts=True)

single_cluster_count = multiplicity_counts[0]

for multiplicity, multiplicity_count in zip(multiplicities, multiplicity_counts, strict=False):
    multiplicity_str = str(int(multiplicity)).ljust(4)
    count_str = str(int(multiplicity_count)).ljust(8)
    count_frac = multiplicity_count / len(targets["cluster_multiplicity"])
    pct_str = 100 * count_frac
    weight_str = single_cluster_count / multiplicity_count
    print(f"{multiplicity_str} | {count_str} | {pct_str:.2f}% | {weight_str:.2f}")


particle_class_name_to_label = {
    "hadron": 1,
    "photon": 2,
    "electron": 3,
    "muon": 4,
    "tau": 5,
    "other": 6,
}
particle_class_label_to_name = {v: k for k, v in particle_class_name_to_label.items()}


class_labels, class_counts = np.unique(targets["particle_class_label"][targets["particle_valid"]], return_counts=True)
num_particles = targets["particle_valid"].float().sum()

print(f"\nParticle classes ({num_particles} particles)\n")

for class_label, class_count in zip(class_labels, class_counts, strict=False):
    class_name = particle_class_label_to_name[class_label].ljust(12)
    count_str = str(int(class_count)).ljust(8)
    pct_str = 100 * class_count / targets["particle_valid"].sum()
    weight = num_particles / class_count
    print(f"{class_name} | {count_str} | {pct_str:.2f}% | {weight:.2f}")

print(f"\nParticle truth types ({num_particles} particles)\n")

# Plot the particle counts
fig, ax = plt.subplots()
fig.set_size_inches(8, 2)

for mask_name, particle_mask in particle_masks.items():
    pct = 100 * particle_mask.sum() / targets["particle_valid"].sum()
    num_type = particle_mask.sum().item()
    weight = num_particles / num_type
    print(f"{mask_name.ljust(24)}| {str(num_type).ljust(6)} | {pct:.2f}% | {weight:.2f}")

    cluster_num_particles = particle_mask.sum(-1)
    ax.hist(
        cluster_num_particles,
        bins=np.arange(0, 16) - 0.5,
        label=mask_name,
        histtype="step",
        density=False,
    )

ax.legend(fontsize=6)
ax.set_yscale("log")
ax.set_xticks(np.arange(0, 16))
ax.grid(alpha=0.25, linestyle="--")
ax.set_ylabel("Density")
ax.set_xlabel("Number of Particles of Given Origin on Cluster")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cluster_num_particles_hist.png")

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(8, 2)

ax[0].hist(inputs["cluster_width_x"], bins=32, histtype="step", color="cornflowerblue")
ax[1].hist(inputs["cluster_width_y"], bins=32, histtype="step", color="cornflowerblue")

ax[0].grid(alpha=0.25, linestyle="--")
ax[1].grid(alpha=0.25, linestyle="--")

ax[0].set_xlabel(r"Cluster Width $x$")
ax[1].set_xlabel(r"Cluster Width $y$")

ax[0].set_ylabel("Count")
ax[1].set_ylabel("Count")

ax[0].set_yscale("log")
ax[1].set_yscale("log")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cluster_width.png")


fields = ["global_r", "global_eta", "global_phi"]
field_aliases = {
    "global_r": r"Cluster Global $r$",
    "global_eta": r"Cluster Global $\eta$",
    "global_phi": r"Cluster Global $\phi$",
    "global_x": r"Cluster Global $x$",
    "global_y": r"Cluster Global $y$",
    "global_z": r"Cluster Global $z$",
}

fig, ax = plt.subplots(nrows=1, ncols=len(fields))
fig.set_size_inches(8, 2)

for i, field in enumerate(fields):
    ax[i].hist(inputs[f"cluster_{field}"], bins=32, histtype="step", color="cornflowerblue")
    ax[i].set_xlabel(field_aliases[field])
    ax[i].set_ylabel("Count")
    ax[i].grid(alpha=0.25, linestyle="--")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cluster_global_coords_angular.png")


fields = ["global_x", "global_y", "global_z"]

fig, ax = plt.subplots(nrows=1, ncols=len(fields))
fig.set_size_inches(8, 2)

for i, field in enumerate(fields):
    ax[i].hist(inputs[f"cluster_{field}"], bins=32, histtype="step", color="cornflowerblue")
    ax[i].set_xlabel(field_aliases[field])
    ax[i].set_ylabel("Count")
    ax[i].grid(alpha=0.25, linestyle="--")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cluster_global_coords_cartesian.png")


fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(8, 4)

mask = torch.abs(inputs["cluster_global_eta"]) >= 2.5

ax[0].scatter(inputs["cluster_global_x"][mask], inputs["cluster_global_y"][mask], alpha=0.5, s=1.0, color="cornflowerblue")
ax[1].scatter(inputs["cluster_global_z"][mask], inputs["cluster_global_y"][mask], alpha=0.5, s=1.0, color="cornflowerblue")

ax[0].set_xlabel(r"Cluster Global $x$")
ax[0].set_ylabel(r"Cluster Global $y$")

ax[1].set_xlabel(r"Cluster Global $z$")
ax[1].set_ylabel(r"Cluster Global $y$")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cluster_higheta.png")

# Plot the particle fields

fields = ["x", "y"]

field_bins = {
    "x": np.linspace(-8, 8, 24),
    "y": np.linspace(-4, 4, 24),
    "theta": np.linspace(-np.pi, np.pi, 24),
    "phi": np.linspace(-np.pi / 4, np.pi / 4, 24),
    "p": np.logspace(-1, 4, 24),
}

field_symbols = {
    "x": r"$x$",
    "y": r"$y$",
    "theta": r"$\theta$",
    "phi": r"$\phi$",
    "p": r"$p$ [GeV]",
}

log_fields = ["p"]

fields = ["x", "y"]

fig, ax = plt.subplots(nrows=1, ncols=len(fields))
fig.set_size_inches(8, 2)

for i, field in enumerate(fields):
    for mask_name, particle_mask in particle_masks.items():
        ax[i].hist(
            targets[f"particle_{field}"][particle_mask.to(torch.bool)],
            bins=field_bins[field],
            label=mask_name,
            histtype="step",
            density=True,
        )

        if field in log_fields:
            ax[i].set_xscale("log")

    ax[i].set_xlabel("Particle " + field_symbols[field])
    ax[i].set_yscale("log")
    ax[i].grid(alpha=0.25, linestyle="--")

ax[0].set_ylabel("Density")
ax[-1].legend(fontsize=6)

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cluster_particle_xy_hist.png")

fields = ["theta", "phi", "p"]

fig, ax = plt.subplots(nrows=1, ncols=len(fields))
fig.set_size_inches(8, 2)

for i, field in enumerate(fields):
    for mask_name, particle_mask in particle_masks.items():
        ax[i].hist(
            targets[f"particle_{field}"][particle_mask.to(torch.bool)],
            bins=field_bins[field],
            label=mask_name,
            histtype="step",
            density=True,
        )

        if field in log_fields:
            ax[i].set_xscale("log")

    ax[i].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax[i].tick_params(axis="y", which="both", bottom=False, top=False, labelbottom=False)
    ax[i].set_xlabel("Particle " + field_symbols[field])
    ax[i].set_yscale("log")
    ax[i].grid(True, alpha=0.25, linestyle="--")

ax[0].set_ylabel("Density")
ax[-1].legend(fontsize=6)

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cluster_particle_angles_hist.png")

# Plot the pixel charge

fig, ax = plt.subplots()
fig.set_size_inches(8, 2)

ax.hist(
    inputs["pixel_charge"][inputs["pixel_valid"]],
    bins=np.logspace(-2, 0, 64),
    histtype="step",
    color="cornflowerblue",
)

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_ylabel("Density")
ax.set_xlabel("Pixel Charge [ke / 100]")
ax.grid(alpha=0.25, linestyle="--")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/pixel_charge.png")

# Plot the pixel coordinates

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(8, 2)

ax[0].hist(
    inputs["pixel_x"][inputs["pixel_valid"]],
    bins=np.linspace(-5, 5, 11),
    histtype="step",
    color="cornflowerblue",
    density=True,
)

ax[1].hist(
    inputs["pixel_y"][inputs["pixel_valid"]],
    bins=np.linspace(-5, 5, 11),
    histtype="step",
    color="cornflowerblue",
    density=True,
)

ax[0].set_xlabel(r"Pixel $x$ Index Position")
ax[0].set_ylabel("Density")
ax[0].set_yscale("log")
ax[0].grid(alpha=0.25, linestyle="--")

ax[1].set_xlabel(r"Pixel $y$ Index Position")
ax[1].set_ylabel("Density")
ax[1].set_yscale("log")
ax[1].grid(alpha=0.25, linestyle="--")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/pixel_local_xy.png")

# Plot some examples

fig, ax = plt.subplots(6, 6)
fig.set_size_inches(8, 8)

ax = ax.flatten()

idx = torch.argsort(targets["particle_valid"].sum(-1), descending=True)
idx = torch.argsort(inputs["pixel_valid"].sum(-1), descending=True)
# idx = np.arange(len(targets["particle_valid"]))

pixel_charge = inputs["pixel_charge"][inputs["pixel_valid"]]
norm = colors.LogNorm(vmin=0.01, vmax=1.0)
cmap = plt.get_cmap("viridis")

for ax_idx in range(len(ax) - 1):
    i = idx[ax_idx]

    lx = torch.floor(torch.min(inputs["pixel_x"][i])) - 1.5
    ux = torch.floor(torch.max(inputs["pixel_x"][i])) + 1.5
    ly = torch.floor(torch.min(inputs["pixel_y"][i])) - 1.5
    uy = torch.floor(torch.max(inputs["pixel_y"][i])) + 1.5

    ax[ax_idx].set_xticks(np.linspace(lx, ux, int(abs(lx) + abs(ux) + 1)))
    ax[ax_idx].set_yticks(np.linspace(ly, uy, int(abs(ly) + abs(uy) + 1)))

    ax[ax_idx].grid(alpha=0.5)
    ax[ax_idx].grid(alpha=0.5)

    ax[ax_idx].set_xticklabels([])
    ax[ax_idx].set_yticklabels([])

    ax[ax_idx].imshow(np.zeros(shape=(10, 10)) * np.nan, extent=(lx, ux, ly, uy))

    for j in range(len(inputs["pixel_valid"][i])):
        size = 0.8
        x = inputs["pixel_x"][i][j]
        y = inputs["pixel_y"][i][j]
        charge = inputs["pixel_charge"][i][j]
        pixel_patch = Rectangle((x - size / 2, y - size / 2), size, size, color=cmap(norm(charge)))
        ax[ax_idx].add_patch(pixel_patch)

    particles_x = targets["particle_x"][i][targets["particle_valid"][i]]
    particles_y = targets["particle_y"][i][targets["particle_valid"][i]]

    particles_phi = targets["particle_phi"][i][targets["particle_valid"][i]]
    particles_theta = targets["particle_theta"][i][targets["particle_valid"][i]]

    particles_dx = particles_phi / torch.sqrt(particles_phi**2 + particles_theta**2)
    particles_dy = particles_theta / torch.sqrt(particles_phi**2 + particles_theta**2)

    num_particles = targets["particle_valid"][i].sum(-1)

    for j in range(num_particles):
        if targets["particle_primary"][i][j]:
            c = "crimson"
        elif targets["particle_secondary"][i][j]:
            c = "darkorange"
        elif targets["particle_notruth"][i][j]:
            c = "darkgray"

        ax[ax_idx].scatter(particles_x[j], particles_y[j], c=c, ec="black", s=32, linewidths=1)

        ax[ax_idx].arrow(
            particles_x[j],
            particles_y[j],
            particles_dx[j],
            particles_dy[j],
            width=0.01,
        )

ax[-1].axis("off")


custom_markers = [
    Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markeredgecolor="black", markersize=10)
    for label, color in [
        ("Primary", "crimson"),
        ("Secondary", "darkorange"),
        ("No Truth", "darkgray"),
    ]
]

ax[-1].legend(handles=custom_markers, fontsize=8)


sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

colorbar = fig.colorbar(sm, ax=ax[-1], location="bottom", aspect=5.0, panchor=(0.5, 0.5))
colorbar.set_label("Pixel Charge [ke / 100]", fontsize=8)

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/examples.png")
