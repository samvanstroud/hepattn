from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from hepattn.experiments.cld.data_new import CLDDataModule

from hepattn.experiments.cld.plot_event import plot_cld_event_reconstruction
from hepattn.models.loss import cost_fns

plt.rcParams["figure.dpi"] = 300


config_path = Path("src/hepattn/experiments/cld/configs/base_new.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0
config["batch_size"] = 5

datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")

dataset = datamodule.test_dataloader().dataset
inputs, targets = dataset[0]

print("Read data from dataloader")

for k, v in inputs.items():
    print(k, v.shape, v.dtype)

for k, v in targets.items():
    print(k, v.shape, v.dtype)

dataloader = datamodule.test_dataloader()
data_iterator = iter(dataloader)

inputs, targets = next(data_iterator)

print("Read data from dataloader")

for k, v in inputs.items():
    print(k, v.shape, v.dtype)

for k, v in targets.items():
    print(k, v.shape, v.dtype)


targets["particle_invalid"] = ~targets["particle_valid"]

hit_cost_weights = {
    "trkr": {
        # "mask_bce": 10.0,
        # "mask_coefficient": 1.0,
        "mask_dice": 1.0,
    },
    "vtxd": {
        # "mask_bce": 10.0,
        # "mask_coefficient": 1.0,
        "mask_dice": 1.0,
    },
    "ecal": {
        # "mask_bce": 0.1,
        # "mask_coefficient": 0.1,
        "mask_dice": 1.0,
    },
    "hcal": {
        # "mask_bce": 0.5,
        # "mask_coefficient": 0.5,
        "mask_dice": 1.0,
    },
}

axes_spec = [
    {
        "x": "pos.x",
        "y": "pos.y",
        "px": "mom.x",
        "py": "mom.y",
        "input_names": [
            "vtxd",
            "trkr",
            "ecal",
            "hcal",
            "muon",
        ],
    },
    {
        "x": "pos.z",
        "y": "pos.y",
        "px": "mom.z",
        "py": "mom.y",
        "input_names": [
            "vtxd",
            "trkr",
            "ecal",
            "hcal",
            "muon",
        ],
    },
]

fig = plot_cld_event_reconstruction(inputs, targets, axes_spec, "particle")
fig.savefig("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/true.png")

fig = plot_cld_event_reconstruction(inputs, targets, axes_spec, "pandora")
fig.savefig("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/reco.png")

batch_size = targets["particle_valid"].shape[0]
num_objects = targets["particle_valid"].shape[1]

target = targets["particle_valid"].float()

flow = targets["pandora_valid"].float()
costs = 1.0 * cost_fns["object_bce"](target, flow)

for hit_name, cost_weights in hit_cost_weights.items():
    for cost_name, cost_weight in cost_weights.items():
        target = targets[f"particle_{hit_name}_valid"].float()
        flow = targets[f"pandora_{hit_name}_valid"].float()
        logits = 100 * target - 100 * (1 - target)

        weight = None
        costs += cost_weight * cost_fns[cost_name](logits, flow, input_pad_mask=targets[f"{hit_name}_valid"], sample_weight=weight)

fig, axes = plt.subplots(3, 1)
fig.set_size_inches(12, 6)

valid = targets["particle_valid"]

valid_masks = {
    "Both Valid": valid.unsqueeze(1) & valid.unsqueeze(2),
    "Valid-Invalid": (valid.unsqueeze(1) & (~valid.unsqueeze(2))) | ((~valid.unsqueeze(1)) & valid.unsqueeze(2)),
    "Both Invalid": (~valid.unsqueeze(1)) & (~valid.unsqueeze(2)),
}

same = torch.eye(num_objects).expand(batch_size, num_objects, num_objects).bool()

same_masks = {
    "Same": same,
    "Different": ~same,
}

for same_mask_name, same_mask in same_masks.items():
    for valid_mask_name, valid_mask in valid_masks.items():
        mask = same_mask & valid_mask
        bins = np.linspace(torch.min(costs), torch.max(costs), 100)
        axes[0].hist(costs[mask], bins=bins, histtype="step", label=f"{same_mask_name}, {valid_mask_name}")

        if valid_mask_name == "Both Valid":
            valid_costs = costs[valid_mask]
            bins = np.linspace(torch.min(valid_costs), torch.max(valid_costs), 100)
            axes[1].hist(costs[mask], bins=bins, histtype="step", label=f"{same_mask_name}, {valid_mask_name}")

            if same_mask_name == "Same":
                axes[2].hist(costs[mask], bins=100, histtype="step", label=f"{same_mask_name}, {valid_mask_name}")

for ax in axes:
    ax.set_yscale("log")
    ax.set_ylabel("Count")
    ax.set_xlabel("Cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--")

fig.tight_layout()
fig.savefig(Path("src/hepattn/experiments/cld/plots/data/cld_particle_costs.png"))
