import math
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.models.matcher import Matcher
from hepattn.utils.eval import apply_matching, calc_binary_reco_metrics, calc_cost, calculate_selections
from hepattn.utils.histogram import PoissonHistogram

object_names = ["particle", "pandora", "flow"]
truth_object_name = "particle"

tracker_hit_names = ["sihit"]
calo_hit_names = ["ecal", "hcal"]
muon_hit_names = ["muon"]

hit_names = tracker_hit_names + calo_hit_names + muon_hit_names

# Define all the particle classes
particle_types = [
    "charged",
    "charged_hadron",
    "neutral_hadron",
    "electron",
    "photon",
    "muon",
]

# Define which detector subsystems each particle type interacts with
particle_type_hits = {
    "charged": ["sihit"],
    "charged_hadron": ["sihit", "ecal", "hcal"],
    "neutral_hadron": ["ecal", "hcal"],
    "electron": ["ecal"],
    "photon": ["sihit", "ecal"],
    "muon": ["sihit", "ecal", "hcal", "muon"],
}

matching_metric = {hit_name: {"weight": 1.0, "metric": "iou", "field": "valid"} for hit_name in hit_names}

binary_metrics_config = {}
working_points = {"dm": 0.5, "lhc": 0.75, "tight": 0.95}

for particle_type, hits in particle_type_hits.items():
    for wp_name, wp_thresh in working_points.items():
        binary_metric = []
        for hit in hits:
            binary_metric.extend({"hit": hit, "thresh": wp_thresh, "metric": metric, "field": "valid"} for metric in ["eff", "pur"])
        binary_metrics_config[f"{particle_type}_{wp_name}"] = binary_metric

# Define some flags which tell us whether the object left a signature in each sub detetector
selections = {
    "has_sihit": [
        "num_sihit >= 4",
    ],
    "has_ecal": [
        "num_ecal >= 10",
        "energy_ecal >= 0.1",
    ],
    "has_hcal": [
        "num_hcal >= 10",
        "energy_hcal >= 0.1",
    ],
    "has_muon": [
        "num_muon >= 4",
    ],
}

# If the particle left a signature in all the subdetetcors it should interact with,
# we can mark it as likely being that particle type
for particle_type, hits in particle_type_hits.items():
    selections[f"sig_{particle_type}"] = [f"has_{hit}" for hit in hits]

particle_selections = {
    "tight": [
        "is_primary",
        "mom.r >= 0.1",
        "mom.abs_eta <= 2.44",
        "vtx.r <= 50",
        "isolation >= 0.02",
    ],
    "loose": ["mom.r >= 0.01"],
}

# Define whether each particle type is reconstructable
for particle_type in particle_types:
    particle_selections[f"{particle_type}_loose"] = [f"is_{particle_type}", "loose"]
    particle_selections[f"{particle_type}_tight"] = [f"is_{particle_type}", "tight"]


field_aliases = {
    "mom.r": r"$p_T$",
    "mom.eta": r"$\eta$",
    "mom.phi": r"$\phi$",
    "vtx.r": r"Vertex $r$",
    "isolation": r"Angular Isolation",
    "num_vtxd": "Num. VTXD Hits",
    "num_trkr": "Num. Tracker Hits",
    "num_sihit": "Num. Si Hits",
    "num_ecal": "Num. ECAL Hits",
    "num_hcal": "Num. HCAL Hits",
    "num_muon": "Num. Muon Hits",
}

field_scales = {
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
}

field_bins = {
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
}

binary_histogram_config = {}

for object_name in object_names:
    for particle_type in particle_types:
        for selection in ["tight", "loose"]:
            selection_name = f"particle_{particle_type}_{selection}"
            for wp_name in working_points:
                binary_metric_name = f"particle_{object_name}_{particle_type}_{wp_name}"

                for field in field_aliases:
                    binary_histogram_config[f"{object_name}_{particle_type}_{selection}_{wp_name}_{field}"] = {
                        "field": f"particle_{field}",
                        "bins": field,
                        "selection": selection_name,
                        "numerator": binary_metric_name,
                        "denominator": selection_name,
                    }


binary_histograms = {
    name: PoissonHistogram(
        cfg["field"],
        field_bins[cfg["bins"]],
        cfg["selection"],
        cfg["numerator"],
        cfg["denominator"],
    )
    for name, cfg in binary_histogram_config.items()
}

print("\nBinary histogram spec:\n")
for name, hist in binary_histograms.items():
    print(name, hist.field, hist.selection, hist.numerator, hist.denominator)


residual_metrics = {}


residual_histogram_config = {}
for particle_type in particle_types:
    for selection in ["tight", "loose"]:
        selection_name = f"particle_{particle_type}_{selection}"
        for hit in ["ecal", "hcal"]:
            # Skip if the particle type doesnt involve the ecal/hcal
            if hit not in particle_type_hits[particle_type]:
                continue


# Setup the dataset to load truth and pandora info from
config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]

# Need to overwrite whatever is in the config
config["num_workers"] = 0
config["batch_size"] = 1
config["num_test"] = -1  # Load the entire test set, so that we can access any sample id

# Get the dataset object
datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")
dataset = datamodule.test_dataloader().dataset


eval_file_path = Path(
    "/share/rcifdata/maxhart/hepattn/logs/CLD_8_320_10MeV_neutrals_muon_20250809-T183715/ckpts/epoch=007-train_loss=2.60941_test_eval_old.h5"
)


matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)


with h5py.File(eval_file_path, "r") as eval_file:
    sample_ids = list(eval_file.keys())


with h5py.File(eval_file_path, "r") as eval_file:
    for sample_id in sample_ids:
        data = {}

        # Get the predictions just from the final layer
        final_preds = eval_file[f"{sample_id}/preds/final/"]
        final_outputs = eval_file[f"{sample_id}/outputs/final/"]

        # Load whether each slot was predicted as valid or not
        data["flow_valid"] = torch.from_numpy(final_preds["flow_valid/flow_valid"][:])
        data["flow_logit"] = torch.from_numpy(final_outputs["flow_valid/flow_logit"][:])

        data["flow_valid"] = data["flow_logit"].sigmoid() >= 0.5

        for hit in ["vtxd", "trkr", "ecal", "hcal", "muon"]:
            # Make sure to drop any invalid hit slots from the mask
            data[f"flow_{hit}_valid"] = torch.from_numpy(final_preds[f"flow_{hit}_assignment/flow_{hit}_valid"][:])

        # Get the sample
        sample = dataset.load_sample(int(sample_id))

        # Prepare it by adding any necessary padding and converting to tensorsl,
        inputs, targets = dataset.prep_sample(sample)

        # Add the input and target data from the dataset
        data |= targets
        data |= inputs

        # Add extra information to the objects
        for object_name in object_names:
            # Make sihit collection which merges vtxd and tracker
            data[f"{object_name}_sihit_valid"] = torch.cat((data[f"{object_name}_vtxd_valid"], data[f"{object_name}_trkr_valid"]), -1)

            # Add hit count info to objects
            for hit_name in hit_names:
                if f"{object_name}_{hit_name}_valid" in data:
                    data[f"{object_name}_num_{hit_name}"] = data[f"{object_name}_{hit_name}_valid"].sum(-1)

            # Calculate the total calo hit energy assigned to the object
            for hit_name in calo_hit_names:
                if f"{object_name}_{hit_name}_valid" in data:
                    object_hit_energy = data[f"{object_name}_{hit_name}_valid"].float() * data[f"{hit_name}_energy"].unsqueeze(-2)
                    data[f"{object_name}_{hit_name}_energy"] = object_hit_energy
                    data[f"{object_name}_energy_{hit_name}"] = object_hit_energy.sum(-1)

        for object_name in object_names:
            # Calculate the costs for the matching
            costs = calc_cost(data, truth_object_name, object_name, {k: v for k, v in matching_metric.items() if f"{object_name}_{k}_valid" in data})

            # Permute the obejcts to minimise the cost
            data = apply_matching(data, truth_object_name, object_name, costs, matcher)

            eff_metrics = calc_binary_reco_metrics(data, "particle", object_name, binary_metrics_config)
            pur_metrics = calc_binary_reco_metrics(data, object_name, "particle", binary_metrics_config)

            data |= eff_metrics
            data |= pur_metrics

            # Calculate selections that can be done on all objects
            data = calculate_selections(data, object_name, selections)

        # Calculate selections that can be done on particles only
        data = calculate_selections(data, truth_object_name, particle_selections)

        for histogram in binary_histograms.values():
            histogram.fill(data)
