from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from hepattn.experiments.odd.data import ODDDataset
from hepattn.utils.histogram import CountingHistogram
from hepattn.utils.plotting import plot_hist_to_ax

plt.rcParams["figure.dpi"] = 300


aliases = {
    "pt": r"$p_T$ [GeV]",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "d0": r"$d_0$",
    "z0": r"$z_0$",
    "num_sihits": "Num. Si Hits",
    "num_calohits": "Num. Calo Hits",
}

scales = {
    "pt": "log",
    "eta": "linear",
    "phi": "linear",
    "d0": "linear",
    "z0": "linear",
    "num_sihits": "linear",
    "num_calohits": "linear",
}

bins = {
    "pt": np.geomspace(0.01, 500.0, 40),
    "eta": np.linspace(-6.0, 6.0, 40),
    "phi": np.linspace(-np.pi, np.pi, 40),
    "d0": np.linspace(-2.0, 2.0, 40),
    "z0": np.linspace(-3.0, 3.0, 40),
    "num_sihits": np.linspace(0.0, 800.0, 41),
    "num_calohits": np.linspace(0.0, 5000.0, 41),
}

selection_aliases = {
    "valid": "All",
    "charged": "Charged",
    "neutral": "Neutral",
}

selection_colours = {
    "valid": "tab:blue",
    "charged": "tab:orange",
    "neutral": "tab:green",
}

hists = {field: {selection: CountingHistogram(field_bins) for selection in selection_aliases} for field, field_bins in bins.items()}


def _load_config():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return yaml.safe_load(config_path.read_text())["data"]


def _build_dataset_kwargs(config):
    dataset_kwargs = {
        "dirpath": config.get("test_dir", config["train_dir"]),
        "num_events": 10,
        "particle_min_pt": config["particle_min_pt"],
        "particle_max_abs_eta": config["particle_max_abs_eta"],
        "particle_include_charged": config["particle_include_charged"],
        "particle_include_neutral": config["particle_include_neutral"],
        "event_type": config.get("event_type", "ttbar"),
        "debug": False,
        "return_calohits": True,
        "return_tracks": False,
    }

    if "particle_min_num_sihits" in config:
        dataset_kwargs["particle_min_num_sihits"] = config["particle_min_num_sihits"]

    return dataset_kwargs


def _to_numpy(x):
    return x.detach().cpu().numpy()


config = _load_config()
dataset = ODDDataset(**_build_dataset_kwargs(config))
num_events = min(10, len(dataset))

for event_idx in tqdm(range(num_events)):
    _, targets = dataset[event_idx]
    particle_valid = targets["particle_valid"][0].bool()
    particle_charged = targets["particle_charged"][0].bool()
    particle_neutral = targets["particle_neutral"][0].bool()

    particle_values = {
        "pt": targets["particle_pt"][0],
        "eta": targets["particle_eta"][0],
        "phi": targets["particle_phi"][0],
        "d0": targets["particle_d0"][0],
        "z0": targets["particle_z0"][0],
        "num_sihits": targets["particle_num_sihits"][0],
        "num_calohits": np.diff(_to_numpy(targets["particle_calohit_indptr"][0])),
    }

    selection_masks = {
        "valid": particle_valid,
        "charged": particle_valid & particle_charged,
        "neutral": particle_valid & particle_neutral,
    }

    for field, selection_hists in hists.items():
        values = particle_values[field]
        if isinstance(values, np.ndarray):
            values_np = values
        else:
            values_np = _to_numpy(values)

        for selection, hist in selection_hists.items():
            sel = _to_numpy(selection_masks[selection])
            hist.fill(values_np[sel])


plots = {
    "odd_particle_kinematics": ["pt", "eta", "phi"],
    "odd_particle_impact_params": ["d0", "z0"],
    "odd_particle_hit_multiplicity": ["num_sihits", "num_calohits"],
}

plot_dir = Path(__file__).resolve().parents[1] / "plots" / "data"
plot_dir.mkdir(parents=True, exist_ok=True)

for plot_name, fields in plots.items():
    fig, ax = plt.subplots(1, len(fields))
    fig.set_size_inches(4 * len(fields), 3)
    ax = [ax] if len(fields) == 1 else ax

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
    fig.savefig(plot_dir / f"{plot_name}.png")
    plt.close(fig)
