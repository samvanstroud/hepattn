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
    "x": "Position $x$ [m]",
    "y": "Position $y$ [m]",
    "z": "Position $z$ [m]",
}

scales = {
    "x": "linear",
    "y": "linear",
    "z": "linear",
}

bins = {
    "x": np.linspace(-4.0, 4.0, 40),
    "y": np.linspace(-4.0, 4.0, 40),
    "z": np.linspace(-8.0, 8.0, 40),
}

hit_aliases = {
    "sihit": "Si Hits",
    "calohit": "Calo Hits",
}

hit_colours = {
    "sihit": "tab:blue",
    "calohit": "tab:orange",
}

hists = {field: {hit: CountingHistogram(field_bins) for hit in hit_aliases} for field, field_bins in bins.items()}


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
    inputs, _ = dataset[event_idx]

    for field, hit_hists in hists.items():
        for hit, hist in hit_hists.items():
            values = inputs[f"{hit}_{field}"][0]
            valid = inputs[f"{hit}_valid"][0].bool()
            hist.fill(_to_numpy(values[valid]))


plots = {
    "odd_hit_xyz": ["x", "y", "z"],
}

plot_dir = Path(__file__).resolve().parents[1] / "plots" / "data"
plot_dir.mkdir(parents=True, exist_ok=True)

for plot_name, fields in plots.items():
    fig, ax = plt.subplots(1, len(fields))
    fig.set_size_inches(4 * len(fields), 3)
    ax = [ax] if len(fields) == 1 else ax

    for ax_idx, field in enumerate(fields):
        for hit, hist in hists[field].items():
            plot_hist_to_ax(
                ax[ax_idx],
                hist.counts,
                hist.bins,
                label=hit_aliases[hit],
                color=hit_colours[hit],
                vertical_lines=True,
            )

        ax[ax_idx].set_yscale("log")
        ax[ax_idx].set_xscale(scales[field])
        ax[ax_idx].set_xlabel(aliases[field])
        ax[ax_idx].set_ylabel("Count")
        ax[ax_idx].grid(zorder=0, alpha=0.25, linestyle="--")

    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{plot_name}.png")
    plt.close(fig)
