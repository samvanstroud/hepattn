from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
import yaml

from hepattn.experiments.odd.data import ODDDataset
from hepattn.experiments.odd.event_display import plot_odd_event


def _load_config():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return yaml.safe_load(config_path.read_text())["data"]


def _build_dataset_kwargs(config):
    return {
        "dirpath": config.get("test_dir", config["train_dir"]),
        "num_events": config.get("num_test", -1),
        "particle_min_pt": config["particle_min_pt"],
        "particle_max_abs_eta": config["particle_max_abs_eta"],
        "particle_hit_cuts": config.get("particle_hit_cuts"),
        "particle_include_charged": config["particle_include_charged"],
        "particle_include_neutral": config["particle_include_neutral"],
        "event_type": config.get("event_type", "ttbar"),
        "debug": config.get("plot_debug", False),
    }


def _load_first_event(dataset):
    for idx, sample_id in enumerate(dataset.sample_ids):
        print(f"Trying sample_id={sample_id}", flush=True)
        inputs, targets = dataset[idx]
        print(f"Loaded sample_id={sample_id}")
        return inputs | targets

    msg = "No valid events were found with the current plotting config."
    raise RuntimeError(msg)


def _format_nbytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KiB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MiB"
    return f"{num_bytes / (1024**3):.2f} GiB"


def _print_tensor_summary(tensors: dict[str, torch.Tensor]) -> None:
    print(f"Event tensor summary ({len(tensors)} tensors):", flush=True)
    total_nbytes = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        nbytes = int(tensor.numel() * tensor.element_size())
        total_nbytes += nbytes
        print(
            f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} size={_format_nbytes(nbytes)}",
            flush=True,
        )
    print(f"Event tensor total size={_format_nbytes(total_nbytes)}", flush=True)


def _run_plot_round(
    event_data: dict[str, torch.Tensor],
    plot_save_dir: Path,
    plot_configs: list[tuple[str, str, dict[str, object]]],
    top_n_particles_by_pt: int | None = None,
) -> None:
    plot_save_dir.mkdir(exist_ok=True, parents=True)
    round_label = "all particles" if top_n_particles_by_pt is None else f"top {top_n_particles_by_pt} particles by pt"
    print(f"Saving {round_label} plots to: {plot_save_dir}", flush=True)

    for filename, label, kwargs in plot_configs:
        t0 = perf_counter()
        print(f"{label} [{round_label}]...", flush=True)
        fig = plot_odd_event(
            event_data,
            top_n_particles_by_pt=top_n_particles_by_pt,
            **kwargs,
        )
        fig.savefig(plot_save_dir / filename)
        plt.close(fig)
        print(f"Saved {filename} in {perf_counter() - t0:.2f}s", flush=True)


config = _load_config()
dataset_kwargs = _build_dataset_kwargs(config)

print("Loading event (sihits + calohits + tracks)...")
event_dataset = ODDDataset(**dataset_kwargs, return_calohits=True, return_tracks=True)
event_data = _load_first_event(event_dataset)
_print_tensor_summary(event_data)

plot_base_dir = Path(__file__).resolve().parents[1] / "plots" / "event_displays"
plot_all_particles_dir = plot_base_dir / "all_particles"
top_n_particles_by_pt = 100
plot_top_particles_dir = plot_base_dir / f"top{top_n_particles_by_pt}_particles"
plot_all_particles_round = bool(config.get("plot_all_particles_round", False))
plot_top_particles_round = bool(config.get("plot_top_particles_round", True))

particle_plot_configs = [
    (
        "particles.png",
        "Plotting particles (full detector)",
        {
            "plot_sihits": True,
            "plot_particle_sihits": True,
            "plot_calohits": True,
            "plot_particle_calohits": True,
        },
    ),
    (
        "calohits_by_detector.png",
        "Plotting calohits by detector ID",
        {
            "plot_calohits_by_detector": True,
        },
    ),
    (
        "particles_tracker.png",
        "Plotting particles (tracker only)",
        {
            "plot_sihits": True,
            "plot_particle_sihits": True,
        },
    ),
    (
        "particles_helix.png",
        "Plotting particles helices (full detector)",
        {
            "plot_sihits": True,
            "plot_calohits": True,
            "plot_particle_calohits": True,
            "plot_particles": True,
        },
    ),
    (
        "particles_tracker_helix.png",
        "Plotting particles helices (tracker only)",
        {
            "plot_sihits": True,
            "plot_particles": True,
        },
    ),
    (
        "particles_tracker_hits_and_helix.png",
        "Plotting particle tracker hits + helices",
        {
            "plot_particle_sihits": True,
            "plot_particles": True,
            "particle_helix_linestyle": "--",
        },
    ),
]

track_plot_configs = [
    (
        "tracks_tracker.png",
        "Plotting tracks (tracker only)",
        {
            "plot_sihits": True,
            "plot_track_sihits": True,
        },
    ),
    (
        "tracks_tracker_helix.png",
        "Plotting tracks helices (tracker only)",
        {
            "plot_sihits": True,
            "plot_tracker": True,
        },
    ),
]

if plot_all_particles_round:
    _run_plot_round(
        event_data,
        plot_all_particles_dir,
        particle_plot_configs + track_plot_configs,
    )
else:
    print("Skipping all-particles plot round (plot_all_particles_round=False)", flush=True)

if plot_top_particles_round:
    _run_plot_round(
        event_data,
        plot_top_particles_dir,
        particle_plot_configs,
        top_n_particles_by_pt=top_n_particles_by_pt,
    )
else:
    print("Skipping top-particles plot round (plot_top_particles_round=False)", flush=True)
