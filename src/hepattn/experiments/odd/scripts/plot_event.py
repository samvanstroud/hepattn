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
    dataset_kwargs = {
        "dirpath": config.get("test_dir", config["train_dir"]),
        "num_events": config.get("num_test", -1),
        "particle_min_pt": config["particle_min_pt"],
        "particle_max_abs_eta": config["particle_max_abs_eta"],
        "particle_include_charged": config["particle_include_charged"],
        "particle_include_neutral": config["particle_include_neutral"],
        "event_type": config.get("event_type", "ttbar"),
        "debug": config.get("plot_debug", False),
    }

    if "particle_min_num_sihits" in config:
        dataset_kwargs["particle_min_num_sihits"] = config["particle_min_num_sihits"]
    if "particle_min_num_calohits" in config:
        dataset_kwargs["particle_min_num_calohits"] = config["particle_min_num_calohits"]

    return dataset_kwargs


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


config = _load_config()
dataset_kwargs = _build_dataset_kwargs(config)

print("Loading event (sihits + calohits + tracks)...")
event_dataset = ODDDataset(**dataset_kwargs, return_calohits=True, return_tracks=True)
event_data = _load_first_event(event_dataset)
_print_tensor_summary(event_data)

# Define plotting config
plot_save_dir = Path(__file__).resolve().parents[1] / "plots" / "event_displays"
plot_save_dir.mkdir(exist_ok=True, parents=True)

# Plot full detector for particles
t0 = perf_counter()
print("Plotting particles (full detector)...", flush=True)
fig = plot_odd_event(
    event_data,
    plot_sihits=True,
    plot_particle_sihits=True,
    plot_calohits=True,
    plot_particle_calohits=True,
)
fig.savefig(plot_save_dir / Path("particles.png"))
plt.close(fig)
print(f"Saved particles.png in {perf_counter() - t0:.2f}s", flush=True)

# Plot full detector calohits coloured by detector ID
t0 = perf_counter()
print("Plotting calohits by detector ID...", flush=True)
fig = plot_odd_event(
    event_data,
    plot_calohits_by_detector=True,
)
fig.savefig(plot_save_dir / Path("calohits_by_detector.png"))
plt.close(fig)
print(f"Saved calohits_by_detector.png in {perf_counter() - t0:.2f}s", flush=True)

# Plot tracker for particles
t0 = perf_counter()
print("Plotting particles (tracker only)...", flush=True)
fig = plot_odd_event(
    event_data,
    plot_sihits=True,
    plot_particle_sihits=True,
)
fig.savefig(plot_save_dir / Path("particles_tracker.png"))
plt.close(fig)
print(f"Saved particles_tracker.png in {perf_counter() - t0:.2f}s", flush=True)

# Plot tracker for tracks
t0 = perf_counter()
print("Plotting tracks (tracker only)...", flush=True)
fig = plot_odd_event(
    event_data,
    plot_sihits=True,
    plot_track_sihits=True,
)
fig.savefig(plot_save_dir / Path("tracks_tracker.png"))
plt.close(fig)
print(f"Saved tracks_tracker.png in {perf_counter() - t0:.2f}s", flush=True)

# Plot full detector for particles using helix overlays
t0 = perf_counter()
print("Plotting particles helices (full detector)...", flush=True)
fig = plot_odd_event(
    event_data,
    plot_sihits=True,
    plot_calohits=True,
    plot_particle_calohits=True,
    plot_particles=True,
)
fig.savefig(plot_save_dir / Path("particles_helix.png"))
plt.close(fig)
print(f"Saved particles_helix.png in {perf_counter() - t0:.2f}s", flush=True)

# Plot tracker for particles using helix overlays
t0 = perf_counter()
print("Plotting particles helices (tracker only)...", flush=True)
fig = plot_odd_event(
    event_data,
    plot_sihits=True,
    plot_particles=True,
)
fig.savefig(plot_save_dir / Path("particles_tracker_helix.png"))
plt.close(fig)
print(f"Saved particles_tracker_helix.png in {perf_counter() - t0:.2f}s", flush=True)

# Plot tracker for tracks using helix overlays
t0 = perf_counter()
print("Plotting tracks helices (tracker only)...", flush=True)
fig = plot_odd_event(
    event_data,
    plot_sihits=True,
    plot_tracker=True,
)
fig.savefig(plot_save_dir / Path("tracks_tracker_helix.png"))
plt.close(fig)
print(f"Saved tracks_tracker_helix.png in {perf_counter() - t0:.2f}s", flush=True)
