from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap

from hepattn.experiments.trackml.data import TrackMLDataset


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_truth_level_mask(inputs, targets, event_idx=0, save_path="truth_level_mask.png"):
    # Get the particle_hit_valid mask from targets
    particle_hit_valid = targets["particle_hit_valid"][event_idx]  # Shape: (num_particles, num_hits)
    particle_valid = targets["particle_valid"][event_idx]  # Shape: (num_particles,)
    particle_phi = targets["particle_phi"][event_idx]  # Shape: (num_particles,)

    # Get hit phi values
    hit_phi = inputs["hit_phi"][event_idx]  # Shape: (num_hits,)

    # Get dimensions
    num_particles, num_hits = particle_hit_valid.shape

    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create a mask for valid particles only
    valid_particle_mask = particle_valid.unsqueeze(-1).expand_as(particle_hit_valid)

    # Sort particles by phi value in ascending order
    valid_particles = particle_valid.nonzero(as_tuple=True)[0]
    valid_phi = particle_phi[valid_particles]
    particle_sort_indices = torch.argsort(valid_phi)
    sorted_valid_particles = valid_particles[particle_sort_indices]

    # Sort hits by phi value in ascending order
    hit_sort_indices = torch.argsort(hit_phi)
    sorted_hit_indices = hit_sort_indices

    # Create sorted masks
    sorted_particle_hit_valid = particle_hit_valid[sorted_valid_particles][:, sorted_hit_indices]
    sorted_valid_particle_mask = valid_particle_mask[sorted_valid_particles][:, sorted_hit_indices]

    # Create the image array
    # 0 = blue (no association), 1 = yellow (association)
    image_array = np.zeros((len(sorted_valid_particles), num_hits), dtype=np.float32)

    # Set yellow (1.0) where there's a valid association
    valid_associations = sorted_particle_hit_valid & sorted_valid_particle_mask

    image_array[valid_associations] = 1.0  # No transpose needed - particles on y, hits on x

    colors = ["blue", "yellow"]
    cmap = LinearSegmentedColormap.from_list("blue_yellow", colors, N=2)

    # Plot the mask
    im = ax.imshow(image_array, cmap=cmap, aspect="auto", interpolation="nearest")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(["No Association", "Hit-Track Association"])

    # Set labels and title
    ax.set_xlabel("Hit Index (sorted by φ)")
    ax.set_ylabel("Particle/Track Index (sorted by φ)")
    ax.set_title(f"Truth Level Mask: Hit-Track Associations\nEvent {event_idx} - {len(sorted_valid_particles)} valid particles, {num_hits} hits")

    # Add statistics
    num_valid_particles = len(sorted_valid_particles)
    num_associations = valid_associations.sum().item()
    total_possible = num_valid_particles * num_hits

    stats_text = (
        f"Valid particles: {num_valid_particles}/{num_particles}\n"
        f"Total associations: {num_associations}\n"
        # f"Association density: {num_associations / total_possible:.3f}"
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Truth level mask saved as '{save_path}'")

    return fig


def create_enhanced_truth_mask(inputs, targets, event_idx=0, save_path="enhanced_truth_mask.png"):
    """
    Create an enhanced version with additional information about particle properties.
    """

    particle_hit_valid = targets["particle_hit_valid"][event_idx]
    particle_valid = targets["particle_valid"][event_idx]

    # Get particle properties for coloring
    particle_pt = targets["particle_pt"][event_idx]
    particle_eta = targets["particle_eta"][event_idx]
    particle_phi = targets["particle_phi"][event_idx]

    # Get hit phi values
    hit_phi = inputs["hit_phi"][event_idx]

    num_particles, num_hits = particle_hit_valid.shape

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Sort particles by phi value in ascending order
    valid_particles = particle_valid.nonzero(as_tuple=True)[0]
    valid_phi = particle_phi[valid_particles]
    particle_sort_indices = torch.argsort(valid_phi)
    sorted_valid_particles = valid_particles[particle_sort_indices]

    # Sort hits by phi value in ascending order
    hit_sort_indices = torch.argsort(hit_phi)
    sorted_hit_indices = hit_sort_indices

    # Create sorted masks
    sorted_particle_hit_valid = particle_hit_valid[sorted_valid_particles][:, sorted_hit_indices]
    sorted_valid_particle_mask = particle_valid.unsqueeze(-1).expand_as(particle_hit_valid)[sorted_valid_particles][:, sorted_hit_indices]

    # Plot 1: Basic truth mask
    image_array = np.zeros((len(sorted_valid_particles), num_hits), dtype=np.float32)
    valid_associations = sorted_particle_hit_valid & sorted_valid_particle_mask
    image_array[valid_associations] = 1.0

    colors = ["blue", "yellow"]
    cmap = LinearSegmentedColormap.from_list("blue_yellow", colors, N=2)

    im1 = ax1.imshow(image_array, cmap=cmap, aspect="auto", interpolation="nearest")
    ax1.set_xlabel("Hit Index (sorted by φ)")
    ax1.set_ylabel("Particle/Track Index (sorted by φ)")
    ax1.set_title(f"Truth Level Mask - Event {event_idx}")

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1])
    cbar1.set_ticklabels(["No Association", "Hit-Track Association"])

    # Plot 2: Particle properties (sorted by phi)
    sorted_valid_pt = particle_pt[sorted_valid_particles]
    sorted_valid_eta = particle_eta[sorted_valid_particles]
    sorted_valid_phi = particle_phi[sorted_valid_particles]

    scatter = ax2.scatter(range(len(sorted_valid_particles)), sorted_valid_pt, c=sorted_valid_eta, cmap="viridis", s=50, alpha=0.7)
    ax2.set_xlabel("Particle Index (sorted by φ)")
    ax2.set_ylabel("Particle pT [GeV]")
    ax2.set_title("Particle Properties (pT vs Index, colored by η)")
    ax2.grid(True, alpha=0.3)

    # Add colorbar for eta
    cbar2 = plt.colorbar(scatter, ax=ax2)
    cbar2.set_label("Particle η")

    # Add statistics
    num_valid_particles = len(sorted_valid_particles)
    num_associations = valid_associations.sum().item()
    total_possible = num_valid_particles * num_hits

    stats_text = (
        f"Valid particles: {num_valid_particles}/{num_particles}\n"
        f"Total associations: {num_associations}\n"
        f"Association density: {num_associations / total_possible:.3f}\n"
        f"Avg pT: {sorted_valid_pt.mean():.2f} GeV\n"
        f"Avg |η|: {sorted_valid_eta.abs().mean():.2f}"
    )

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Enhanced truth mask saved as '{save_path}'")

    return fig


def test_positional_encodings_truth_mask():
    """Test function to create truth level mask visualization for TrackML data."""

    # Configuration similar to the tracking config
    input_fields = {"hit": ["x", "y", "z", "r", "s", "eta", "phi", "u", "v", "charge_frac", "leta", "lphi", "lx", "ly", "lz", "geta", "gphi"]}

    target_fields = {"particle": ["pt", "eta", "phi"]}

    # Data path - adjust as needed
    dirpath = "data/trackml/prepped/"

    # Check if data directory exists
    if not Path(dirpath).exists():
        pytest.skip(f"Data directory {dirpath} not found. Skipping test.")

    # Create dataset with smaller parameters for testing
    dataset = TrackMLDataset(
        dirpath=dirpath,
        inputs=input_fields,
        targets=target_fields,
        num_events=2,  # Just 2 events for testing
        hit_volume_ids=[8],  # Just barrel pixels
        particle_min_pt=1.0,
        particle_max_abs_eta=2.5,
        particle_min_num_hits=3,
        event_max_num_particles=100,  # Smaller for testing
    )

    # Create output directory
    output_dir = Path("tests/")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Process each event
    for event_idx in range(min(2, len(dataset))):
        print(f"Processing event {event_idx}...")

        inputs, targets = dataset[event_idx]

        # Create basic truth mask
        mask_path = output_dir / f"truth_mask_event_{event_idx}.png"
        fig1 = create_truth_level_mask(inputs, targets, event_idx, str(mask_path))
        plt.close(fig1)

        # Create enhanced truth mask
        enhanced_path = output_dir / f"enhanced_truth_mask_event_{event_idx}.png"
        fig2 = create_enhanced_truth_mask(inputs, targets, event_idx, str(enhanced_path))
        plt.close(fig2)

        # Print some statistics
        particle_hit_valid = targets["particle_hit_valid"][0]
        particle_valid = targets["particle_valid"][0]
        num_valid_particles = particle_valid.sum().item()
        num_associations = (particle_hit_valid & particle_valid.unsqueeze(-1)).sum().item()

        print(f"Event {event_idx}: {num_valid_particles} valid particles, {num_associations} hit-track associations")

    print(f"All visualizations saved to: {output_dir}")


def test_positional_encodings_with_config():
    """Test function that loads data using a configuration file."""

    # Try to load from a config file
    config_path = "/share/rcifdata/pduckett/hepattn-log-attn-mask/src/hepattn/experiments/trackml/configs/queryPE-lite-alpha0p1base100.yaml"

    if not Path(config_path).exists():
        pytest.skip(f"Config file {config_path} not found. Skipping test.")

    # Load configuration
    config = load_config(config_path)
    data_config = config.get("data", {})

    # Create dataset using config
    dataset = TrackMLDataset(
        dirpath=data_config.get("train_dir", data_config.get("train_dir")),
        inputs=data_config["inputs"],
        targets=data_config["targets"],
        num_events=1,  # Just 1 event for testing
        hit_volume_ids=data_config.get("hit_volume_ids"),
        particle_min_pt=data_config.get("particle_min_pt", 1.0),
        particle_max_abs_eta=data_config.get("particle_max_abs_eta", 2.5),
        particle_min_num_hits=data_config.get("particle_min_num_hits", 3),
        event_max_num_particles=data_config.get("event_max_num_particles", 100),
    )

    # Create output directory
    output_dir = Path("tests/")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Process the event
    if len(dataset) > 0:
        inputs, targets = dataset[0]

        # Create truth mask
        mask_path = output_dir / "truth_mask_from_config.png"
        fig = create_truth_level_mask(inputs, targets, 0, str(mask_path))
        plt.close(fig)

        print(f"Truth mask from config saved as '{mask_path}'")


if __name__ == "__main__":
    # Run the tests directly if script is executed
    # test_positional_encodings_truth_mask()
    test_positional_encodings_with_config()
