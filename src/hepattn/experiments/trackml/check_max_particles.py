#!/usr/bin/env python3
"""
Script to check the maximum number of particles in TrackML events.

This script loads TrackML data using the same configuration structure as run_tracking.py
and finds the maximum number of particles across all events in the dataset.

Usage:
    python check_max_particles.py --config path/to/config.yaml
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add the src directory to the path so we can import hepattn modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import TrackMLDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function to check maximum particles in TrackML data."""
    parser = argparse.ArgumentParser(
        description="Check maximum number of particles in TrackML events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--histogram-out",
        type=str,
        default=None,
        help="Path to save histogram PNG; defaults near the config file",
    )

    args = parser.parse_args()

    # Extract dataset parameters
    num_events = -1

    config_path = Path(args.config)
    print("loading config :", config_path)

    config = load_config(config_path)

    # Extract data configuration
    data_config = config.get("data", {})

    data_dir = data_config["train_dir"]
    hit_eval_path = data_config["hit_eval_train"]

    hit_volume_ids = data_config["hit_volume_ids"]
    particle_min_pt = data_config["particle_min_pt"]
    particle_max_abs_eta = data_config["particle_max_abs_eta"]
    particle_min_num_hits = data_config["particle_min_num_hits"]
    event_max_num_particles = data_config["event_max_num_particles"]

    # Extract inputs and targets configuration
    inputs = data_config["inputs"]
    targets = data_config["targets"]
    # Extract inputs and targets configuration
    inputs = {"hit": ["x", "y", "z", "r", "eta", "phi"]}
    targets = {"particle": ["pt", "eta", "phi"]}

    print(f"Loading TrackML dataset from: {data_dir}")
    print("Configuration parameters:")
    print(f"  - Number of events: {num_events}")
    print(f"  - Hit volume IDs: {hit_volume_ids}")
    print(f"  - Particle min pt: {particle_min_pt}")
    print(f"  - Particle max abs eta: {particle_max_abs_eta}")
    print(f"  - Particle min num hits: {particle_min_num_hits}")
    print(f"  - Event max num particles: {event_max_num_particles}")
    print(f"  - Hit eval path: {hit_eval_path}")
    print()

    dataset = TrackMLDataset(
        dirpath=data_dir,
        inputs=inputs,
        targets=targets,
        num_events=num_events,
        hit_volume_ids=hit_volume_ids,
        particle_min_pt=particle_min_pt,
        particle_max_abs_eta=particle_max_abs_eta,
        particle_min_num_hits=particle_min_num_hits,
        event_max_num_particles=event_max_num_particles,
        hit_eval_path=hit_eval_path,
    )

    print(f"Dataset created successfully with {len(dataset)} events")
    print("Processing events to count particles...")
    print()

    # Track maximum particles and some statistics
    max_particles = 0
    max_particles_event = -1
    total_events = len(dataset)
    total_particles = 0
    particle_counts = []

    # Process each event
    # total_events = 1000
    for idx in range(total_events):
        _, particles = dataset.load_event(idx)
        num_particles = len(particles)

        # Update statistics
        particle_counts.append(num_particles)
        total_particles += num_particles

        if num_particles > max_particles:
            max_particles = num_particles
            max_particles_event = idx

        # Print progress every 10 events
        if (idx + 1) % 10 == 0 or idx == total_events - 1:
            print(f"Processed {idx + 1}/{total_events} events...")

    # Calculate statistics
    if particle_counts:
        avg_particles = total_particles / total_events
        min_particles = min(particle_counts)
        particles_per_event = np.array(particle_counts)
        # Filter particles per event at 99th percentile
        particles_99th_percentile = np.percentile(particles_per_event, 97)
        particles_per_event_filtered = particles_per_event[particles_per_event <= particles_99th_percentile]

        print(
            f"After 99th percentile filtering: {len(particles_per_event_filtered)} events (removed {len(particles_per_event) - len(particles_per_event_filtered)} outliers)"
        )
        print(f"New max particles per event: {particles_per_event_filtered.max()}")

        print()
        print("=" * 60)
        print("PARTICLE COUNT STATISTICS")
        print("=" * 60)
        print(f"Total events processed: {total_events}")
        print(f"Maximum particles in any event: {max_particles}")
        print(f"Event with maximum particles: {max_particles_event}")
        print(f"Minimum particles in any event: {min_particles}")
        print(f"Average particles per event: {avg_particles:.2f}")
        print(f"Total particles across all events: {total_particles}")
        print()

        # Check against configured limit
        if max_particles > event_max_num_particles:
            print(f"⚠️  WARNING: Maximum particles ({max_particles}) exceeds configured limit ({event_max_num_particles})")
            print("   This may cause issues during training/inference.")
        else:
            print(f"✅ Maximum particles ({max_particles}) is within configured limit ({event_max_num_particles})")

        # Plot and save histogram of particles per event
        bin_width = 30
        bins = np.arange(0, 6000 + bin_width, bin_width)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(particles_per_event, bins=bins, color="C0", edgecolor="black")
        ax.set_xlabel("Particles per event")
        ax.set_ylabel("Number of events")
        ax.set_title("Distribution of particles per event")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xlim(min_particles - 0.5, max_particles + 0.5)

        if args.histogram_out is not None:
            out_path = Path(args.histogram_out)
        else:
            out_name = f"particles_per_event_hist_{config_path.stem}.png"
            out_path = config_path.parent / out_name

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved histogram to: {out_path}")

    else:
        print("No events were successfully processed.")


if __name__ == "__main__":
    main()
