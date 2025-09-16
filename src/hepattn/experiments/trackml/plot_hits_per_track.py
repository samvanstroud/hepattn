#!/usr/bin/env python3
"""
Script to plot a histogram showing the number of hits per track over all tracks across all events.

This script loads TrackML data using the same configuration structure as run_tracking.py
and creates a histogram showing the distribution of hits per track.

Usage:
    python plot_hits_per_track_histogram.py --config path/to/config.yaml
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    """Main function to create hits per track histogram."""
    parser = argparse.ArgumentParser(
        description="Plot histogram of hits per track across all events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--output_file",
        type=str,
        default="hits_per_track_histogram.png",
        help="Output file path for the histogram",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=-1,
        help="Maximum number of events to process (-1 for all events)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    print("Loading config:", config_path)

    config = load_config(config_path)

    # Extract data configuration
    data_config = config.get("data", {})

    data_dir = data_config["train_dir"]
    hit_eval_path = None

    hit_volume_ids = data_config["hit_volume_ids"]
    particle_min_pt = data_config["particle_min_pt"]
    particle_max_abs_eta = data_config["particle_max_abs_eta"]
    particle_min_num_hits = data_config["particle_min_num_hits"]
    event_max_num_particles = data_config["event_max_num_particles"]

    # Extract inputs and targets configuration
    inputs = data_config["inputs"]
    targets = data_config["targets"]

    print(f"Loading TrackML dataset from: {data_dir}")
    print(f"Configuration parameters:")
    print(f"  - Max events to process: {args.max_events}")
    print(f"  - Hit volume IDs: {hit_volume_ids}")
    print(f"  - Particle min pt: {particle_min_pt}")
    print(f"  - Particle max abs eta: {particle_max_abs_eta}")
    print(f"  - Particle min num hits: {particle_min_num_hits}")
    print(f"  - Event max num particles: {event_max_num_particles}")
    print(f"  - Hit eval path: {hit_eval_path}")
    print()

    # Create dataset
    dataset = TrackMLDataset(
        dirpath=data_dir,
        inputs=inputs,
        targets=targets,
        num_events=args.max_events,
        hit_volume_ids=hit_volume_ids,
        particle_min_pt=particle_min_pt,
        particle_max_abs_eta=particle_max_abs_eta,
        particle_min_num_hits=particle_min_num_hits,
        event_max_num_particles=5000,
        hit_eval_path=hit_eval_path,
    )

    total_events = len(dataset)
    print(f"Processing {total_events} events...")

    # Collect hits per track data across all events
    hits_per_track = []
    total_tracks = 0

    for idx in range(total_events):
        if idx % 100 == 0:
            print(f"Processing event {idx}/{total_events}")

        # Load the event data
        inputs, targets = dataset[idx]

        # # Get particle valid mask and hit-to-particle assignments
        # targets_particle_valid = targets[targets["particle_valid"][0]]  # Shape: [max_particles]
        # hit_on_valid_particle = targets_particle_valid["particle_hit_valid"][0]
        # print(hit_on_valid_particle.shape)
        # # sys.exit()
        # n_hits = hit_on_valid_particle.sum(axis=1)
        particle_valid = targets["particle_valid"][0]  # Shape: [max_particles]
        particle_hit_valid = targets["particle_hit_valid"][0]  # Shape: [max_particles, max_hits]
        hit_valid = targets["hit_on_valid_particle"][0]
        particle_hits_reduced = particle_hit_valid[:, hit_valid]
        hit_on_valid_particle = particle_hits_reduced[particle_valid, :]
        # print(particle_reduced_hits_reduced.shape)

        # sys.exit()

        # # Only consider valid particles
        # valid_particle_mask = particle_valid  # Boolean mask for valid particles

        # # Get hit assignments for valid particles only
        # hit_on_valid_particle = particle_hit_valid[valid_particle_mask]  # Shape: [num_valid_particles, max_hits]
        # print(hit_on_valid_particle.shape)

        # print(torch.allclose(hit_on_valid_particle, targets["hit_on_valid_particle"][0]))
        # sys.exit()
        # hit_on_valid_particle = targets["hit_on_valid_particle"][0]

        # Count hits per valid particle
        n_hits_per_particle = hit_on_valid_particle.sum(axis=1)  # Shape: [num_valid_particles]

        # Find index of particle with max hits and remove it.This is the null particle (I think)
        # print("---------------------------")
        # print(torch.max(n_hits_per_particle))
        # max_hits_idx = torch.argmax(n_hits_per_particle)
        # # Remove the particle with max hits
        # n_hits_per_particle = torch.cat([
        #     n_hits_per_particle[:max_hits_idx],
        #     n_hits_per_particle[max_hits_idx + 1 :],
        # ])
        # print(torch.max(n_hits_per_particle))
        # print("---------------------------")

        hits_per_track += n_hits_per_particle.tolist()

    # Convert to numpy array
    hits_per_track = np.array(hits_per_track)

    # print(len(hits_per_track[hits_per_track >= 30]))

    # hits_per_track = hits_per_track[hits_per_track < 30]

    # print(f"\nSuccessfully processed {total_events} events")

    # Remove outliers using IQR method
    Q3 = np.percentile(hits_per_track, 99.8)

    print(f"Before outlier removal: {len(hits_per_track)} tracks, max hits: {hits_per_track.max()}")
    # Filter outliers
    hits_per_track = hits_per_track[hits_per_track <= Q3]

    print(f"After outlier removal: {len(hits_per_track)} tracks, max hits: {hits_per_track.max()}")

    # Create the histogram
    plt.figure(figsize=(20, 8))

    # Create histogram with appropriate bins
    bins = np.arange(hits_per_track.min(), hits_per_track.max() + 2) - 0.5
    n, _, patches = plt.hist(hits_per_track, bins=bins, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Add statistics text
    stats_text = f"""Statistics:
    Mean hits/track: {hits_per_track.mean():.2f}
    Median hits/track: {np.median(hits_per_track):.2f}
    Std dev: {hits_per_track.std():.2f}
    Min: {hits_per_track.min()}
    """

    plt.text(
        1.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    # Add mean line
    mean_hits = hits_per_track.mean()
    plt.axvline(
        mean_hits,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_hits:.2f}",
    )

    # Add median line
    median_hits = np.median(hits_per_track)
    plt.axvline(
        median_hits,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_hits:.2f}",
    )

    plt.xlabel("Number of Hits per Track", fontsize=14)
    plt.ylabel("Number of Tracks", fontsize=14)
    plt.title(
        f"Distribution of Hits per Track\n({total_events} events)",
        fontsize=16,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set x-axis to show integer values
    plt.xticks(range(int(hits_per_track.min()), int(hits_per_track.max()) + 1))

    plt.tight_layout()

    # Save the plot
    plt.savefig(args.output_file, dpi=300, bbox_inches="tight")
    print(f"\nHistogram saved to {args.output_file}")

    # Also create a cumulative distribution plot
    plt.figure(figsize=(50, 6))

    # Calculate cumulative distribution
    sorted_hits = np.sort(hits_per_track)
    cumulative = np.arange(1, len(sorted_hits) + 1) / len(sorted_hits)

    plt.plot(sorted_hits, cumulative, linewidth=2)
    plt.xlabel("Number of Hits per Track", fontsize=14)
    plt.ylabel("Cumulative Probability", fontsize=14)
    plt.title(
        f"Cumulative Distribution of Hits per Track\n({total_events} events, {total_tracks:,} tracks)",
        fontsize=16,
    )
    plt.grid(True, alpha=0.3)

    # Add percentile lines
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(hits_per_track, p)
        plt.axvline(value, color="red", linestyle="--", alpha=0.7)
        plt.text(value, 0.1, f"{p}%", rotation=90, fontsize=10)

    plt.tight_layout()

    # Save cumulative plot
    cum_output = args.output_file.replace(".png", "_cumulative.png")
    plt.savefig(cum_output, dpi=300, bbox_inches="tight")
    print(f"Cumulative distribution plot saved to {cum_output}")

    plt.show()


if __name__ == "__main__":
    main()
