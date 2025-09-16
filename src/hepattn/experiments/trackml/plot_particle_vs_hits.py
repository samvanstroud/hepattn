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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# Add the src directory to the path so we can import hepattn modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import TrackMLDataset


def is_valid_track(assigned_hits, class_pred):
    """Cbeck whether an output track query is valid  - i.e. not a null output."""
    return assigned_hits >= 3 and class_pred == 0


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
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")

    parser.add_argument(
        "--eval_file_path",
        type=str,
        default=None,
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

    print(f"Loading TrackML dataset from: {data_dir}")
    print(f"Configuration parameters:")
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
        event_max_num_particles=5000,
        hit_eval_path=hit_eval_path,
    )

    total_events = len(dataset)

    # Collect data across events
    particle_counts = []
    hit_counts = []

    for idx in range(total_events):
        # # Load the event data
        # hits, particles = dataset.load_event(idx)
        # num_particles = len(particles)

        # Load the event data
        inputs, targets = dataset[idx]

        # Extract the counts
        num_valid_particles = torch.sum(targets["particle_valid"]).item()
        # num_hits_on_valid_particles = torch.sum(
        #     targets["hit_on_valid_particle"]
        # ).item()
        num_hits = inputs["hit_phi"].shape[1]

        particle_counts.append(num_valid_particles)
        # hit_counts.append(num_hits_on_valid_particles)
        hit_counts.append(num_hits)

    # Convert to numpy arrays
    particle_counts = np.array(particle_counts)
    hit_counts = np.array(hit_counts)

    print(f"Successfully processed {len(particle_counts)} events")
    print(f"Particle counts: min={particle_counts.min()}, max={particle_counts.max()}, mean={particle_counts.mean():.2f}")
    print(f"Hit counts: min={hit_counts.min()}, max={hit_counts.max()}, mean={hit_counts.mean():.2f}")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Main scatter plot
    plt.scatter(particle_counts, hit_counts, alpha=0.6, s=50)

    # Add trend line
    if len(particle_counts) > 1:
        z = np.polyfit(particle_counts, hit_counts, 1)
        p = np.poly1d(z)
        plt.plot(particle_counts, p(particle_counts), "r--", alpha=0.8, linewidth=2)

        # Calculate correlation
        correlation = np.corrcoef(particle_counts, hit_counts)[0, 1]
        plt.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Add diagonal line for reference (1:1 relationship)
    max_val = max(particle_counts.max(), hit_counts.max())
    plt.plot([0, max_val], [0, max_val], "k--", alpha=0.5, linewidth=1, label="1:1 line")

    plt.xlabel("Number of Valid Particles", fontsize=14)
    plt.ylabel("Number of Hits on Valid Particles", fontsize=14)
    plt.title(
        f"Particle vs Hits Relationship (test set, {len(particle_counts)} events)",
        fontsize=16,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add some statistics as text
    mean_particles = particle_counts.mean()
    mean_hits = hit_counts.mean()
    ratio = hit_counts.mean() / particle_counts.mean()
    stats_text = f"Mean particles: {mean_particles:.1f}\nMean hits: {mean_hits:.1f}\nRatio: {ratio:.2f}"
    plt.text(
        0.05,
        0.85,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()

    output_file = "./outfile.png"
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Also create a histogram of the ratio
    if len(particle_counts) > 0:
        ratios = hit_counts / (particle_counts + 1e-8)  # Avoid division by zero

        plt.figure(figsize=(10, 6))
        plt.hist(ratios, bins=30, alpha=0.7, edgecolor="black")
        plt.xlabel("Hits per Particle Ratio", fontsize=14)
        plt.ylabel("Number of Events", fontsize=14)
        plt.title(f"Distribution of Hits per Particle Ratio (test set)", fontsize=16)
        plt.grid(True, alpha=0.3)

        # Add mean line
        mean_ratio = np.mean(ratios)
        plt.axvline(
            mean_ratio,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_ratio:.2f}",
        )
        plt.legend()

        ratio_output = output_file.replace(".png", "_ratio_hist.png")
        plt.savefig(ratio_output, dpi=300, bbox_inches="tight")
        print(f"Ratio histogram saved to {ratio_output}")

    plt.show()


if __name__ == "__main__":
    main()
