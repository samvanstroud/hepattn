#!/usr/bin/env python3
"""Script to calculate the average number of hits per event in TrackML data.

This script loads TrackML data using the same configuration structure as run_tracking.py
and calculates the average number of hits across all events in the dataset.

Usage:
    python plot_particle_vs_hits.py --config path/to/config.yaml
"""

import argparse
import sys
from pathlib import Path

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
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def calculate_avg_hits_for_config(config_path: str) -> float:
    """Calculate average hits per event for a single config.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Average number of hits per event
    """
    # Extract dataset parameters
    num_events = -1

    config_path = Path(config_path)
    print("Loading config:", config_path)

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
        event_max_num_particles=5000,
        hit_eval_path=hit_eval_path,
    )

    total_events = len(dataset)

    # Collect hit counts across events
    hit_counts = []

    for idx in range(total_events):
        # Load the event data
        inputs, targets = dataset[idx]

        # Extract the number of hits
        num_hits = inputs["hit_phi"].shape[1]
        hit_counts.append(num_hits)

    # Calculate average
    average_hits = sum(hit_counts) / len(hit_counts)

    print(f"Successfully processed {len(hit_counts)} events")
    print(f"Hit counts: min={min(hit_counts)}, max={max(hit_counts)}, mean={average_hits:.2f}")
    print(f"Average number of hits per event: {average_hits:.2f}")

    return average_hits


def main(config_dict: dict) -> dict:
    """Main function to calculate average hits per event for multiple configs.

    Args:
        config_dict: Dictionary mapping config names to config file paths

    Returns:
        Dictionary mapping config names to average number of hits
    """
    results = {}

    for name, config_path in config_dict.items():
        print(f"\n{'=' * 50}")
        print(f"Processing config: {name}")
        print(f"{'=' * 50}")

        try:
            avg_hits = calculate_avg_hits_for_config(config_path)
            results[name] = avg_hits
        except Exception as e:
            print(f"Error processing config {name}: {e}")
            results[name] = None

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    for name, avg_hits in results.items():
        if avg_hits is not None:
            print(f"{name}: {avg_hits:.2f} hits per event")
        else:
            print(f"{name}: ERROR")

    return results

def save_results(results: dict, output_path: str) -> None:
    """Save results to a YAML file.

    Args:
        results: Dictionary mapping config names to average hit counts
        output_path: Path where to save the results
    """
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results to YAML file
    with output_path.open("w") as f:
        yaml.dump(results, f, default_flow_style=False, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage: Define your config dictionary here

    config_dict = {
        "pT=600MeV, eta=4, strips": "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn-scale-up/src/hepattn/experiments/trackml/configs/epochs30-lca-eta4-pt600-strips.yaml",
        # "pT=1GeV, eta=2.5": "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn-scale-up/src/hepattn/experiments/trackml/configs/epochs30-lca-eta2p5-pt1gev.yaml",
        # "pT=600MeV, eta=2.5 ": "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn-scale-up/src/hepattn/experiments/trackml/configs/epochs10-alpha1p3-no-bd-local-ca-w512-flex.yaml",
        # "pT=500MeV, eta=2.5": "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn-scale-up/src/hepattn/experiments/trackml/configs/epochs10-alpha1p3-no-bd-local-ca-w512-flex-eta2p5-pt500.yaml",
        # "pT=600MeV, eta=4": "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn-scale-up/src/hepattn/experiments/trackml/configs/epochs10-alpha1p3-no-bd-local-ca-w512-flex-eta4.yaml",
        # "pT=500MeV, eta=4": "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn-scale-up/src/hepattn/experiments/trackml/configs/epochs10-alpha1p3-no-bd-local-ca-w512-flex-eta4-pt500.yaml",
    }

    # Output path for saving results
    output_path = "results/avg_hits_per_config.yaml"

    # Call main with the config dictionary
    results = main(config_dict)

    # Print final results
    print(f"\nFinal results: {results}")

    # Save results to file
    save_results(results, output_path)