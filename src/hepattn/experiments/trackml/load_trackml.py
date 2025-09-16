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

import yaml
import h5py

import pandas as pd
import numpy as np

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
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )

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

    # dataset = TrackMLDataset(
    #     dirpath=data_dir,
    #     inputs=inputs,
    #     targets=targets,
    #     num_events=num_events,
    #     hit_volume_ids=hit_volume_ids,
    #     particle_min_pt=particle_min_pt,
    #     particle_max_abs_eta=particle_max_abs_eta,
    #     particle_min_num_hits=particle_min_num_hits,
    #     event_max_num_particles=event_max_num_particles,
    #     hit_eval_path=hit_eval_path,
    #     )
    
    test_eval_path = "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn/logs/TRK-ISAMBARD-SCALEUP-epochs10-alpha1p3-ma-regression-cylindrical-enc-pes_20250907-T171936/ckpts/epoch=009-val_loss=0.53207_test_eval.h5"

    with h5py.File(test_eval_path, "r") as hit_eval_file:
        tracking_test = hit_eval_file["29800"]["preds/final"]["query_regression"]['track_px'][:]
        # tracking_test = hit_eval_file["29800"]["preds"]["final"]["track_hit_valid"]
        # ["track_valid"]
        print(tracking_test)

        # ['targets']['hit_on_valid_particle', 'hit_valid', 'particle_eta', 'particle_hit_valid', 'particle_phi', 'particle_pt', 'particle_px', 'particle_py', 'particle_pz', 'particle_valid', 'sample_id']

    # total_events = len(dataset)

    # for idx in range(total_events):

    #     # Load the event data
    #     hits, particles = dataset.load_event(idx)
    #     num_particles = len(particles)


def load_event(fname, idx):
    """
    Load an event from an evaluation file and convert to DataFrame.
    """
    f = h5py.File(fname)
    g = f[idx]

    # load unfiltered truth csv
    truth = pd.DataFrame({"pid": g["targets/sample_id"][0]})
    truth.index = truth.pid

    # load particles
    parts = pd.DataFrame({"pt": g["targets/particle_pt"][0], "eta": g["targets/particle_eta"][0], "phi": g["targets/particle_phi"][0]})
    parts.index = parts.pid

    # load hits
    hits = pd.DataFrame({"pid": g["hits/pids"][:]})
    hits.index = hits.pid

    # get masks
    masks = g["preds/final/track_hit_valid/track_hit_valid"][0]

    # load tracks (model outputs)
    tracks = pd.DataFrame({"class_pred": g["preds/final/track_valid/track_valid"][0]})
    tracks["n_assigned"] = masks.sum(-1)
    # for k in g["preds/regression"]:
    #     tracks[k] = g[f"preds/regression/{k}"][:]
    # tracks["pt"] = np.sqrt(tracks.px**2 + tracks.py**2)
    # tracks["phi"] = np.arctan2(tracks.py, tracks.px)
    # theta = np.arctan2(tracks.pt, tracks.pz)
    # tracks["eta"] = -np.log(np.tan(0.5 * theta))

    # basic sanity checks
    assert len(np.unique(parts.pid) == len(parts)), "particle ids are not unique!"

    return hits, tracks, masks, parts, truth



if __name__ == "__main__":
    main()
