#!/usr/bin/env python3
# ruff: noqa: PLC0415,N806,E501,DTZ005,ARG001,D417,EXE001
"""Evaluation script for ATLAS muon hit filtering model using DataLoader approach.
This version uses the AtlasMuonDataModule for proper multi-worker data loading.

PERFORMANCE OPTIMIZATIONS:
- Memory-efficient data structures with proper dtypes
- Pre-allocated arrays and vectorized operations
- Reduced redundant ROC calculations
- Optimized plotting with selective output
- Proper signal handling for background execution
- Memory monitoring and garbage collection
"""

import argparse

import h5py
import matplotlib as mpl
import numpy as np

mpl.use("Agg")  # Set backend before importing pyplot
import gc
import multiprocessing as mp
import signal
import sys
import traceback
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Memory monitoring
import psutil
import yaml
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

# Import the data module
sys.path.append(str(Path(__file__).parent / "../../.."))

from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

warnings.filterwarnings("ignore")


# Configure signal handling for proper background execution
def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}. Cleaning up and exiting...")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Global configuration constants
# DEFAULT_WORKING_POINTS = [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
DEFAULT_WORKING_POINTS = [0.975, 0.985, 0.99, 0.995]

# Set matplotlib backend and style
plt.switch_backend("Agg")
plt.style.use("default")
plt.rcParams.update({
    "font.size": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (10, 6),
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "errorbar.capsize": 4,
})


def _process_track_chunk(
    track_chunk, all_event_ids, all_particle_ids, all_particle_pts, all_particle_etas, all_station_indices, true_hit_mask, min_pt
):
    """Worker function to process a chunk of tracks for baseline filtering.
    This function is designed to be called by multiprocessing.Pool.

    Args:
        track_chunk: List of (event_id, particle_id) tuples to process
        all_event_ids: Array of event IDs for all hits
        all_particle_ids: Array of particle IDs for all hits
        all_particle_pts: Array of particle pT values for all hits
        all_particle_etas: Array of particle eta values for all hits
        all_station_indices: Array of station indices for all hits
        true_hit_mask: Boolean mask for true hits

    Returns:
        Dictionary with qualified tracks and statistics for this chunk
    """
    chunk_stats = {
        "total_tracks_checked": 0,
        "tracks_failed_min_hits": 0,
        "tracks_failed_eta_cuts": 0,
        "tracks_failed_pt_cuts": 0,
        "tracks_failed_station_cuts": 0,
        "tracks_passed_all_cuts": 0,
    }

    qualified_tracks = set()
    # print("This is events ids unique: ", np.unique(all_particle_ids) )
    # print("This is the number of unique event ids: ", len(np.unique(all_event_ids)) )
    # print("sum true hits: ", np.sum(true_hit_mask))
    for event_id, particle_id in track_chunk:
        chunk_stats["total_tracks_checked"] += 1

        # Get hits for this specific track
        track_mask = (all_event_ids == event_id) & (all_particle_ids == particle_id) & true_hit_mask
        # print("this is the track mask sum: ", np.sum(track_mask) )
        track_hits = np.sum(track_mask)

        # Pre-filter 1: tracks must have at least 9 hits total
        if track_hits < 9:
            chunk_stats["tracks_failed_min_hits"] += 1
            continue

        # Get particle kinematic properties for this track
        track_indices = np.where(track_mask)[0]
        if len(track_indices) == 0:
            chunk_stats["tracks_failed_min_hits"] += 1
            continue

        # Use the first hit to get particle properties (all hits from same particle should have same pt/eta)
        first_hit_idx = track_indices[0]
        track_pt = all_particle_pts[first_hit_idx]
        track_eta = all_particle_etas[first_hit_idx]

        # Pre-filter 2: eta acceptance cuts |eta| >= 0.1 and |eta| <= 2.7
        if np.abs(track_eta) < 0.1 or np.abs(track_eta) > 2.7:
            chunk_stats["tracks_failed_eta_cuts"] += 1
            continue

        # Pre-filter 3: pt threshold >= self.min_pt
        if track_pt < min_pt:
            chunk_stats["tracks_failed_pt_cuts"] += 1
            continue

        # Get station indices for this track
        track_stations = all_station_indices[track_mask]
        unique_stations, station_counts = np.unique(track_stations, return_counts=True)
        # print("this is unique stations: ", unique_stations)

        # Check station requirements:
        # 1. At least 3 different stations
        if len(unique_stations) < 3:
            chunk_stats["tracks_failed_station_cuts"] += 1
            continue

        # 2. Each station must have at least 3 hits
        n_good_stations = np.sum(station_counts >= 3)
        if n_good_stations < 3:
            chunk_stats["tracks_failed_station_cuts"] += 1
            continue

        # Track passed all criteria
        qualified_tracks.add((event_id, particle_id))
        chunk_stats["tracks_passed_all_cuts"] += 1

    return {"qualified_tracks": qualified_tracks, "stats": chunk_stats}


def _process_track_chunk_ml_region(track_chunk, all_event_ids, all_particle_ids, all_particle_pts, all_particle_etas, true_hit_mask):
    """Worker function to process a chunk of tracks for ML region filtering.
    Uses same criteria as prep_events_multiprocess.py:
    - pt >= 5.0 GeV
    - |eta| <= 2.7
    - >= 3 hits per track.

    Args:
        track_chunk: List of (event_id, particle_id) tuples to process
        all_event_ids: Array of event IDs for all hits
        all_particle_ids: Array of particle IDs for all hits
        all_particle_pts: Array of particle pT values for all hits
        all_particle_etas: Array of particle eta values for all hits
        true_hit_mask: Boolean mask for true hits

    Returns:
        Dictionary with qualified tracks and statistics for this chunk
    """
    chunk_stats = {
        "total_tracks_checked": 0,
        "tracks_failed_min_hits": 0,
        "tracks_failed_eta_cuts": 0,
        "tracks_failed_pt_cuts": 0,
        "tracks_passed_all_cuts": 0,
    }

    qualified_tracks = set()

    # ML region criteria (matching prep_events_multiprocess.py defaults)
    ML_PT_THRESHOLD = 5.0  # GeV
    ML_ETA_THRESHOLD = 2.7  # |eta| <= 2.7
    ML_MIN_HITS = 3  # minimum hits per track

    for event_id, particle_id in track_chunk:
        chunk_stats["total_tracks_checked"] += 1

        # Get hits for this specific track
        track_mask = (all_event_ids == event_id) & (all_particle_ids == particle_id) & true_hit_mask
        track_hits = np.sum(track_mask)

        # ML Pre-filter 1: tracks must have at least 3 hits total
        if track_hits < ML_MIN_HITS:
            chunk_stats["tracks_failed_min_hits"] += 1
            continue

        # Get particle kinematic properties for this track
        track_indices = np.where(track_mask)[0]
        if len(track_indices) == 0:
            chunk_stats["tracks_failed_min_hits"] += 1
            continue

        # Use the first hit to get particle properties (all hits from same particle should have same pt/eta)
        first_hit_idx = track_indices[0]
        track_pt = all_particle_pts[first_hit_idx]
        track_eta = all_particle_etas[first_hit_idx]

        # ML Pre-filter 2: eta acceptance cuts |eta| <= 2.7
        if np.abs(track_eta) > ML_ETA_THRESHOLD:
            chunk_stats["tracks_failed_eta_cuts"] += 1
            continue

        # ML Pre-filter 3: pt threshold >= 5.0 GeV
        if track_pt < ML_PT_THRESHOLD:
            chunk_stats["tracks_failed_pt_cuts"] += 1
            continue

        # Track passed all ML region criteria
        qualified_tracks.add((event_id, particle_id))
        chunk_stats["tracks_passed_all_cuts"] += 1

    return {"qualified_tracks": qualified_tracks, "stats": chunk_stats}


def _process_track_chunk_time_region(track_chunk, all_event_ids, all_particle_ids, all_particle_pts, all_particle_etas, true_hit_mask):
    """Worker function to process a chunk of tracks for time region filtering.
    Uses same criteria as ML region (no time-based track filtering):
    - pt >= 5.0 GeV
    - |eta| <= 2.7
    - >= 3 hits per track.

    Note: Time filtering is applied at the hit level, not track level.

    Args:
        track_chunk: List of (event_id, particle_id) tuples to process
        all_event_ids: Array of event IDs for all hits
        all_particle_ids: Array of particle IDs for all hits
        all_particle_pts: Array of particle pT values for all hits
        all_particle_etas: Array of particle eta values for all hits
        true_hit_mask: Boolean mask for true hits

    Returns:
        Dictionary with qualified tracks and statistics for this chunk
    """
    chunk_stats = {
        "total_tracks_checked": 0,
        "tracks_failed_min_hits": 0,
        "tracks_failed_eta_cuts": 0,
        "tracks_failed_pt_cuts": 0,
        "tracks_passed_all_cuts": 0,
    }

    qualified_tracks = set()

    # Time region criteria (same as ML region - no time constraint at track level)
    TIME_PT_THRESHOLD = 5.0  # GeV
    TIME_ETA_THRESHOLD = 2.7  # |eta| <= 2.7
    TIME_MIN_HITS = 3  # minimum hits per track

    for event_id, particle_id in track_chunk:
        chunk_stats["total_tracks_checked"] += 1

        # Get hits for this specific track
        track_mask = (all_event_ids == event_id) & (all_particle_ids == particle_id) & true_hit_mask
        track_hits = np.sum(track_mask)

        # Time Pre-filter 1: tracks must have at least 3 hits total
        if track_hits < TIME_MIN_HITS:
            chunk_stats["tracks_failed_min_hits"] += 1
            continue

        # Get particle kinematic properties for this track
        track_indices = np.where(track_mask)[0]
        if len(track_indices) == 0:
            chunk_stats["tracks_failed_min_hits"] += 1
            continue

        # Use the first hit to get particle properties (all hits from same particle should have same pt/eta)
        first_hit_idx = track_indices[0]
        track_pt = all_particle_pts[first_hit_idx]
        track_eta = all_particle_etas[first_hit_idx]

        # Time Pre-filter 2: eta acceptance cuts |eta| <= 2.7
        if np.abs(track_eta) > TIME_ETA_THRESHOLD:
            chunk_stats["tracks_failed_eta_cuts"] += 1
            continue

        # Time Pre-filter 3: pt threshold >= 5.0 GeV
        if track_pt < TIME_PT_THRESHOLD:
            chunk_stats["tracks_failed_pt_cuts"] += 1
            continue

        # Track passed all time region criteria (same as ML region)
        qualified_tracks.add((event_id, particle_id))
        chunk_stats["tracks_passed_all_cuts"] += 1

    return {"qualified_tracks": qualified_tracks, "stats": chunk_stats}


class AtlasMuonEvaluatorDataLoader:
    """Evaluation class for ATLAS muon hit filtering using DataLoader."""

    def __init__(self, eval_path, data_dir, config_path, output_dir, max_events=None, min_pt=0.0, max_pt=float("inf")):
        self.eval_path = Path(eval_path)
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.min_pt = min_pt  # Minimum pT threshold for track filtering
        self.max_pt = max_pt  # Maximum pT threshold for track filtering

        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for all tracks, baseline filtered tracks, ml region tracks, time region tracks, and rejected tracks
        self.all_tracks_dir = self.output_dir / "all_tracks"
        self.baseline_filtered_dir = self.output_dir / "baseline_filtered_tracks"
        self.ml_region_dir = self.output_dir / "ml_region"
        self.time_region_dir = self.output_dir / "time_region"
        self.rejected_tracks_dir = self.output_dir / "rejected_tracks"
        self.all_tracks_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_filtered_dir.mkdir(parents=True, exist_ok=True)
        self.ml_region_dir.mkdir(parents=True, exist_ok=True)
        self.time_region_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_tracks_dir.mkdir(parents=True, exist_ok=True)

        self.max_events = max_events

        # Define sensor technologies mapping
        self.technology_mapping = {"MDT": 0, "RPC": 2, "TGC": 3, "STGC": 4, "MM": 5}
        self.technology_names = list(self.technology_mapping.keys())

        # Initialize data module
        self.setup_data_module()

        # Storage for all collected data
        self.all_logits = None
        self.all_true_labels = None
        self.all_particle_pts = None
        self.all_particle_ids = None
        self.all_event_ids = None
        self.all_station_indices = None
        self.all_hit_times = None

    def setup_data_module(self):
        """Initialize the AtlasMuonDataModule with proper configuration."""
        print("Setting up data module...")

        # Load config
        with Path(self.config_path).open() as f:
            config = yaml.safe_load(f)

        # Extract inputs and targets from config
        data_config = config.get("data", {})
        inputs = data_config.get("inputs", {})
        targets = data_config.get("targets", {})

        # Create fresh copies to avoid corruption
        inputs_eval = {k: list(v) for k, v in inputs.items()}
        targets_eval = {k: list(v) for k, v in targets.items()}

        # Set evaluation parameters
        # When max_events is None or -1, use all available events
        num_test_events = self.max_events if self.max_events is not None and self.max_events != -1 else -1

        # Initialize data module following the working example
        # Note: Even for test-only evaluation, we need to set num_train > 0
        self.data_module = AtlasMuonDataModule(
            train_dir=str(self.data_dir),
            val_dir=str(self.data_dir),
            test_dir=str(self.data_dir),
            num_workers=100,  # Use many workers for maximum speed
            num_train=abs(num_test_events) if num_test_events != -1 else 1000,  # Set to positive value
            num_val=abs(num_test_events) if num_test_events != -1 else 1000,  # Set to positive value
            num_test=num_test_events,  # -1 means use all available events
            batch_size=1,
            inputs=inputs_eval,
            targets=targets_eval,
            pin_memory=True,
        )

        # Setup the data module
        self.data_module.setup(stage="test")
        self.test_dataloader = self.data_module.test_dataloader(shuffle=True)

        # print(f"DataLoader setup complete with 100 workers, processing {num_test_events} events")

    def collect_data(self):  # verified: (check!)
        """Collect all data for analysis using the DataLoader with memory optimizations."""
        print("Collecting data from all events using DataLoader (memory optimized)...")

        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")

        # First, let's check what's in the evaluation file
        with h5py.File(self.eval_path, "r") as eval_file:
            eval_keys = list(eval_file.keys())
            print(f"Evaluation file contains {len(eval_keys)} events")

        # Pre-allocate storage with estimated sizes (more memory efficient)
        estimated_hits = min(self.max_events * 1000 if self.max_events and self.max_events > 0 else 10000000, 50000000)

        # Use memory-efficient dtypes
        try:
            # Pre-allocate arrays with conservative size estimates
            all_logits = np.zeros(estimated_hits, dtype=np.float32)
            all_true_labels = np.zeros(estimated_hits, dtype=np.bool_)
            all_particle_pts = np.zeros(estimated_hits, dtype=np.float32)
            all_particle_eta = np.zeros(estimated_hits, dtype=np.float32)
            all_particle_phi = np.zeros(estimated_hits, dtype=np.float32)
            all_particle_ids = np.zeros(estimated_hits, dtype=np.int32)
            all_particle_technology = np.zeros(estimated_hits, dtype=np.int8)
            all_event_ids = np.zeros(estimated_hits, dtype=np.int32)
            all_station_indices = np.zeros(estimated_hits, dtype=np.int32)
            all_hit_times = np.zeros(estimated_hits, dtype=np.float32)

            current_idx = 0
            events_processed = 0
            events_attempted = 0

        except MemoryError:
            print("ERROR: Not enough memory to pre-allocate arrays. Using list-based approach.")
            # Fall back to lists if pre-allocation fails
            all_logits = []
            all_true_labels = []
            all_particle_pts = []
            all_particle_eta = []
            all_particle_phi = []
            all_particle_ids = []
            all_particle_technology = []
            all_event_ids = []
            all_station_indices = []
            all_hit_times = []
            current_idx = None

        try:
            with h5py.File(self.eval_path, "r") as eval_file:
                for batch_idx, batch_data in enumerate(tqdm(self.test_dataloader, desc="Processing events")):
                    events_attempted += 1

                    # Only break if max_events is explicitly set (not None or -1)
                    if self.max_events is not None and self.max_events > 0 and events_processed >= self.max_events:
                        break

                    # Memory monitoring every 1000 events
                    if events_processed % 1000 == 0 and events_processed > 0:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        print(f"Processed {events_processed} events. Memory: {current_memory:.1f} MB (+{current_memory - initial_memory:.1f} MB)")

                        # Force garbage collection if memory usage is high
                        if current_memory > 8000:  # 8GB threshold
                            gc.collect()

                    try:
                        # Unpack batch data
                        if len(batch_data) == 2:
                            inputs_batch, targets_batch = batch_data
                        else:
                            print(f"Unexpected batch data structure: {len(batch_data)} elements")
                            continue

                        # Extract event index
                        if "sample_id" not in targets_batch:
                            print(f"Warning: sample_id not found in targets, skipping batch {batch_idx}")
                            continue

                        event_idx = targets_batch["sample_id"][0].item()

                        # Load predictions for this event
                        if str(event_idx) not in eval_file:
                            continue

                        # Get hit logits from evaluation file
                        hit_logits = eval_file[f"{event_idx}/outputs/final/hit_filter/hit_logit"][0].astype(np.float32)

                        # Get truth labels from DataLoader
                        if "hit_on_valid_particle" not in targets_batch:
                            continue

                        true_labels = targets_batch["hit_on_valid_particle"][0].numpy().astype(np.bool_)
                        hit_particle_ids = inputs_batch["plotting_spacePoint_truthLink"][0].numpy().astype(np.int32)
                        hit_technologies = inputs_batch["hit_spacePoint_technology"][0].numpy().astype(np.int8)
                        hit_station_indices = inputs_batch["hit_spacePoint_stationIndex"][0].numpy().astype(np.int32)
                        hit_times = inputs_batch["hit_spacePoint_time"][0].numpy().astype(np.float32)
                        # print ("This is hit_station_indices: ", hit_station_indices / 0.1)

                        # Verify shapes match
                        n_hits = len(hit_logits)
                        if (
                            n_hits != len(true_labels)
                            or n_hits != len(hit_particle_ids)
                            or n_hits != len(hit_station_indices)
                            or n_hits != len(hit_times)
                        ):
                            print(f"Warning: Shape mismatch in event {event_idx}")
                            continue

                        # Get particle pt values
                        if "particle_truthMuon_pt" not in targets_batch:
                            continue

                        particle_pts = targets_batch["particle_truthMuon_pt"][0].numpy().astype(np.float32)
                        particle_etas = targets_batch["particle_truthMuon_eta"][0].numpy().astype(np.float32)
                        particle_phis = targets_batch["particle_truthMuon_phi"][0].numpy().astype(np.float32)

                        # Map hits to particle pt values (vectorized for speed)
                        hit_pts = np.full(n_hits, -1.0, dtype=np.float32)  # Default for noise hits
                        hit_etas = np.full(n_hits, -1.0, dtype=np.float32)
                        hit_phis = np.full(n_hits, -1.0, dtype=np.float32)

                        # Vectorized mapping using advanced indexing
                        unique_particle_ids = np.unique(hit_particle_ids)
                        valid_particle_ids = unique_particle_ids[unique_particle_ids >= 0]  # Skip -1 (noise)

                        for idx, particle_id in enumerate(valid_particle_ids):
                            if idx < len(particle_pts):  # Safety check
                                hit_mask = hit_particle_ids == particle_id
                                hit_pts[hit_mask] = particle_pts[idx]
                                hit_etas[hit_mask] = particle_etas[idx]
                                hit_phis[hit_mask] = particle_phis[idx]

                        # Store data efficiently
                        if current_idx is not None:  # Pre-allocated arrays
                            if current_idx + n_hits >= len(all_logits):
                                # Resize arrays if needed (rare case)
                                new_size = max(len(all_logits) * 2, current_idx + n_hits + 100000)
                                all_logits = np.resize(all_logits, new_size)
                                all_true_labels = np.resize(all_true_labels, new_size)
                                all_particle_pts = np.resize(all_particle_pts, new_size)
                                all_particle_eta = np.resize(all_particle_eta, new_size)
                                all_particle_phi = np.resize(all_particle_phi, new_size)
                                all_particle_ids = np.resize(all_particle_ids, new_size)
                                all_particle_technology = np.resize(all_particle_technology, new_size)
                                all_event_ids = np.resize(all_event_ids, new_size)
                                all_station_indices = np.resize(all_station_indices, new_size)
                                all_hit_times = np.resize(all_hit_times, new_size)

                            # Copy data to pre-allocated arrays
                            all_logits[current_idx : current_idx + n_hits] = hit_logits
                            all_true_labels[current_idx : current_idx + n_hits] = true_labels
                            all_particle_pts[current_idx : current_idx + n_hits] = hit_pts
                            all_particle_eta[current_idx : current_idx + n_hits] = hit_etas
                            all_particle_phi[current_idx : current_idx + n_hits] = hit_phis
                            all_particle_ids[current_idx : current_idx + n_hits] = hit_particle_ids
                            all_particle_technology[current_idx : current_idx + n_hits] = hit_technologies
                            all_event_ids[current_idx : current_idx + n_hits] = event_idx
                            all_station_indices[current_idx : current_idx + n_hits] = hit_station_indices
                            all_hit_times[current_idx : current_idx + n_hits] = hit_times
                            current_idx += n_hits
                        else:
                            # Fall back to list append
                            all_logits.append(hit_logits)
                            all_true_labels.append(true_labels)
                            all_particle_pts.append(hit_pts)
                            all_particle_eta.append(hit_etas)
                            all_particle_phi.append(hit_phis)
                            all_particle_ids.append(hit_particle_ids)
                            all_particle_technology.append(hit_technologies)
                            all_event_ids.append(np.full(n_hits, event_idx, dtype=np.int32))
                            all_station_indices.append(hit_station_indices)
                            all_hit_times.append(hit_times)

                        events_processed += 1

                    except (ValueError, KeyError, OSError, RuntimeError) as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        continue

        except (ValueError, KeyError, OSError, RuntimeError) as e:
            print(f"Error during data collection: {e}")
            traceback.print_exc()
            return False

        print(f"\nDataLoader provided {events_attempted} batches, successfully processed {events_processed} events")

        if events_processed == 0:
            print("ERROR: No events were successfully processed!")
            return False

        # Convert to final numpy arrays
        if current_idx is not None:  # Pre-allocated arrays
            self.all_logits = all_logits[:current_idx]
            self.all_true_labels = all_true_labels[:current_idx]
            self.all_particle_pts = all_particle_pts[:current_idx]
            self.all_particle_etas = all_particle_eta[:current_idx]
            self.all_particle_phis = all_particle_phi[:current_idx]
            self.all_particle_ids = all_particle_ids[:current_idx]
            self.all_particle_technology = all_particle_technology[:current_idx]
            self.all_event_ids = all_event_ids[:current_idx]
            self.all_station_indices = all_station_indices[:current_idx]
            self.all_hit_times = all_hit_times[:current_idx]
        else:
            # Concatenate lists
            self.all_logits = np.concatenate(all_logits) if all_logits else np.array([])
            self.all_true_labels = np.concatenate(all_true_labels) if all_true_labels else np.array([], dtype=bool)
            self.all_particle_pts = np.concatenate(all_particle_pts) if all_particle_pts else np.array([])
            self.all_particle_etas = np.concatenate(all_particle_eta) if all_particle_eta else np.array([])
            self.all_particle_phis = np.concatenate(all_particle_phi) if all_particle_phi else np.array([])
            self.all_particle_ids = np.concatenate(all_particle_ids) if all_particle_ids else np.array([])
            self.all_particle_technology = np.concatenate(all_particle_technology) if all_particle_technology else np.array([])
            self.all_event_ids = np.concatenate(all_event_ids) if all_event_ids else np.array([])
            self.all_station_indices = np.concatenate(all_station_indices) if all_station_indices else np.array([])
            self.all_hit_times = np.concatenate(all_hit_times) if all_hit_times else np.array([])

        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"\nData collection complete! Final memory: {final_memory:.1f} MB (+{final_memory - initial_memory:.1f} MB)")
        print(f"Events processed: {events_processed}")
        print(f"Total hits: {len(self.all_logits):,}")
        print(f"True hits: {np.sum(self.all_true_labels):,}")
        print(f"Noise hits: {np.sum(~self.all_true_labels):,}")
        print(f"Valid particle hits (pt > 0): {np.sum(self.all_particle_pts > 0):,}")

        # Print time statistics for hits
        valid_time_mask = self.all_hit_times > -999  # Assuming -999 or similar indicates invalid times
        if np.any(valid_time_mask):
            valid_times = self.all_hit_times[valid_time_mask]
            print("\nTime statistics for valid hits:")
            print(f"  Min: {np.min(valid_times):.6f}")
            print(f"  Max: {np.max(valid_times):.6f}")
            print(f"  Mean: {np.mean(valid_times):.6f}")
            print(f"  Median: {np.median(valid_times):.6f}")
            time_threshold = 2000 * 0.00001  # 0.02
            hits_below_time_threshold = np.sum(valid_times < time_threshold)
            print(
                f"  Hits below time threshold ({time_threshold:.6f}): {hits_below_time_threshold:,} ({hits_below_time_threshold / len(valid_times) * 100:.1f}%)"
            )

        # Apply pT filter if specified
        if self.min_pt > 0 or self.max_pt < float("inf"):
            filter_desc = []
            if self.min_pt > 0:
                filter_desc.append(f"pT >= {self.min_pt} GeV")
            if self.max_pt < float("inf"):
                filter_desc.append(f"pT <= {self.max_pt} GeV")
            print(f"\nApplying pT filter: excluding all hits from tracks with {' and '.join(filter_desc)}...")
            # self._apply_pt_filter()
            print("After pT filter:")
            print(f"  Total hits: {len(self.all_logits):,}")
            print(f"  True hits: {np.sum(self.all_true_labels):,}")
            print(f"  Noise hits: {np.sum(~self.all_true_labels):,}")
            print(f"  Valid particle hits (pt > 0): {np.sum(self.all_particle_pts > 0):,}")

        # Print pt statistics for valid particles
        valid_pt_mask = self.all_particle_pts > 0
        if np.any(valid_pt_mask):
            valid_pts = self.all_particle_pts[valid_pt_mask]
            print("\nPT statistics for valid particles:")
            print(f"  Min: {np.min(valid_pts):.1f} GeV")
            print(f"  Max: {np.max(valid_pts):.1f} GeV")
            print(f"  Mean: {np.mean(valid_pts):.1f} GeV")
            print(f"  Median: {np.median(valid_pts):.1f} GeV")

        # Force garbage collection
        gc.collect()

        return True

    def create_baseline_track_filter(self):
        """Create filter masks for baseline evaluation that includes:
        - ALL noise hits (maintains realistic background for both categories)
        - True hits from tracks meeting baseline requirements (baseline category)
        - True hits from tracks NOT meeting baseline requirements (rejected category).

        Baseline requirements:
          * At least 3 different stations
          * At least 3 hits per station (meaning >= 9 hits total per track)
          * |eta| >= 0.1 and |eta| <= 2.7 (detector acceptance region)
          * pt >= some GeV (minimum pt threshold)

        Note: If global pT filtering was applied earlier (--min_pt), only tracks
        above that threshold will be considered here.

        This approach maintains signal-to-noise ratio while allowing comparison
        between high-quality and low-quality tracks.

        Returns:
            baseline_mask: Boolean array for hits in baseline evaluation
            rejected_mask: Boolean array for hits in rejected tracks evaluation
            stats: Dictionary with detailed filtering statistics
        """
        print("Creating baseline track filter (>=3 stations, >=3 hits per station, eta cuts, pt cuts)...")
        print("Strategy: Keep ALL noise hits + true hits from baseline-qualified tracks only")

        # Only consider true hits (noise hits are not part of tracks)
        true_hit_mask = self.all_true_labels
        print(f"Total true hits available for track evaluation: {np.sum(true_hit_mask):,}")
        # Get unique combinations of (event_id, particle_id) for valid tracks
        valid_event_particle_combinations = np.unique(
            np.column_stack([self.all_event_ids[true_hit_mask], self.all_particle_ids[true_hit_mask]]), axis=0
        )
        print()
        print(f"Found {len(valid_event_particle_combinations)} unique tracks with truth hits")
        baseline_qualified_tracks = set()

        # Track statistics for detailed reporting
        stats = {
            "total_tracks_checked": 0,
            "tracks_failed_min_hits": 0,
            "tracks_failed_eta_cuts": 0,
            "tracks_failed_pt_cuts": 0,
            "tracks_failed_station_cuts": 0,
            "tracks_passed_all_cuts": 0,
        }

        # Parallel processing of tracks
        print(f"Processing {len(valid_event_particle_combinations)} tracks using parallel workers...")

        # Determine optimal number of workers
        n_workers = min(mp.cpu_count(), max(1, len(valid_event_particle_combinations) // 100))
        n_workers = min(n_workers, 100)  # Cap at 100 to avoid excessive overhead
        print(f"Using {n_workers} parallel workers")

        # Split tracks into chunks for parallel processing
        chunk_size = max(1, len(valid_event_particle_combinations) // n_workers)
        track_chunks = [valid_event_particle_combinations[i : i + chunk_size] for i in range(0, len(valid_event_particle_combinations), chunk_size)]

        print(f"Split tracks into {len(track_chunks)} chunks (avg size: {chunk_size})")

        # Create worker function with pre-bound arguments
        worker_fn = partial(
            _process_track_chunk,
            all_event_ids=self.all_event_ids,
            all_particle_ids=self.all_particle_ids,
            all_particle_pts=self.all_particle_pts,
            all_particle_etas=self.all_particle_etas,
            all_station_indices=self.all_station_indices,
            true_hit_mask=true_hit_mask,
            min_pt=self.min_pt,
        )

        # Process tracks in parallel
        baseline_qualified_tracks = set()
        with mp.Pool(processes=n_workers) as pool:
            # Use tqdm to show progress
            results = list(tqdm(pool.imap(worker_fn, track_chunks), total=len(track_chunks), desc="Processing track chunks"))

        # Aggregate results from all workers
        for result in results:
            baseline_qualified_tracks.update(result["qualified_tracks"])
            for key in stats:
                stats[key] += result["stats"][key]

        # Print detailed statistics
        print("Baseline filtering results:")
        print(f"  Total tracks checked: {stats['total_tracks_checked']}")
        print(
            f"  Failed minimum hits (>=9): {stats['tracks_failed_min_hits']} ({stats['tracks_failed_min_hits'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed eta cuts (0.1 <= |eta| <= 2.7): {stats['tracks_failed_eta_cuts']} ({stats['tracks_failed_eta_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed pt cuts (pt >= {self.min_pt} GeV): {stats['tracks_failed_pt_cuts']} ({stats['tracks_failed_pt_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed station cuts (>=3 stations, >=3 hits/station): {stats['tracks_failed_station_cuts']} ({stats['tracks_failed_station_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Tracks passing all cuts: {stats['tracks_passed_all_cuts']} ({stats['tracks_passed_all_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )

        # Create masks for hits to include in baseline and rejected evaluations
        # Strategy: Include all noise hits + only true hits from respective track categories
        print("Creating hit masks for baseline and rejected tracks...")
        baseline_hit_mask = np.zeros(len(self.all_logits), dtype=bool)
        rejected_hit_mask = np.zeros(len(self.all_logits), dtype=bool)

        # Include ALL noise hits in both categories (these don't belong to any track)
        noise_hit_mask = ~true_hit_mask
        baseline_hit_mask |= noise_hit_mask
        rejected_hit_mask |= noise_hit_mask

        # Create sets for baseline and rejected tracks
        baseline_qualified_track_set = set(baseline_qualified_tracks)
        all_track_set = {tuple(track) for track in valid_event_particle_combinations}
        rejected_track_set = all_track_set - baseline_qualified_track_set

        print(f"Creating masks for {len(baseline_qualified_tracks)} baseline tracks and {len(rejected_track_set)} rejected tracks...")

        # OPTIMIZED: Memory-efficient chunked mask creation for large datasets
        # Create arrays for efficient vectorized comparison
        hit_event_ids = self.all_event_ids[true_hit_mask]
        hit_particle_ids = self.all_particle_ids[true_hit_mask]
        true_hit_indices = np.where(true_hit_mask)[0]

        # Process tracks in chunks to avoid excessive memory usage
        chunk_size = min(1000, max(100, len(baseline_qualified_tracks) // 10))

        # Convert track sets to arrays for vectorized operations
        if baseline_qualified_tracks:
            baseline_tracks_array = np.array(list(baseline_qualified_tracks))
            print(f"Processing {len(baseline_tracks_array)} baseline tracks in chunks of {chunk_size}...")

            for i in range(0, len(baseline_tracks_array), chunk_size):
                chunk_tracks = baseline_tracks_array[i : i + chunk_size]
                chunk_event_ids = chunk_tracks[:, 0]
                chunk_particle_ids = chunk_tracks[:, 1]

                # Vectorized comparison for this chunk
                baseline_matches = (hit_event_ids[:, np.newaxis] == chunk_event_ids[np.newaxis, :]) & (
                    hit_particle_ids[:, np.newaxis] == chunk_particle_ids[np.newaxis, :]
                )
                baseline_hit_indices = true_hit_indices[np.any(baseline_matches, axis=1)]
                baseline_hit_mask[baseline_hit_indices] = True

        if rejected_track_set:
            rejected_tracks_array = np.array(list(rejected_track_set))
            print(f"Processing {len(rejected_tracks_array)} rejected tracks in chunks of {chunk_size}...")

            for i in range(0, len(rejected_tracks_array), chunk_size):
                chunk_tracks = rejected_tracks_array[i : i + chunk_size]
                chunk_event_ids = chunk_tracks[:, 0]
                chunk_particle_ids = chunk_tracks[:, 1]

                # Vectorized comparison for this chunk
                rejected_matches = (hit_event_ids[:, np.newaxis] == chunk_event_ids[np.newaxis, :]) & (
                    hit_particle_ids[:, np.newaxis] == chunk_particle_ids[np.newaxis, :]
                )
                rejected_hit_indices = true_hit_indices[np.any(rejected_matches, axis=1)]
                rejected_hit_mask[rejected_hit_indices] = True

        # Calculate statistics for both categories
        baseline_hit_count = np.sum(baseline_hit_mask)
        baseline_true_hits = np.sum(baseline_hit_mask & true_hit_mask)
        baseline_noise_hits = np.sum(baseline_hit_mask & ~true_hit_mask)

        rejected_hit_count = np.sum(rejected_hit_mask)
        rejected_true_hits = np.sum(rejected_hit_mask & true_hit_mask)
        rejected_noise_hits = np.sum(rejected_hit_mask & ~true_hit_mask)

        total_hits = len(self.all_logits)

        print(f"  Baseline hits: {baseline_hit_count:,} / {total_hits:,} ({baseline_hit_count / total_hits * 100:.1f}%)")
        print(f"  Baseline true hits: {baseline_true_hits:,}")
        print(f"  Baseline noise hits: {baseline_noise_hits:,}")
        print(
            f"  Baseline signal/noise ratio: {baseline_true_hits / baseline_noise_hits:.4f}"
            if baseline_noise_hits > 0
            else "  Baseline signal/noise ratio: inf"
        )

        print(f"  Rejected hits: {rejected_hit_count:,} / {total_hits:,} ({rejected_hit_count / total_hits * 100:.1f}%)")
        print(f"  Rejected true hits: {rejected_true_hits:,}")
        print(f"  Rejected noise hits: {rejected_noise_hits:,}")
        print(
            f"  Rejected signal/noise ratio: {rejected_true_hits / rejected_noise_hits:.4f}"
            if rejected_noise_hits > 0
            else "  Rejected signal/noise ratio: inf"
        )

        # Additional statistics for the summary
        stats["baseline_hit_count"] = baseline_hit_count
        stats["baseline_true_hits"] = baseline_true_hits
        stats["baseline_noise_hits"] = baseline_noise_hits
        stats["rejected_hit_count"] = rejected_hit_count
        stats["rejected_true_hits"] = rejected_true_hits
        stats["rejected_noise_hits"] = rejected_noise_hits
        stats["total_hits"] = total_hits
        stats["rejected_tracks"] = len(rejected_track_set)

        return baseline_hit_mask, rejected_hit_mask, stats

    def create_ml_region_track_filter(self):
        """Create filter masks for ML region evaluation that includes:
        - ALL noise hits (maintains realistic background for both categories)
        - True hits from tracks meeting ML region requirements (ml_region category)
        - True hits from tracks NOT meeting ML region requirements (rejected category).

        ML region requirements (matching prep_events_multiprocess.py defaults):
          * pt >= 5.0 GeV
          * |eta| <= 2.7 (detector acceptance region)
          * >= 3 hits per track (minimum viable track)

        This approach uses the same filtering criteria as the data preprocessing
        to create a "ML training region" for evaluation.

        Returns:
            ml_region_mask: Boolean array for hits in ML region evaluation
            rejected_mask: Boolean array for hits in rejected tracks evaluation
            stats: Dictionary with detailed filtering statistics
        """
        print("Creating ML region track filter (pt >= 5.0 GeV, |eta| <= 2.7, >= 3 hits)...")
        print("Strategy: Keep ALL noise hits + true hits from ML region-qualified tracks only")

        # Only consider true hits (noise hits are not part of tracks)
        true_hit_mask = self.all_true_labels
        print(f"Total true hits available for track evaluation: {np.sum(true_hit_mask):,}")

        # Get unique combinations of (event_id, particle_id) for valid tracks
        valid_event_particle_combinations = np.unique(
            np.column_stack([self.all_event_ids[true_hit_mask], self.all_particle_ids[true_hit_mask]]), axis=0
        )
        print(f"Found {len(valid_event_particle_combinations)} unique tracks with truth hits")

        # Track statistics for detailed reporting
        stats = {
            "total_tracks_checked": 0,
            "tracks_failed_min_hits": 0,
            "tracks_failed_eta_cuts": 0,
            "tracks_failed_pt_cuts": 0,
            "tracks_passed_all_cuts": 0,
        }

        # Parallel processing of tracks
        print(f"Processing {len(valid_event_particle_combinations)} tracks using parallel workers...")

        # Determine optimal number of workers
        n_workers = min(mp.cpu_count(), max(1, len(valid_event_particle_combinations) // 100))
        n_workers = min(n_workers, 100)  # Cap at 100 to avoid excessive overhead
        print(f"Using {n_workers} parallel workers")

        # Split tracks into chunks for parallel processing
        chunk_size = max(1, len(valid_event_particle_combinations) // n_workers)
        track_chunks = [valid_event_particle_combinations[i : i + chunk_size] for i in range(0, len(valid_event_particle_combinations), chunk_size)]

        print(f"Split tracks into {len(track_chunks)} chunks (avg size: {chunk_size})")

        # Create worker function with pre-bound arguments
        worker_fn = partial(
            _process_track_chunk_ml_region,
            all_event_ids=self.all_event_ids,
            all_particle_ids=self.all_particle_ids,
            all_particle_pts=self.all_particle_pts,
            all_particle_etas=self.all_particle_etas,
            true_hit_mask=true_hit_mask,
        )

        # Process tracks in parallel
        ml_region_qualified_tracks = set()
        with mp.Pool(processes=n_workers) as pool:
            # Use tqdm to show progress
            results = list(tqdm(pool.imap(worker_fn, track_chunks), total=len(track_chunks), desc="Processing ML region track chunks"))

        # Aggregate results from all workers
        for result in results:
            ml_region_qualified_tracks.update(result["qualified_tracks"])
            for key in stats:
                stats[key] += result["stats"][key]

        # Print detailed statistics
        print("ML region filtering results:")
        print(f"  Total tracks checked: {stats['total_tracks_checked']}")
        print(
            f"  Failed minimum hits (>= 3): {stats['tracks_failed_min_hits']} ({stats['tracks_failed_min_hits'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed eta cuts (|eta| <= 2.7): {stats['tracks_failed_eta_cuts']} ({stats['tracks_failed_eta_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed pt cuts (pt >= 5.0 GeV): {stats['tracks_failed_pt_cuts']} ({stats['tracks_failed_pt_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Tracks passing all cuts: {stats['tracks_passed_all_cuts']} ({stats['tracks_passed_all_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )

        # Create masks for hits to include in ML region and rejected evaluations
        print("Creating hit masks for ML region and rejected tracks...")
        ml_region_hit_mask = np.zeros(len(self.all_logits), dtype=bool)
        rejected_hit_mask = np.zeros(len(self.all_logits), dtype=bool)

        # Include ALL noise hits in both categories (these don't belong to any track)
        noise_hit_mask = ~true_hit_mask
        ml_region_hit_mask |= noise_hit_mask
        rejected_hit_mask |= noise_hit_mask

        # Create sets for ML region and rejected tracks
        ml_region_qualified_track_set = set(ml_region_qualified_tracks)
        all_track_set = {tuple(track) for track in valid_event_particle_combinations}
        rejected_track_set = all_track_set - ml_region_qualified_track_set

        print(f"Creating masks for {len(ml_region_qualified_tracks)} ML region tracks and {len(rejected_track_set)} rejected tracks...")

        # OPTIMIZED: Memory-efficient chunked mask creation for large datasets
        # Create arrays for efficient vectorized comparison
        hit_event_ids = self.all_event_ids[true_hit_mask]
        hit_particle_ids = self.all_particle_ids[true_hit_mask]
        true_hit_indices = np.where(true_hit_mask)[0]

        # Process tracks in chunks to avoid excessive memory usage
        chunk_size = min(1000, max(100, len(ml_region_qualified_tracks) // 10))

        # Convert track sets to arrays for vectorized operations
        if ml_region_qualified_tracks:
            ml_region_tracks_array = np.array(list(ml_region_qualified_tracks))
            print(f"Processing {len(ml_region_tracks_array)} ML region tracks in chunks of {chunk_size}...")

            for i in range(0, len(ml_region_tracks_array), chunk_size):
                chunk_tracks = ml_region_tracks_array[i : i + chunk_size]
                chunk_event_ids = chunk_tracks[:, 0]
                chunk_particle_ids = chunk_tracks[:, 1]

                # Vectorized comparison for this chunk
                ml_region_matches = (hit_event_ids[:, np.newaxis] == chunk_event_ids[np.newaxis, :]) & (
                    hit_particle_ids[:, np.newaxis] == chunk_particle_ids[np.newaxis, :]
                )
                ml_region_hit_indices = true_hit_indices[np.any(ml_region_matches, axis=1)]
                ml_region_hit_mask[ml_region_hit_indices] = True

        if rejected_track_set:
            rejected_tracks_array = np.array(list(rejected_track_set))
            print(f"Processing {len(rejected_tracks_array)} rejected tracks in chunks of {chunk_size}...")

            for i in range(0, len(rejected_tracks_array), chunk_size):
                chunk_tracks = rejected_tracks_array[i : i + chunk_size]
                chunk_event_ids = chunk_tracks[:, 0]
                chunk_particle_ids = chunk_tracks[:, 1]

                # Vectorized comparison for this chunk
                rejected_matches = (hit_event_ids[:, np.newaxis] == chunk_event_ids[np.newaxis, :]) & (
                    hit_particle_ids[:, np.newaxis] == chunk_particle_ids[np.newaxis, :]
                )
                rejected_hit_indices = true_hit_indices[np.any(rejected_matches, axis=1)]
                rejected_hit_mask[rejected_hit_indices] = True

        # Calculate statistics for both categories
        ml_region_hit_count = np.sum(ml_region_hit_mask)
        ml_region_true_hits = np.sum(ml_region_hit_mask & true_hit_mask)
        ml_region_noise_hits = np.sum(ml_region_hit_mask & ~true_hit_mask)

        rejected_hit_count = np.sum(rejected_hit_mask)
        rejected_true_hits = np.sum(rejected_hit_mask & true_hit_mask)
        rejected_noise_hits = np.sum(rejected_hit_mask & ~true_hit_mask)

        total_hits = len(self.all_logits)

        print(f"  ML region hits: {ml_region_hit_count:,} / {total_hits:,} ({ml_region_hit_count / total_hits * 100:.1f}%)")
        print(f"  ML region true hits: {ml_region_true_hits:,}")
        print(f"  ML region noise hits: {ml_region_noise_hits:,}")
        print(
            f"  ML region signal/noise ratio: {ml_region_true_hits / ml_region_noise_hits:.4f}"
            if ml_region_noise_hits > 0
            else "  ML region signal/noise ratio: inf"
        )

        print(f"  Rejected hits: {rejected_hit_count:,} / {total_hits:,} ({rejected_hit_count / total_hits * 100:.1f}%)")
        print(f"  Rejected true hits: {rejected_true_hits:,}")
        print(f"  Rejected noise hits: {rejected_noise_hits:,}")
        print(
            f"  Rejected signal/noise ratio: {rejected_true_hits / rejected_noise_hits:.4f}"
            if rejected_noise_hits > 0
            else "  Rejected signal/noise ratio: inf"
        )

        # Additional statistics for the summary
        stats["ml_region_hit_count"] = ml_region_hit_count
        stats["ml_region_true_hits"] = ml_region_true_hits
        stats["ml_region_noise_hits"] = ml_region_noise_hits
        stats["rejected_hit_count"] = rejected_hit_count
        stats["rejected_true_hits"] = rejected_true_hits
        stats["rejected_noise_hits"] = rejected_noise_hits
        stats["total_hits"] = total_hits
        stats["rejected_tracks"] = len(rejected_track_set)

        return ml_region_hit_mask, rejected_hit_mask, stats

    def create_time_region_track_filter(self):
        """Create filter masks for time region evaluation that includes:
        - ALL noise hits with time < 0.02 (removes high-time noise)
        - True hits from tracks meeting ML region requirements (same as ML region).

        Time region requirements (same as ML region + hit-level time filtering):
          * pt >= 5.0 GeV
          * |eta| <= 2.7 (detector acceptance region)
          * >= 3 hits per track (minimum viable track)
          * Hit-level filtering: only keep hits with time < 2000*0.00001 (0.02)

        This approach removes high-time noise hits while keeping all qualifying tracks,
        improving signal-to-noise ratio based on the discriminating time feature.

        Returns:
            time_region_mask: Boolean array for hits in time region evaluation
            rejected_mask: Boolean array for hits in rejected tracks evaluation (same as ML region)
            stats: Dictionary with detailed filtering statistics
        """
        print("Creating time region hit filter (ML region tracks + time < 0.02 hit filtering)...")
        print("Strategy: Keep tracks meeting ML criteria + remove noise hits with time >= 0.02")

        # Apply hit-level time filtering first
        TIME_THRESHOLD = 2000 * 0.00001  # 0.02 - discriminating time threshold
        time_filtered_hit_mask = self.all_hit_times < TIME_THRESHOLD

        print(f"Hit-level time filtering (time < {TIME_THRESHOLD:.6f}):")
        print(f"  Total hits before time filter: {len(self.all_hit_times):,}")
        print(
            f"  Hits passing time filter: {np.sum(time_filtered_hit_mask):,} ({np.sum(time_filtered_hit_mask) / len(self.all_hit_times) * 100:.1f}%)"
        )
        print(
            f"  Hits removed by time filter: {np.sum(~time_filtered_hit_mask):,} ({np.sum(~time_filtered_hit_mask) / len(self.all_hit_times) * 100:.1f}%)"
        )

        # Separate impact on true hits vs noise hits
        true_hit_mask = self.all_true_labels
        noise_hit_mask = ~true_hit_mask

        time_filtered_true_hits = np.sum(time_filtered_hit_mask & true_hit_mask)
        time_filtered_noise_hits = np.sum(time_filtered_hit_mask & noise_hit_mask)
        removed_true_hits = np.sum(~time_filtered_hit_mask & true_hit_mask)
        removed_noise_hits = np.sum(~time_filtered_hit_mask & noise_hit_mask)

        print(
            f"  True hits passing time filter: {time_filtered_true_hits:,} / {np.sum(true_hit_mask):,} ({time_filtered_true_hits / np.sum(true_hit_mask) * 100:.1f}%)"
        )
        print(
            f"  Noise hits passing time filter: {time_filtered_noise_hits:,} / {np.sum(noise_hit_mask):,} ({time_filtered_noise_hits / np.sum(noise_hit_mask) * 100:.1f}%)"
        )
        print(f"  True hits removed by time filter: {removed_true_hits:,}")
        print(f"  Noise hits removed by time filter: {removed_noise_hits:,}")

        # Only consider true hits for track evaluation (but after time filtering)
        true_hit_mask_after_time_filter = true_hit_mask & time_filtered_hit_mask
        print(f"True hits available for track evaluation after time filter: {np.sum(true_hit_mask_after_time_filter):,}")

        # Get unique combinations of (event_id, particle_id) for valid tracks (after time filtering)
        valid_event_particle_combinations = np.unique(
            np.column_stack([self.all_event_ids[true_hit_mask_after_time_filter], self.all_particle_ids[true_hit_mask_after_time_filter]]), axis=0
        )
        print(f"Found {len(valid_event_particle_combinations)} unique tracks with truth hits after time filtering")

        # Track statistics for detailed reporting (using same logic as ML region)
        stats = {
            "total_tracks_checked": 0,
            "tracks_failed_min_hits": 0,
            "tracks_failed_eta_cuts": 0,
            "tracks_failed_pt_cuts": 0,
            "tracks_passed_all_cuts": 0,
            "hits_removed_by_time_filter": np.sum(~time_filtered_hit_mask),
            "true_hits_removed_by_time_filter": removed_true_hits,
            "noise_hits_removed_by_time_filter": removed_noise_hits,
        }

        # Parallel processing of tracks (same as ML region)
        print(f"Processing {len(valid_event_particle_combinations)} tracks using parallel workers...")

        # Determine optimal number of workers
        n_workers = min(mp.cpu_count(), max(1, len(valid_event_particle_combinations) // 100))
        n_workers = min(n_workers, 100)  # Cap at 100 to avoid excessive overhead
        print(f"Using {n_workers} parallel workers")

        # Split tracks into chunks for parallel processing
        chunk_size = max(1, len(valid_event_particle_combinations) // n_workers)
        track_chunks = [valid_event_particle_combinations[i : i + chunk_size] for i in range(0, len(valid_event_particle_combinations), chunk_size)]

        print(f"Split tracks into {len(track_chunks)} chunks (avg size: {chunk_size})")

        # Create worker function with pre-bound arguments (no time filtering at track level)
        worker_fn = partial(
            _process_track_chunk_time_region,
            all_event_ids=self.all_event_ids,
            all_particle_ids=self.all_particle_ids,
            all_particle_pts=self.all_particle_pts,
            all_particle_etas=self.all_particle_etas,
            true_hit_mask=true_hit_mask_after_time_filter,  # Use time-filtered true hits
        )

        # Process tracks in parallel
        time_region_qualified_tracks = set()
        with mp.Pool(processes=n_workers) as pool:
            # Use tqdm to show progress
            results = list(tqdm(pool.imap(worker_fn, track_chunks), total=len(track_chunks), desc="Processing time region track chunks"))

        # Aggregate results from all workers
        for result in results:
            time_region_qualified_tracks.update(result["qualified_tracks"])
            for key in stats:
                if key in result["stats"]:
                    stats[key] += result["stats"][key]

        # Print detailed statistics
        print("Time region track filtering results (same as ML region):")
        print(f"  Total tracks checked: {stats['total_tracks_checked']}")
        print(
            f"  Failed minimum hits (>= 3): {stats['tracks_failed_min_hits']} ({stats['tracks_failed_min_hits'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed eta cuts (|eta| <= 2.7): {stats['tracks_failed_eta_cuts']} ({stats['tracks_failed_eta_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed pt cuts (pt >= 5.0 GeV): {stats['tracks_failed_pt_cuts']} ({stats['tracks_failed_pt_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Tracks passing all cuts: {stats['tracks_passed_all_cuts']} ({stats['tracks_passed_all_cuts'] / stats['total_tracks_checked'] * 100:.1f}%)"
        )

        # Create masks for hits to include in time region evaluation
        print("Creating hit masks for time region evaluation...")
        time_region_hit_mask = np.zeros(len(self.all_logits), dtype=bool)
        rejected_hit_mask = np.zeros(len(self.all_logits), dtype=bool)

        # Include noise hits that pass time filter
        noise_hits_passing_time = noise_hit_mask & time_filtered_hit_mask
        time_region_hit_mask |= noise_hits_passing_time

        # For rejected tracks, use same logic as ML region but with time-filtered noise
        # This ensures consistent comparison methodology
        rejected_hit_mask |= noise_hits_passing_time

        # Create sets for time region qualified tracks (same as ML region tracks)
        time_region_qualified_track_set = set(time_region_qualified_tracks)
        all_track_set = {tuple(track) for track in valid_event_particle_combinations}
        rejected_track_set = all_track_set - time_region_qualified_track_set

        print(f"Creating masks for {len(time_region_qualified_tracks)} time region tracks and {len(rejected_track_set)} rejected tracks...")

        # Create arrays for efficient vectorized comparison (use time-filtered true hits)
        hit_event_ids = self.all_event_ids[true_hit_mask_after_time_filter]
        hit_particle_ids = self.all_particle_ids[true_hit_mask_after_time_filter]
        true_hit_indices = np.where(true_hit_mask_after_time_filter)[0]

        # Process tracks in chunks to avoid excessive memory usage
        chunk_size = min(1000, max(100, len(time_region_qualified_tracks) // 10))

        # Convert track sets to arrays for vectorized operations
        if time_region_qualified_tracks:
            time_region_tracks_array = np.array(list(time_region_qualified_tracks))
            print(f"Processing {len(time_region_tracks_array)} time region tracks in chunks of {chunk_size}...")

            for i in range(0, len(time_region_tracks_array), chunk_size):
                chunk_tracks = time_region_tracks_array[i : i + chunk_size]
                chunk_event_ids = chunk_tracks[:, 0]
                chunk_particle_ids = chunk_tracks[:, 1]

                # Vectorized comparison for this chunk
                time_region_matches = (hit_event_ids[:, np.newaxis] == chunk_event_ids[np.newaxis, :]) & (
                    hit_particle_ids[:, np.newaxis] == chunk_particle_ids[np.newaxis, :]
                )
                time_region_hit_indices = true_hit_indices[np.any(time_region_matches, axis=1)]
                time_region_hit_mask[time_region_hit_indices] = True

        if rejected_track_set:
            rejected_tracks_array = np.array(list(rejected_track_set))
            print(f"Processing {len(rejected_tracks_array)} rejected tracks in chunks of {chunk_size}...")

            for i in range(0, len(rejected_tracks_array), chunk_size):
                chunk_tracks = rejected_tracks_array[i : i + chunk_size]
                chunk_event_ids = chunk_tracks[:, 0]
                chunk_particle_ids = chunk_tracks[:, 1]

                # Vectorized comparison for this chunk
                rejected_matches = (hit_event_ids[:, np.newaxis] == chunk_event_ids[np.newaxis, :]) & (
                    hit_particle_ids[:, np.newaxis] == chunk_particle_ids[np.newaxis, :]
                )
                rejected_hit_indices = true_hit_indices[np.any(rejected_matches, axis=1)]
                rejected_hit_mask[rejected_hit_indices] = True

        # Calculate statistics for both categories
        time_region_hit_count = np.sum(time_region_hit_mask)
        time_region_true_hits = np.sum(time_region_hit_mask & true_hit_mask)
        time_region_noise_hits = np.sum(time_region_hit_mask & noise_hit_mask)

        rejected_hit_count = np.sum(rejected_hit_mask)
        rejected_true_hits = np.sum(rejected_hit_mask & true_hit_mask)
        rejected_noise_hits = np.sum(rejected_hit_mask & noise_hit_mask)

        total_hits = len(self.all_logits)

        print(f"  Time region hits: {time_region_hit_count:,} / {total_hits:,} ({time_region_hit_count / total_hits * 100:.1f}%)")
        print(f"  Time region true hits: {time_region_true_hits:,}")
        print(f"  Time region noise hits: {time_region_noise_hits:,}")
        print(
            f"  Time region signal/noise ratio: {time_region_true_hits / time_region_noise_hits:.4f}"
            if time_region_noise_hits > 0
            else "  Time region signal/noise ratio: inf"
        )

        print(f"  Rejected hits: {rejected_hit_count:,} / {total_hits:,} ({rejected_hit_count / total_hits * 100:.1f}%)")
        print(f"  Rejected true hits: {rejected_true_hits:,}")
        print(f"  Rejected noise hits: {rejected_noise_hits:,}")
        print(
            f"  Rejected signal/noise ratio: {rejected_true_hits / rejected_noise_hits:.4f}"
            if rejected_noise_hits > 0
            else "  Rejected signal/noise ratio: inf"
        )

        # Additional statistics for the summary
        stats["time_region_hit_count"] = time_region_hit_count
        stats["time_region_true_hits"] = time_region_true_hits
        stats["time_region_noise_hits"] = time_region_noise_hits
        stats["rejected_hit_count"] = rejected_hit_count
        stats["rejected_true_hits"] = rejected_true_hits
        stats["rejected_noise_hits"] = rejected_noise_hits
        stats["total_hits"] = total_hits
        stats["rejected_tracks"] = len(rejected_track_set)

        return time_region_hit_mask, rejected_hit_mask, stats

    def _backup_original_data(self):
        """Backup the original data before applying any filters."""
        self._original_logits = self.all_logits.copy()
        self._original_true_labels = self.all_true_labels.copy()
        self._original_particle_pts = self.all_particle_pts.copy()
        self._original_particle_etas = self.all_particle_etas.copy()
        self._original_particle_phis = self.all_particle_phis.copy()
        self._original_particle_ids = self.all_particle_ids.copy()
        self._original_event_ids = self.all_event_ids.copy()
        self._original_station_indices = self.all_station_indices.copy()
        self._original_hit_times = self.all_hit_times.copy()
        if hasattr(self, "all_particle_technology"):
            self._original_particle_technology = self.all_particle_technology.copy()

    def _restore_original_data(self):
        """Restore the original unfiltered data."""
        self.all_logits = self._original_logits.copy()
        self.all_true_labels = self._original_true_labels.copy()
        self.all_particle_pts = self._original_particle_pts.copy()
        self.all_particle_etas = self._original_particle_etas.copy()
        self.all_particle_phis = self._original_particle_phis.copy()
        self.all_particle_ids = self._original_particle_ids.copy()
        self.all_event_ids = self._original_event_ids.copy()
        self.all_station_indices = self._original_station_indices.copy()
        self.all_hit_times = self._original_hit_times.copy()
        if hasattr(self, "_original_particle_technology"):
            self.all_particle_technology = self._original_particle_technology.copy()

    def _apply_hit_filter(self, hit_mask):
        """Apply a boolean mask to filter the data arrays."""
        self.all_logits = self.all_logits[hit_mask]
        self.all_true_labels = self.all_true_labels[hit_mask]
        self.all_particle_pts = self.all_particle_pts[hit_mask]
        self.all_particle_etas = self.all_particle_etas[hit_mask]
        self.all_particle_phis = self.all_particle_phis[hit_mask]
        self.all_particle_ids = self.all_particle_ids[hit_mask]
        self.all_event_ids = self.all_event_ids[hit_mask]
        self.all_station_indices = self.all_station_indices[hit_mask]
        self.all_hit_times = self.all_hit_times[hit_mask]
        if hasattr(self, "all_particle_technology"):
            self.all_particle_technology = self.all_particle_technology[hit_mask]

    def _apply_pt_filter(self):
        """Apply pT filter to exclude all hits from tracks outside the pT range.
        This preserves all noise hits (which have pt <= 0) but removes true hits
        from tracks outside the specified pT range [min_pt, max_pt].
        """
        filter_applied = False
        filter_description = []

        if self.min_pt > 0:
            filter_description.append(f"pT >= {self.min_pt} GeV")
            filter_applied = True

        if self.max_pt < float("inf"):
            filter_description.append(f"pT <= {self.max_pt} GeV")
            filter_applied = True

        if not filter_applied:
            print("No pT filtering applied")
            return

        print(f"Filtering tracks with {' and '.join(filter_description)}...")

        # Create a mask for hits to keep
        # Keep all noise hits (pT <= 0) and all hits from tracks within pT range
        keep_mask = np.ones(len(self.all_logits), dtype=bool)

        # Only filter true hits (noise hits always have pT <= 0 and should be kept)
        true_hit_mask = self.all_true_labels

        # For true hits, only keep those from tracks within the pT range
        for i in range(len(self.all_logits)):
            if true_hit_mask[i]:  # This is a true hit
                track_pt = self.all_particle_pts[i]
                # Exclude if below minimum pT or above maximum pT (but keep noise hits with pt <= 0)
                if track_pt > 0 and (track_pt < self.min_pt or track_pt > self.max_pt):
                    keep_mask[i] = False

        n_removed = np.sum(~keep_mask)
        n_total = len(self.all_logits)
        print(f"Removing {n_removed:,} hits from tracks outside pT range ({n_removed / n_total * 100:.1f}%)")

        # Apply the filter
        self._apply_hit_filter(keep_mask)

    def plot_roc_curve(self, output_subdir=None):  # verified: (check!)
        """Generate ROC curve with AUC score."""
        print("Generating ROC curve...")

        # Safety checks to prevent NaN AUC
        if len(self.all_logits) == 0:
            print("Warning: No data available for ROC curve calculation")
            return 0.5

        n_true = np.sum(self.all_true_labels)
        n_false = np.sum(~self.all_true_labels)

        if n_true == 0:
            print("Warning: No true positive samples available for ROC curve")
            return 0.5

        if n_false == 0:
            print("Warning: No true negative samples available for ROC curve")
            return 0.5

        # Calculate ROC curve
        try:
            fpr, tpr, _thresholds = roc_curve(self.all_true_labels, self.all_logits)
            roc_auc = auc(fpr, tpr)

            # Additional check for NaN
            if np.isnan(roc_auc):
                print("Warning: AUC calculation resulted in NaN, using default value")
                roc_auc = 0.5

        except (ValueError, KeyError, OSError, RuntimeError) as e:
            print(f"Error calculating ROC curve: {e}")
            roc_auc = 0.5
            fpr = np.array([0, 1])
            tpr = np.array([0, 1])

        print(f"Data statistics: {len(self.all_logits)} hits, {n_true} true, {n_false} false")

        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="green", lw=0.8, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="red", lw=0.8, linestyle="--", label="Random classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - ATLAS Muon Hit Filter")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.1)

        # Save plot to appropriate directory
        save_dir = self.output_dir if output_subdir is None else output_subdir
        output_path = save_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"ROC curve saved to {output_path}")
        print(f"AUC Score: {roc_auc:.4f}")

        return roc_auc

    def calculate_efficiency_by_pt(self, working_point, technology_mask=None):
        """Calculate efficiency binned by truthMuon_pt and overall purity."""
        # Calculate ROC curve to determine threshold for target efficiency
        if technology_mask is not None:
            # Filter data by technology
            logits = self.all_logits[technology_mask]
            true_labels = self.all_true_labels[technology_mask]
            particle_pts = self.all_particle_pts[technology_mask]
        else:
            logits = self.all_logits
            true_labels = self.all_true_labels
            particle_pts = self.all_particle_pts

        _fpr, tpr, thresholds = roc_curve(true_labels, logits)

        # Find threshold that gives the desired efficiency (recall)
        target_efficiency = working_point

        # Find the threshold that achieves the target efficiency
        if not np.any(tpr >= target_efficiency):
            print(f"Warning: Cannot achieve efficiency {target_efficiency}")
            return None, None, None, None

        threshold = thresholds[tpr >= target_efficiency][0]

        # Apply threshold to get predictions
        cut_predictions = logits >= threshold

        # Calculate overall purity for this working point
        total_true_positives = np.sum(true_labels & cut_predictions)
        total_predicted_positives = np.sum(cut_predictions)

        overall_purity = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0

        # Define pt bins
        pt_min, pt_max = self.min_pt, 200.0
        pt_bins = np.linspace(pt_min, pt_max, 21)  # 20 bins
        pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2

        # Calculate efficiency for each bin
        efficiencies = []
        eff_errors = []

        for i in range(len(pt_bins) - 1):
            pt_mask = (particle_pts >= pt_bins[i]) & (particle_pts < pt_bins[i + 1])

            if not np.any(pt_mask):
                efficiencies.append(0.0)
                eff_errors.append(0.0)
                continue

            bin_true_labels = true_labels[pt_mask]
            bin_predictions = cut_predictions[pt_mask]

            # Calculate efficiency (recall)
            true_positives = np.sum(bin_true_labels & bin_predictions)
            total_positives = np.sum(bin_true_labels)

            if total_positives > 0:
                efficiency = true_positives / total_positives
                # Binomial error for efficiency
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_positives)
            else:
                efficiency = 0.0
                eff_error = 0.0

            efficiencies.append(efficiency)
            eff_errors.append(eff_error)

        return pt_centers, np.array(efficiencies), np.array(eff_errors), overall_purity

    def binomial_err(self, p, n):
        """Calculate binomial error."""
        return ((p * (1 - p)) / n) ** 0.5

    def plot_efficiency_vs_pt(self, working_points=None, skip_individual_plots=True, output_subdir=None):
        """Plot efficiency vs pT for different working points with overall purity in legend
        OPTIMIZATION: Skip individual plots by default to reduce file I/O.
        """
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS

        print("Generating efficiency plots (optimized)...")

        # Only create the main combined plots, skip individual plots by default

        if not skip_individual_plots:
            # Only create technology-specific plots if explicitly requested
            self._plot_efficiency_vs_pt_by_technology(working_points)

    def _plot_efficiency_vs_pt_by_technology(self, working_points, output_subdir=None):
        """Create efficiency vs pT plots for each sensor technology."""
        # Only generate plots when output_subdir is specified to avoid duplicates in main run directory
        if output_subdir is None:
            print("Skipping technology plots: output_subdir not specified (avoiding duplicates)")
            return

        for tech_name, tech_value in self.technology_mapping.items():
            print(f"Generating efficiency plots for {tech_name} technology...")

            # Create technology mask
            tech_mask = self.all_particle_technology == tech_value

            if not np.any(tech_mask):
                print(f"Warning: No hits found for technology {tech_name} (value {tech_value})")
                continue

            print(f"Found {np.sum(tech_mask)} hits for {tech_name} technology")

            # Prepare data for all working points
            results_dict = {}

            for wp in working_points:
                pt_centers, efficiencies, eff_errors, overall_purity = self.calculate_efficiency_by_pt(wp, tech_mask)

                if pt_centers is None:
                    continue

                # Create pt bins from centers (approximate)
                pt_bins = np.zeros(len(pt_centers) + 1)
                if len(pt_centers) > 1:
                    bin_width = pt_centers[1] - pt_centers[0]
                    pt_bins[0] = pt_centers[0] - bin_width / 2
                    for i in range(len(pt_centers)):
                        pt_bins[i + 1] = pt_centers[i] + bin_width / 2
                else:
                    pt_bins = np.array([0, 200])  # fallback

                results_dict[f"WP {wp:.3f}"] = {
                    "pt_bins": pt_bins,
                    "pt_centers": pt_centers,
                    "efficiency": efficiencies,
                    "efficiency_err": eff_errors,
                    "overall_purity": overall_purity,
                    "counts": None,
                }

            if not results_dict:
                print(f"Warning: No valid working points for technology {tech_name}")
                continue

            # Create the main combined plot for this technology
            _fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Color schemes for different working points
            colors = ["coral", "royalblue", "forestgreen", "purple", "orange", "brown"]

            for i, (wp_name, wp_data) in enumerate(results_dict.items()):
                color = colors[i % len(colors)]

                # Plot efficiency with step plot and error bands
                self._plot_metric_with_errors(
                    ax,
                    wp_data["pt_bins"],
                    wp_data["efficiency"],
                    wp_data["efficiency_err"],
                    wp_data["counts"],
                    color,
                    f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})",
                    "efficiency",
                )

            # Format efficiency plot
            ax.set_xlabel("Truth Muon $p_T$ [GeV]", fontsize=14)
            ax.set_ylabel("Hit Filter Efficiency", fontsize=14)
            ax.set_title(f"ATLAS Muon Hit Filter - {tech_name} Technology", loc="left", fontsize=14)
            ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
            ax.legend()
            ax.set_ylim([0.85, 1.05])

            plt.tight_layout()

            # Save plot
            if output_subdir is not None:
                output_path = output_subdir / f"efficiency_vs_pt_{tech_name}.png"
            else:
                output_path = self.output_dir / f"efficiency_vs_pt_{tech_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"{tech_name} efficiency vs pT plot saved to: {output_path}")

            # Create individual plots for each working point for this technology
            self._plot_individual_working_points_pt_technology(results_dict, tech_name, output_subdir)

    def _plot_metric_with_errors(self, ax, pt_bins, values, errors, counts, color, label, metric_type):
        """Helper function to plot metrics with error bands and step plots."""
        for i in range(len(pt_bins) - 1):
            lhs, rhs = pt_bins[i], pt_bins[i + 1]
            value = values[i]
            error = errors[i] if errors is not None else 0

            # Create error band
            if error > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                if metric_type == "track_length":
                    # Don't cap values for track length plots
                    y_upper = value + error
                    y_lower = max(value - error, 0.0)  # Only floor at 0.0
                else:
                    # Cap efficiency values between 0 and 1
                    y_upper = min(value + error, 1.0)  # Cap at 1.0
                    y_lower = max(value - error, 0.0)  # Floor at 0.0
                ax.fill_between(point_in_range, y_upper, y_lower, color=color, alpha=0.3, label=f"binomial err - {label}" if i == 0 else "")

            # Step plot
            ax.step([lhs, rhs], [value, value], color=color, linewidth=2.5, label=label if i == 0 else "")

    def _plot_individual_working_points_pt_technology(self, results_dict, tech_name, output_subdir=None):
        """Create separate efficiency plots for each working point for a specific technology (pt)."""
        # Only generate plots when output_subdir is specified to avoid duplicates in main run directory
        if output_subdir is None:
            print(f"Skipping individual pt plots for {tech_name}: output_subdir not specified (avoiding duplicates)")
            return

        # Create subdirectory for organizing plots
        efficiency_dir = output_subdir / f"efficiency_plots_{tech_name}"
        efficiency_dir.mkdir(exist_ok=True)

        for wp_name, wp_data in results_dict.items():
            # Create individual efficiency plot
            _fig, ax = plt.subplots(1, 1, figsize=(10, 7))

            self._plot_metric_with_errors(
                ax,
                wp_data["pt_bins"],
                wp_data["efficiency"],
                wp_data["efficiency_err"],
                wp_data["counts"],
                "coral",
                f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})",
                "efficiency",
            )

            ax.set_xlabel("Truth Muon $p_T$ [GeV]", fontsize=14)
            ax.set_ylabel("Hit Filter Efficiency", fontsize=14)
            ax.set_title(f"ATLAS Muon Hit Filter - {tech_name} - {wp_name}", loc="left", fontsize=14)
            ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
            ax.legend()
            ax.set_ylim([0.85, 1.05])

            plt.tight_layout()

            output_path = efficiency_dir / f"efficiency_vs_pt_{wp_name.lower().replace(' ', '_').replace('.', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        print(f"Individual {tech_name} pt working point plots saved to: {efficiency_dir}")

    def plot_track_lengths(self, output_subdir=None):
        """Plot track lengths (number of hits per track) binned by pT, eta, and phi."""
        print("Generating track length plots...")

        # Create track_lengths subdirectory in appropriate output directory
        save_dir = self.output_dir if output_subdir is None else output_subdir
        track_lengths_dir = save_dir / "track_lengths"
        track_lengths_dir.mkdir(exist_ok=True)

        # Calculate track lengths for each unique (event_id, particle_id) combination
        unique_tracks = {}
        track_properties = {}

        # Group hits by (event_id, particle_id) to count hits per track
        # Only consider true hits for track length calculation
        for i in range(len(self.all_event_ids)):
            # Skip if this is not a true hit
            if not self.all_true_labels[i]:
                continue

            event_id = int(self.all_event_ids[i])
            particle_id = int(self.all_particle_ids[i])
            track_key = (event_id, particle_id)

            if track_key not in unique_tracks:
                unique_tracks[track_key] = 0
                # Store the first occurrence properties for this track
                track_properties[track_key] = {"pt": self.all_particle_pts[i], "eta": self.all_particle_etas[i], "phi": self.all_particle_phis[i]}

            unique_tracks[track_key] += 1

        # Extract track lengths and properties
        track_lengths = list(unique_tracks.values())
        track_pts = [track_properties[key]["pt"] for key in unique_tracks]
        track_etas = [track_properties[key]["eta"] for key in unique_tracks]
        track_phis = [track_properties[key]["phi"] for key in unique_tracks]

        track_lengths = np.array(track_lengths)
        track_pts = np.array(track_pts)
        track_etas = np.array(track_etas)
        track_phis = np.array(track_phis)

        print(f"Found {len(track_lengths)} unique tracks")
        print(f"Track length statistics: min={np.min(track_lengths)}, max={np.max(track_lengths)}, mean={np.mean(track_lengths):.1f}")

        # Plot track length vs pT
        self._plot_track_length_vs_pt(track_lengths, track_pts, track_lengths_dir)

        # Plot track length vs eta
        self._plot_track_length_vs_eta(track_lengths, track_etas, track_lengths_dir)

        # Plot track length vs phi
        self._plot_track_length_vs_phi(track_lengths, track_phis, track_lengths_dir)

    def _plot_track_length_vs_pt(self, track_lengths, track_pts, output_dir):
        """Plot average track length vs pT with binomial errors."""
        # Define pT bins (linear scale) - 0 to 200 GeV
        pt_bins = np.linspace(0, 200, 21)  # 20 bins
        (pt_bins[:-1] + pt_bins[1:]) / 2

        # Calculate average track length in each pT bin
        avg_lengths = []
        std_lengths = []

        for i in range(len(pt_bins) - 1):
            mask = (track_pts >= pt_bins[i]) & (track_pts < pt_bins[i + 1])
            if np.sum(mask) > 0:
                lengths_in_bin = track_lengths[mask]
                avg_lengths.append(np.mean(lengths_in_bin))
                # Use standard error of mean for error bars
                std_lengths.append(np.std(lengths_in_bin) / np.sqrt(len(lengths_in_bin)))
            else:
                avg_lengths.append(0)
                std_lengths.append(0)

        avg_lengths = np.array(avg_lengths)
        std_lengths = np.array(std_lengths)

        # Create plot
        _fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Use the same plotting style as efficiency plots
        self._plot_metric_with_errors(ax, pt_bins, avg_lengths, std_lengths, None, "royalblue", "Average Track Length", "track_length")

        ax.set_xlabel("Truth Muon $p_T$ [GeV]", fontsize=14)
        ax.set_ylabel("Average Track Length [hits]", fontsize=14)
        ax.set_title("ATLAS Muon Track Length vs $p_T$", loc="left", fontsize=14)
        ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
        ax.legend()
        ax.set_xlim([0, 200])
        ax.set_ylim([0, 60])

        plt.tight_layout()

        output_path = output_dir / "track_length_vs_pt.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Track length vs pT plot saved to: {output_path}")

    def _plot_track_length_vs_eta(self, track_lengths, track_etas, output_dir):
        """Plot average track length vs eta with binomial errors."""
        # Define eta bins - same as efficiency plots
        eta_bins = np.linspace(-2.7, 2.7, 21)  # 20 bins
        (eta_bins[:-1] + eta_bins[1:]) / 2

        # Calculate average track length in each eta bin
        avg_lengths = []
        std_lengths = []

        for i in range(len(eta_bins) - 1):
            mask = (track_etas >= eta_bins[i]) & (track_etas < eta_bins[i + 1])
            if np.sum(mask) > 0:
                lengths_in_bin = track_lengths[mask]
                avg_lengths.append(np.mean(lengths_in_bin))
                # Use standard error of mean for error bars
                std_lengths.append(np.std(lengths_in_bin) / np.sqrt(len(lengths_in_bin)))
            else:
                avg_lengths.append(0)
                std_lengths.append(0)

        avg_lengths = np.array(avg_lengths)
        std_lengths = np.array(std_lengths)

        # Create plot
        _fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Use the same plotting style as efficiency plots
        self._plot_metric_with_errors_eta_phi(ax, eta_bins, avg_lengths, std_lengths, None, "forestgreen", "Average Track Length", "track_length")

        ax.set_xlabel("Truth Muon $\\eta$", fontsize=14)
        ax.set_ylabel("Average Track Length [hits]", fontsize=14)
        ax.set_title("ATLAS Muon Track Length vs $\\eta$", loc="left", fontsize=14)
        ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
        ax.legend()
        ax.set_xlim([-2.7, 2.7])
        ax.set_ylim([0, 60])

        plt.tight_layout()

        output_path = output_dir / "track_length_vs_eta.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Track length vs eta plot saved to: {output_path}")

    def _plot_track_length_vs_phi(self, track_lengths, track_phis, output_dir):
        """Plot average track length vs phi with binomial errors."""
        # Define phi bins - same as efficiency plots
        phi_bins = np.linspace(-3.2, 3.2, 21)  # 20 bins
        (phi_bins[:-1] + phi_bins[1:]) / 2

        # Calculate average track length in each phi bin
        avg_lengths = []
        std_lengths = []

        for i in range(len(phi_bins) - 1):
            mask = (track_phis >= phi_bins[i]) & (track_phis < phi_bins[i + 1])
            if np.sum(mask) > 0:
                lengths_in_bin = track_lengths[mask]
                avg_lengths.append(np.mean(lengths_in_bin))
                # Use standard error of mean for error bars
                std_lengths.append(np.std(lengths_in_bin) / np.sqrt(len(lengths_in_bin)))
            else:
                avg_lengths.append(0)
                std_lengths.append(0)

        avg_lengths = np.array(avg_lengths)
        std_lengths = np.array(std_lengths)

        # Create plot
        _fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Use the same plotting style as efficiency plots
        self._plot_metric_with_errors_eta_phi(ax, phi_bins, avg_lengths, std_lengths, None, "purple", "Average Track Length", "track_length")

        ax.set_xlabel("Truth Muon $\\phi$", fontsize=14)
        ax.set_ylabel("Average Track Length [hits]", fontsize=14)
        ax.set_title("ATLAS Muon Track Length vs $\\phi$", loc="left", fontsize=14)
        ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
        ax.legend()
        ax.set_xlim([-3.2, 3.2])
        ax.set_ylim([0, 60])

        plt.tight_layout()

        output_path = output_dir / "track_length_vs_phi.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Track length vs phi plot saved to: {output_path}")

    def plot_working_point_performance(self, working_points=None, output_subdir=None):
        """Plot average purity for different working points with detailed track statistics.

        ULTRA-OPTIMIZATIONS APPLIED:
        1. Use cached ROC curve
        2. Pre-compute all thresholds and predictions in vectorized operations
        3. Use ultra-fast pandas groupby operations for track statistics
        4. Minimize memory allocations and copies
        """
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS

        print("Generating working point performance plot (ultra-optimized)...")

        # Use cached ROC curve
        if not hasattr(self, "_cached_roc"):
            print("Computing ROC curve (cached for reuse)...")
            self._cached_roc = roc_curve(self.all_true_labels, self.all_logits)

        _fpr, tpr, thresholds = self._cached_roc

        avg_purities = []
        avg_purity_errors = []

        # Pre-calculate all thresholds and predictions (vectorized)
        print("Pre-calculating thresholds and predictions (vectorized)...")
        thresholds_dict = {}
        predictions_dict = {}

        for wp in working_points:
            # Find threshold for this working point
            valid_indices = tpr >= wp
            if not np.any(valid_indices):
                avg_purities.append(0.0)
                avg_purity_errors.append(0.0)
                thresholds_dict[wp] = None
                predictions_dict[wp] = None
                continue

            threshold = thresholds[tpr >= wp][0]
            predictions = self.all_logits >= threshold
            thresholds_dict[wp] = threshold
            predictions_dict[wp] = predictions

            # Calculate overall purity (vectorized)
            total_true_positives = np.sum(self.all_true_labels & predictions)
            total_predicted_positives = np.sum(predictions)

            if total_predicted_positives > 0:
                purity = total_true_positives / total_predicted_positives
                purity_error = np.sqrt(purity * (1 - purity) / total_predicted_positives)
            else:
                purity = 0.0
                purity_error = 0.0

            avg_purities.append(purity)
            avg_purity_errors.append(purity_error)

        # Calculate detailed track statistics for all working points at once (ultra-fast)
        track_statistics = self._calculate_track_statistics_ultra_fast_optimized(working_points, predictions_dict)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            working_points,
            avg_purities,
            yerr=avg_purity_errors,
            marker="o",
            capsize=4,
            linewidth=2,
            markersize=8,
            color="darkred",
            label="Average Purity",
        )

        plt.xlabel("Working Point (Target Efficiency)")
        plt.ylabel("Achieved Average Purity")
        plt.title("Working Point Performance - ATLAS Muon Hit Filter")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.1, 1.05)  # Zoom in on y-axis as requested

        # Save plot to appropriate directory
        save_dir = self.output_dir if output_subdir is None else output_subdir
        output_path = save_dir / "working_point_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Working point performance plot saved to {output_path}")

        # Save detailed statistics
        self._save_working_point_statistics(working_points, avg_purities, avg_purity_errors, track_statistics, output_path, output_subdir)

        # Print performance summary
        print("\nWorking Point Performance Summary:")
        for wp, purity, error in zip(working_points, avg_purities, avg_purity_errors, strict=False):
            print(f"  WP {wp:.3f}: Purity = {purity:.4f} ± {error:.4f}")

    def _calculate_track_statistics_ultra_fast_optimized(self, working_points, predictions_dict):
        """Ultra-optimized pandas-based calculation of track statistics for all working points."""
        print("Calculating track statistics (ultra-fast optimized)...")

        # Pre-filter data to reduce memory usage and computation time
        valid_particle_mask = self.all_particle_ids >= 0  # Remove noise hits

        if not np.any(valid_particle_mask):
            # Return empty statistics if no valid particles
            return {wp: {"total_tracks": 0, "tracks_with_few_hits": 0, "tracks_completely_lost": 0, "events_analyzed": 0} for wp in working_points}

        # Create a DataFrame only with valid particles for maximum speed
        df = pd.DataFrame({
            "event_id": self.all_event_ids[valid_particle_mask],
            "particle_id": self.all_particle_ids[valid_particle_mask],
            "true_label": self.all_true_labels[valid_particle_mask],
            "original_idx": np.where(valid_particle_mask)[0],  # Track original indices efficiently
        })

        # Group by track and get track-level true hit counts (ultra-fast)
        track_info = df.groupby(["event_id", "particle_id"], sort=False)["true_label"].sum()

        # Only keep tracks that have at least one true hit
        valid_tracks = track_info[track_info > 0].index
        valid_tracks_set = set(valid_tracks)

        print(f"Found {len(valid_tracks_set)} valid tracks to analyze")

        track_statistics = {}

        for wp in working_points:
            if predictions_dict[wp] is None:
                track_statistics[wp] = {
                    "total_tracks": 0,
                    "tracks_with_few_hits": 0,
                    "tracks_completely_lost": 0,
                    "events_analyzed": len(np.unique(self.all_event_ids)),
                }
                continue

            # Add predictions to dataframe using original indices (vectorized)
            df_wp = df.copy()
            df_wp["predicted"] = predictions_dict[wp][df_wp["original_idx"]]

            # Group by track and count predicted hits (ultra-fast)
            track_predictions = df_wp.groupby(["event_id", "particle_id"], sort=False)["predicted"].sum()

            # Filter to only valid tracks using set intersection (ultra-fast)
            valid_track_predictions = track_predictions.loc[track_predictions.index.intersection(valid_tracks)]

            # Calculate statistics using vectorized pandas operations
            total_tracks = len(valid_track_predictions)
            if total_tracks > 0:
                tracks_completely_lost = int((valid_track_predictions == 0).sum())
                tracks_with_few_hits = int(((valid_track_predictions > 0) & (valid_track_predictions < 3)).sum())
            else:
                tracks_completely_lost = 0
                tracks_with_few_hits = 0

            track_statistics[wp] = {
                "total_tracks": total_tracks,
                "tracks_with_few_hits": tracks_with_few_hits,
                "tracks_completely_lost": tracks_completely_lost,
                "events_analyzed": len(np.unique(self.all_event_ids)),
            }

            if total_tracks > 0:
                print(f"  WP {wp:.3f}: {total_tracks} tracks, {tracks_completely_lost} lost ({tracks_completely_lost / total_tracks * 100:.1f}%)")

        return track_statistics

    def _calculate_track_statistics_vectorized(self, working_points, predictions_dict):
        """Vectorized calculation of track statistics for all working points."""
        print("Calculating track statistics (vectorized)...")

        # Pre-compute masks and data structures
        unique_events = np.unique(self.all_event_ids)
        events_analyzed = len(unique_events)

        # Initialize results dictionary
        track_statistics = {}

        # Create event-to-index mapping for faster lookup
        event_to_indices = {}
        for event_id in unique_events:
            event_mask = self.all_event_ids == event_id
            event_to_indices[event_id] = np.where(event_mask)[0]

        for wp in working_points:
            if predictions_dict[wp] is None:
                track_statistics[wp] = {"total_tracks": 0, "tracks_with_few_hits": 0, "tracks_completely_lost": 0, "events_analyzed": events_analyzed}
                continue

            predictions = predictions_dict[wp]

            total_tracks = 0
            tracks_with_few_hits = 0
            tracks_completely_lost = 0

            # Process events in batches for better memory efficiency
            for event_id in unique_events:
                event_indices = event_to_indices[event_id]

                # Get data for this event using pre-computed indices
                event_particle_ids = self.all_particle_ids[event_indices]
                event_predictions = predictions[event_indices]
                event_true_labels = self.all_true_labels[event_indices]

                # Get unique particles (tracks) excluding noise (-1)
                unique_particles = np.unique(event_particle_ids)
                unique_particles = unique_particles[unique_particles >= 0]

                # Vectorized processing of particles in this event
                for particle_id in unique_particles:
                    particle_mask = event_particle_ids == particle_id
                    particle_true_hits = event_true_labels[particle_mask]

                    # Skip if no true hits for this particle
                    if np.sum(particle_true_hits) == 0:
                        continue

                    total_tracks += 1

                    # Count predicted hits for this track
                    num_predicted_hits = np.sum(event_predictions[particle_mask])

                    if num_predicted_hits == 0:
                        tracks_completely_lost += 1
                    elif num_predicted_hits < 3:
                        tracks_with_few_hits += 1

            track_statistics[wp] = {
                "total_tracks": total_tracks,
                "tracks_with_few_hits": tracks_with_few_hits,
                "tracks_completely_lost": tracks_completely_lost,
                "events_analyzed": events_analyzed,
            }

            print(f"  WP {wp:.3f}: {total_tracks} tracks processed")

        return track_statistics

    def _calculate_technology_statistics(self):
        """Calculate technology distribution in the truth labels."""
        tech_stats = {}
        total_true_hits = np.sum(self.all_true_labels)

        # Calculate statistics for each technology
        for tech_name, tech_value in self.technology_mapping.items():
            # Create mask for this technology
            tech_mask = self.all_particle_technology == tech_value

            # Count true hits for this technology
            tech_true_hits = np.sum(self.all_true_labels & tech_mask)
            total_tech_hits = np.sum(tech_mask)

            # Calculate percentage of total true hits
            percentage_of_true = (tech_true_hits / total_true_hits * 100) if total_true_hits > 0 else 0.0

            tech_stats[tech_name] = {
                "true_hits": int(tech_true_hits),
                "total_hits": int(total_tech_hits),
                "percentage_of_true_hits": percentage_of_true,
            }

        return tech_stats

    def _calculate_track_statistics_per_working_point(self, working_point, threshold, predictions):
        """Calculate detailed track statistics for a specific working point."""
        # Get unique events
        unique_events = np.unique(self.all_event_ids)

        total_tracks = 0
        tracks_with_few_hits = 0  # tracks with < 3 hits after filtering
        tracks_completely_lost = 0  # tracks with 0 hits after filtering
        events_analyzed = len(unique_events)

        for event_id in unique_events:
            # Get hits for this event
            event_mask = self.all_event_ids == event_id
            event_particle_ids = self.all_particle_ids[event_mask]
            event_predictions = predictions[event_mask]
            event_true_labels = self.all_true_labels[event_mask]

            # Get unique particle IDs (tracks) in this event, excluding noise (-1)
            unique_particles = np.unique(event_particle_ids)
            unique_particles = unique_particles[unique_particles >= 0]  # Remove noise hits

            for particle_id in unique_particles:
                # Check if this particle has any true hits (is a valid track)
                particle_mask = event_particle_ids == particle_id
                particle_true_hits = event_true_labels[particle_mask]

                if np.sum(particle_true_hits) == 0:
                    continue  # Skip if no true hits for this particle

                total_tracks += 1

                # Count predicted hits for this track
                particle_predicted_hits = event_predictions[particle_mask]
                num_predicted_hits = np.sum(particle_predicted_hits)

                if num_predicted_hits == 0:
                    tracks_completely_lost += 1
                elif num_predicted_hits < 3:
                    tracks_with_few_hits += 1

        return {
            "total_tracks": total_tracks,
            "tracks_with_few_hits": tracks_with_few_hits,
            "tracks_completely_lost": tracks_completely_lost,
            "events_analyzed": events_analyzed,
        }

    def _save_working_point_statistics(self, working_points, purities, purity_errors, track_statistics, output_plot_path, output_subdir=None):
        """Save comprehensive working point statistics to a text file.

        Parameters:
        -----------
        working_points : list
            List of working point values
        purities : list
            List of purity values for each working point
        purity_errors : list
            List of purity error values for each working point
        track_statistics : dict
            Dictionary containing track statistics for each working point
        output_plot_path : Path
            Path where the plot is saved, used to determine where to save the statistics file
        """
        from datetime import datetime

        # Determine output directory and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_plot_path:
            # Save the txt file in the same directory as the plot
            plot_path = Path(output_plot_path)
            output_dir = plot_path.parent
            filename = output_dir / f"working_point_statistics_{timestamp}.txt"
        else:
            # Default to current directory if no plot path provided
            filename = Path(f"working_point_statistics_{timestamp}.txt")

        with filename.open("w") as f:
            f.write("=" * 80 + "\n")
            f.write("WORKING POINT PERFORMANCE STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max events analyzed: {self.max_events}\n\n")

            # Overall summary
            f.write("OVERALL PERFORMANCE SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.writelines(
                f"Working Point {wp:.3f}: Purity = {purity:.4f} ± {error:.4f}\n"
                for wp, purity, error in zip(working_points, purities, purity_errors, strict=False)
            )
            f.write("\n")

            # Technology distribution in truth labels
            tech_stats = self._calculate_technology_statistics()
            f.write("TECHNOLOGY DISTRIBUTION IN TRUTH LABELS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Technology':<12} {'True Hits':<12} {'Total Hits':<12} {'% of True':<12}\n")
            f.write("-" * 50 + "\n")

            f.writelines(
                f"{tech_name:<12} {stats['true_hits']:<12,} {stats['total_hits']:<12,} {stats['percentage_of_true_hits']:<12.1f}%\n"
                for tech_name, stats in tech_stats.items()
            )

            f.write(f"\nTotal true hits across all technologies: {np.sum(self.all_true_labels):,}\n")
            f.write(f"Total hits across all technologies: {len(self.all_true_labels):,}\n")
            f.write(f"Overall true hit rate: {(np.sum(self.all_true_labels) / len(self.all_true_labels) * 100):.1f}%\n\n")

            # Detailed statistics for each working point
            f.write("DETAILED TRACK STATISTICS BY WORKING POINT:\n")
            f.write("=" * 60 + "\n")

            for wp in working_points:
                stats = track_statistics[wp]
                f.write(f"\nWorking Point {wp:.3f}:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Events analyzed: {stats['events_analyzed']}\n")
                f.write(f"Total valid tracks: {stats['total_tracks']}\n")

                if stats["total_tracks"] > 0:
                    # Tracks completely lost
                    lost_percentage = (stats["tracks_completely_lost"] / stats["total_tracks"]) * 100
                    f.write(f"Tracks completely lost (0 hits): {stats['tracks_completely_lost']} ({lost_percentage:.2f}%)\n")

                    # Tracks with few hits
                    few_hits_percentage = (stats["tracks_with_few_hits"] / stats["total_tracks"]) * 100
                    f.write(f"Tracks with <3 hits: {stats['tracks_with_few_hits']} ({few_hits_percentage:.2f}%)\n")

                    # Tracks with ≥3 hits (good tracks)
                    good_tracks = stats["total_tracks"] - stats["tracks_completely_lost"] - stats["tracks_with_few_hits"]
                    good_tracks_percentage = (good_tracks / stats["total_tracks"]) * 100
                    f.write(f"Tracks with ≥3 hits: {good_tracks} ({good_tracks_percentage:.2f}%)\n")

                    # Track survival rate
                    survived_tracks = stats["total_tracks"] - stats["tracks_completely_lost"]
                    survival_rate = (survived_tracks / stats["total_tracks"]) * 100
                    f.write(f"Track survival rate: {survival_rate:.2f}%\n")
                else:
                    f.write("No valid tracks found\n")

            # Comparison table
            f.write("\n" + "=" * 80 + "\n")
            f.write("COMPARISON TABLE:\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'WP':<6} {'Purity':<8} {'Total':<7} {'Lost':<6} {'Lost%':<7} {'<3hits':<7} {'<3hits%':<8} {'≥3hits':<7} {'≥3hits%':<8}\n")
            f.write("-" * 80 + "\n")

            for wp in working_points:
                stats = track_statistics[wp]
                purity = next(p for w, p in zip(working_points, purities, strict=False) if w == wp)

                if stats["total_tracks"] > 0:
                    lost_pct = (stats["tracks_completely_lost"] / stats["total_tracks"]) * 100
                    few_hits_pct = (stats["tracks_with_few_hits"] / stats["total_tracks"]) * 100
                    good_tracks = stats["total_tracks"] - stats["tracks_completely_lost"] - stats["tracks_with_few_hits"]
                    good_pct = (good_tracks / stats["total_tracks"]) * 100

                    f.write(
                        f"{wp:<6.3f} {purity:<8.4f} {stats['total_tracks']:<7} {stats['tracks_completely_lost']:<6} "
                        f"{lost_pct:<7.1f} {stats['tracks_with_few_hits']:<7} {few_hits_pct:<8.1f} "
                        f"{good_tracks:<7} {good_pct:<8.1f}\n"
                    )
                else:
                    f.write(f"{wp:<6.3f} {purity:<8.4f} {stats['total_tracks']:<7} -      -       -       -        -       -\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("LEGEND:\n")
            f.write("WP      = Working Point (target efficiency)\n")
            f.write("Purity  = Hit filter purity (precision)\n")
            f.write("Total   = Total number of valid tracks\n")
            f.write("Lost    = Tracks with 0 predicted hits\n")
            f.write("Lost%   = Percentage of tracks completely lost\n")
            f.write("<3hits  = Tracks with 1-2 predicted hits\n")
            f.write("<3hits% = Percentage of tracks with <3 hits\n")
            f.write("≥3hits  = Tracks with 3 or more predicted hits\n")
            f.write("≥3hits% = Percentage of tracks with ≥3 hits\n")
            f.write("\nTECHNOLOGY CODES:\n")
            f.writelines(f"{tech_name:<6} = {tech_value}\n" for tech_name, tech_value in self.technology_mapping.items())
            f.write("=" * 80 + "\n")

        print(f"\nWorking point statistics saved to: {filename}")

        # Print summary to console
        if len(working_points) > 0:
            best_wp_idx = np.argmax(purities)
            best_wp = working_points[best_wp_idx]
            best_stats = track_statistics[best_wp]

            print("\nSummary:")
            print(f"  - Best working point: {best_wp:.3f} (purity: {purities[best_wp_idx]:.3f})")
            if best_stats["total_tracks"] > 0:
                survival_rate = ((best_stats["total_tracks"] - best_stats["tracks_completely_lost"]) / best_stats["total_tracks"]) * 100
                print(f"  - Track survival rate at best WP: {survival_rate:.1f}%")
                good_tracks = best_stats["total_tracks"] - best_stats["tracks_completely_lost"] - best_stats["tracks_with_few_hits"]
                good_rate = (good_tracks / best_stats["total_tracks"]) * 100
                print(f"  - Tracks with ≥3 hits at best WP: {good_rate:.1f}%")

    def calculate_efficiency_by_eta(self, working_point, technology_mask=None):
        """Calculate efficiency binned by truthMuon_eta and overall purity."""
        # Calculate ROC curve to determine threshold for target efficiency
        if technology_mask is not None:
            # Filter data by technology
            logits = self.all_logits[technology_mask]
            true_labels = self.all_true_labels[technology_mask]
            particle_etas = self.all_particle_etas[technology_mask]
        else:
            logits = self.all_logits
            true_labels = self.all_true_labels
            particle_etas = self.all_particle_etas

        _fpr, tpr, thresholds = roc_curve(true_labels, logits)

        # Find threshold that gives the desired efficiency (recall)
        target_efficiency = working_point

        # Find the threshold that achieves the target efficiency
        if not np.any(tpr >= target_efficiency):
            print(f"Warning: Cannot achieve efficiency {target_efficiency}")
            return None, None, None, None

        threshold = thresholds[tpr >= target_efficiency][0]

        # Apply threshold to get predictions
        cut_predictions = logits >= threshold

        # Calculate overall purity for this working point
        total_true_positives = np.sum(true_labels & cut_predictions)
        total_predicted_positives = np.sum(cut_predictions)

        overall_purity = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0

        # Define eta bins
        eta_min, eta_max = -2.7, 2.7
        eta_bins = np.linspace(eta_min, eta_max, 21)  # 20 bins
        eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2

        # Calculate efficiency for each bin
        efficiencies = []
        eff_errors = []

        for i in range(len(eta_bins) - 1):
            eta_mask = (particle_etas >= eta_bins[i]) & (particle_etas < eta_bins[i + 1])

            if not np.any(eta_mask):
                efficiencies.append(0.0)
                eff_errors.append(0.0)
                continue

            bin_true_labels = true_labels[eta_mask]
            bin_predictions = cut_predictions[eta_mask]

            # Calculate efficiency (recall)
            true_positives = np.sum(bin_true_labels & bin_predictions)
            total_positives = np.sum(bin_true_labels)

            if total_positives > 0:
                efficiency = true_positives / total_positives
                # Binomial error for efficiency
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_positives)
            else:
                efficiency = 0.0
                eff_error = 0.0

            efficiencies.append(efficiency)
            eff_errors.append(eff_error)

        return eta_centers, np.array(efficiencies), np.array(eff_errors), overall_purity

    def calculate_efficiency_by_phi(self, working_point, technology_mask=None):
        """Calculate efficiency binned by truthMuon_phi and overall purity."""
        # Calculate ROC curve to determine threshold for target efficiency
        if technology_mask is not None:
            # Filter data by technology
            logits = self.all_logits[technology_mask]
            true_labels = self.all_true_labels[technology_mask]
            particle_phis = self.all_particle_phis[technology_mask]
        else:
            logits = self.all_logits
            true_labels = self.all_true_labels
            particle_phis = self.all_particle_phis

        _fpr, tpr, thresholds = roc_curve(true_labels, logits)

        # Find threshold that gives the desired efficiency (recall)
        target_efficiency = working_point

        # Find the threshold that achieves the target efficiency
        if not np.any(tpr >= target_efficiency):
            print(f"Warning: Cannot achieve efficiency {target_efficiency}")
            return None, None, None, None

        threshold = thresholds[tpr >= target_efficiency][0]

        # Apply threshold to get predictions
        cut_predictions = logits >= threshold

        # Calculate overall purity for this working point
        total_true_positives = np.sum(true_labels & cut_predictions)
        total_predicted_positives = np.sum(cut_predictions)

        overall_purity = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0

        # Define phi bins
        phi_min, phi_max = -3.2, 3.2
        phi_bins = np.linspace(phi_min, phi_max, 21)  # 20 bins
        phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2

        # Calculate efficiency for each bin
        efficiencies = []
        eff_errors = []

        for i in range(len(phi_bins) - 1):
            phi_mask = (particle_phis >= phi_bins[i]) & (particle_phis < phi_bins[i + 1])

            if not np.any(phi_mask):
                efficiencies.append(0.0)
                eff_errors.append(0.0)
                continue

            bin_true_labels = true_labels[phi_mask]
            bin_predictions = cut_predictions[phi_mask]

            # Calculate efficiency (recall)
            true_positives = np.sum(bin_true_labels & bin_predictions)
            total_positives = np.sum(bin_true_labels)

            if total_positives > 0:
                efficiency = true_positives / total_positives
                # Binomial error for efficiency
                eff_error = np.sqrt(efficiency * (1 - efficiency) / total_positives)
            else:
                efficiency = 0.0
                eff_error = 0.0

            efficiencies.append(efficiency)
            eff_errors.append(eff_error)

        return phi_centers, np.array(efficiencies), np.array(eff_errors), overall_purity

    def plot_efficiency_vs_eta(self, working_points=None, skip_individual_plots=True, output_subdir=None):
        """Plot efficiency vs eta for different working points with overall purity in legend."""
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        print("Generating efficiency vs eta plots...")

        # Then, create technology-specific plots
        self._plot_efficiency_vs_eta_by_technology(working_points, output_subdir)

    def _plot_efficiency_vs_eta_by_technology(self, working_points, output_subdir=None):
        """Create efficiency vs eta plots for each sensor technology."""
        # Only generate plots when output_subdir is specified to avoid duplicates in main run directory
        if output_subdir is None:
            print("Skipping technology eta plots: output_subdir not specified (avoiding duplicates)")
            return

        for tech_name, tech_value in self.technology_mapping.items():
            print(f"Generating efficiency vs eta plots for {tech_name} technology...")

            # Create technology mask
            tech_mask = self.all_particle_technology == tech_value

            if not np.any(tech_mask):
                print(f"Warning: No hits found for technology {tech_name} (value {tech_value})")
                continue

            # Prepare data for all working points
            results_dict = {}

            for wp in working_points:
                eta_centers, efficiencies, eff_errors, overall_purity = self.calculate_efficiency_by_eta(wp, tech_mask)

                if eta_centers is None:
                    continue

                # Create eta bins from centers (approximate)
                eta_bins = np.zeros(len(eta_centers) + 1)
                if len(eta_centers) > 1:
                    bin_width = eta_centers[1] - eta_centers[0]
                    eta_bins[0] = eta_centers[0] - bin_width / 2
                    for i in range(len(eta_centers)):
                        eta_bins[i + 1] = eta_centers[i] + bin_width / 2
                else:
                    eta_bins = np.array([-2.7, 2.7])  # fallback

                results_dict[f"WP {wp:.3f}"] = {
                    "eta_bins": eta_bins,
                    "eta_centers": eta_centers,
                    "efficiency": efficiencies,
                    "efficiency_err": eff_errors,
                    "overall_purity": overall_purity,
                    "counts": None,
                }

            if not results_dict:
                print(f"Warning: No valid working points for technology {tech_name}")
                continue

            # Create the main combined plot for this technology
            _fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Color schemes for different working points
            colors = ["coral", "royalblue", "forestgreen", "purple", "orange", "brown"]

            for i, (wp_name, wp_data) in enumerate(results_dict.items()):
                color = colors[i % len(colors)]

                # Plot efficiency with step plot and error bands
                self._plot_metric_with_errors_eta_phi(
                    ax,
                    wp_data["eta_bins"],
                    wp_data["efficiency"],
                    wp_data["efficiency_err"],
                    wp_data["counts"],
                    color,
                    f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})",
                    "efficiency",
                )

            # Format efficiency plot
            ax.set_xlabel("Truth Muon $\\eta$", fontsize=14)
            ax.set_ylabel("Hit Filter Efficiency", fontsize=14)
            ax.set_title(f"ATLAS Muon Hit Filter - {tech_name} Technology", loc="left", fontsize=14)
            ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-2.7, 2.7])

            plt.tight_layout()

            # Save plot
            if output_subdir is not None:
                output_path = output_subdir / f"efficiency_vs_eta_{tech_name}.png"
            else:
                output_path = self.output_dir / f"efficiency_vs_eta_{tech_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"{tech_name} efficiency vs eta plot saved to: {output_path}")

            # Create individual plots for each working point for this technology
            self._plot_individual_working_points_eta_technology(results_dict, tech_name, output_subdir)

    def plot_efficiency_vs_phi(self, working_points=None, skip_individual_plots=True, output_subdir=None):
        """Plot efficiency vs phi for different working points with overall purity in legend."""
        if working_points is None:
            working_points = DEFAULT_WORKING_POINTS
        print("Generating efficiency vs phi plots...")

        # Then, create technology-specific plots
        self._plot_efficiency_vs_phi_by_technology(working_points, output_subdir)

    def _plot_efficiency_vs_phi_by_technology(self, working_points, output_subdir=None):
        """Create efficiency vs phi plots for each sensor technology."""
        # Only generate plots when output_subdir is specified to avoid duplicates in main run directory
        if output_subdir is None:
            print("Skipping technology phi plots: output_subdir not specified (avoiding duplicates)")
            return

        for tech_name, tech_value in self.technology_mapping.items():
            print(f"Generating efficiency vs phi plots for {tech_name} technology...")

            # Create technology mask
            tech_mask = self.all_particle_technology == tech_value

            if not np.any(tech_mask):
                print(f"Warning: No hits found for technology {tech_name} (value {tech_value})")
                continue

            # Prepare data for all working points
            results_dict = {}

            for wp in working_points:
                phi_centers, efficiencies, eff_errors, overall_purity = self.calculate_efficiency_by_phi(wp, tech_mask)

                if phi_centers is None:
                    continue

                # Create phi bins from centers (approximate)
                phi_bins = np.zeros(len(phi_centers) + 1)
                if len(phi_centers) > 1:
                    bin_width = phi_centers[1] - phi_centers[0]
                    phi_bins[0] = phi_centers[0] - bin_width / 2
                    for i in range(len(phi_centers)):
                        phi_bins[i + 1] = phi_centers[i] + bin_width / 2
                else:
                    phi_bins = np.array([-3.2, 3.2])  # fallback

                results_dict[f"WP {wp:.3f}"] = {
                    "phi_bins": phi_bins,
                    "phi_centers": phi_centers,
                    "efficiency": efficiencies,
                    "efficiency_err": eff_errors,
                    "overall_purity": overall_purity,
                    "counts": None,
                }

            if not results_dict:
                print(f"Warning: No valid working points for technology {tech_name}")
                continue

            # Create the main combined plot for this technology
            _fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Color schemes for different working points
            colors = ["coral", "royalblue", "forestgreen", "purple", "orange", "brown"]

            for i, (wp_name, wp_data) in enumerate(results_dict.items()):
                color = colors[i % len(colors)]

                # Plot efficiency with step plot and error bands
                self._plot_metric_with_errors_eta_phi(
                    ax,
                    wp_data["phi_bins"],
                    wp_data["efficiency"],
                    wp_data["efficiency_err"],
                    wp_data["counts"],
                    color,
                    f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})",
                    "efficiency",
                )

            # Format efficiency plot
            ax.set_xlabel("Truth Muon $\\phi$", fontsize=14)
            ax.set_ylabel("Hit Filter Efficiency", fontsize=14)
            ax.set_title(f"ATLAS Muon Hit Filter - {tech_name} Technology", loc="left", fontsize=14)
            ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-3.2, 3.2])

            plt.tight_layout()

            # Save plot
            if output_subdir is not None:
                output_path = output_subdir / f"efficiency_vs_phi_{tech_name}.png"
            else:
                output_path = self.output_dir / f"efficiency_vs_phi_{tech_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"{tech_name} efficiency vs phi plot saved to: {output_path}")

            # Create individual plots for each working point for this technology
            self._plot_individual_working_points_phi_technology(results_dict, tech_name, output_subdir)

    def _plot_metric_with_errors_eta_phi(self, ax, bins, values, errors, counts, color, label, metric_type):
        """Helper function to plot metrics with error bands and step plots for eta/phi."""
        for i in range(len(bins) - 1):
            lhs, rhs = bins[i], bins[i + 1]
            value = values[i]
            error = errors[i] if errors is not None else 0

            # Create error band
            if error > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                if metric_type == "track_length":
                    # Don't cap values for track length plots
                    y_upper = value + error
                    y_lower = max(value - error, 0.0)  # Only floor at 0.0
                else:
                    # Cap efficiency values between 0 and 1
                    y_upper = min(value + error, 1.0)  # Cap at 1.0
                    y_lower = max(value - error, 0.0)  # Floor at 0.0
                ax.fill_between(point_in_range, y_upper, y_lower, color=color, alpha=0.3, label=f"binomial err - {label}" if i == 0 else "")

            # Step plot
            ax.step([lhs, rhs], [value, value], color=color, linewidth=2.5, label=label if i == 0 else "")

    def _plot_individual_working_points_eta(self, results_dict, output_subdir=None):
        """Create separate efficiency plots for each working point (eta)."""
        # Only generate plots when output_subdir is specified to avoid duplicates in main run directory
        if output_subdir is None:
            print("Skipping individual eta working point plots: output_subdir not specified (avoiding duplicates)")
            return

        # Create subdirectory for organizing plots
        efficiency_dir = output_subdir / "efficiency_plots"
        efficiency_dir.mkdir(exist_ok=True)

        for wp_name, wp_data in results_dict.items():
            # Create individual efficiency plot
            _fig, ax = plt.subplots(1, 1, figsize=(10, 7))

            self._plot_metric_with_errors_eta_phi(
                ax,
                wp_data["eta_bins"],
                wp_data["efficiency"],
                wp_data["efficiency_err"],
                wp_data["counts"],
                "coral",
                f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})",
                "efficiency",
            )

            ax.set_xlabel("Truth Muon $\\eta$", fontsize=14)
            ax.set_ylabel("Hit Filter Efficiency", fontsize=14)
            ax.set_title(f"ATLAS Muon Hit Filter - {wp_name}", loc="left", fontsize=14)
            ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-2.7, 2.7])

            plt.tight_layout()

            output_path = efficiency_dir / f"efficiency_vs_eta_{wp_name.lower().replace(' ', '_').replace('.', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        print(f"Individual eta working point plots saved to: {efficiency_dir}")

    def _plot_individual_working_points_eta_technology(self, results_dict, tech_name, output_subdir=None):
        """Create individual eta plots for each working point and technology."""
        # Only generate plots when output_subdir is specified to avoid duplicates in main run directory
        if output_subdir is None:
            print(f"Skipping individual eta plots for {tech_name}: output_subdir not specified (avoiding duplicates)")
            return

        # Create efficiency_plots_<technology> subdirectory (unified for all coordinates)
        efficiency_plots_dir = output_subdir / f"efficiency_plots_{tech_name}"
        efficiency_plots_dir.mkdir(exist_ok=True)

        for wp_name, wp_data in results_dict.items():
            _fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            self._plot_metric_with_errors_eta_phi(
                ax,
                wp_data["eta_bins"],
                wp_data["efficiency"],
                wp_data["efficiency_err"],
                wp_data["counts"],
                "royalblue",
                f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})",
                "efficiency",
            )

            ax.set_xlabel("Truth Muon $\\eta$", fontsize=14)
            ax.set_ylabel("Hit Filter Efficiency", fontsize=14)
            ax.set_title(f"ATLAS Muon Hit Filter - {tech_name} Technology - {wp_name}", loc="left", fontsize=12)
            ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-2.7, 2.7])

            plt.tight_layout()

            output_path = efficiency_plots_dir / f"efficiency_vs_eta_{tech_name}_{wp_name.replace(' ', '_').replace('.', '_')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        print(f"Individual {tech_name} eta efficiency plots saved to: {efficiency_plots_dir}")

    def _plot_individual_working_points_phi(self, results_dict, output_subdir=None):
        """Create separate efficiency plots for each working point (phi)."""
        # Only generate plots when output_subdir is specified to avoid duplicates in main run directory
        if output_subdir is None:
            print("Skipping individual phi working point plots: output_subdir not specified (avoiding duplicates)")
            return

        # Create subdirectory for organizing plots
        efficiency_dir = output_subdir / "efficiency_plots"
        efficiency_dir.mkdir(exist_ok=True)

        for wp_name, wp_data in results_dict.items():
            # Create individual efficiency plot
            _fig, ax = plt.subplots(1, 1, figsize=(10, 7))

            self._plot_metric_with_errors_eta_phi(
                ax,
                wp_data["phi_bins"],
                wp_data["efficiency"],
                wp_data["efficiency_err"],
                wp_data["counts"],
                "coral",
                f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})",
                "efficiency",
            )

            ax.set_xlabel("Truth Muon $\\phi$", fontsize=14)
            ax.set_ylabel("Hit Filter Efficiency", fontsize=14)
            ax.set_title(f"ATLAS Muon Hit Filter - {wp_name}", loc="left", fontsize=14)
            ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-3.2, 3.2])

            plt.tight_layout()

            output_path = efficiency_dir / f"efficiency_vs_phi_{wp_name.lower().replace(' ', '_').replace('.', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        print(f"Individual phi working point plots saved to: {efficiency_dir}")

    def _plot_individual_working_points_phi_technology(self, results_dict, tech_name, output_subdir=None):
        """Create individual phi plots for each working point and technology."""
        # Only generate plots when output_subdir is specified to avoid duplicates in main run directory
        if output_subdir is None:
            print(f"Skipping individual phi plots for {tech_name}: output_subdir not specified (avoiding duplicates)")
            return

        # Create efficiency_plots_<technology> subdirectory (unified for all coordinates)
        efficiency_plots_dir = output_subdir / f"efficiency_plots_{tech_name}"
        efficiency_plots_dir.mkdir(exist_ok=True)

        for wp_name, wp_data in results_dict.items():
            _fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            self._plot_metric_with_errors_eta_phi(
                ax,
                wp_data["phi_bins"],
                wp_data["efficiency"],
                wp_data["efficiency_err"],
                wp_data["counts"],
                "royalblue",
                f"{wp_name} (Purity: {wp_data['overall_purity']:.3f})",
                "efficiency",
            )

            ax.set_xlabel("Truth Muon $\\phi$", fontsize=14)
            ax.set_ylabel("Hit Filter Efficiency", fontsize=14)
            ax.set_title(f"ATLAS Muon Hit Filter - {tech_name} Technology - {wp_name}", loc="left", fontsize=12)
            ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="gray")
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="lightgray")
            ax.legend()
            ax.set_ylim([0.85, 1.05])
            ax.set_xlim([-3.2, 3.2])

            plt.tight_layout()

            output_path = efficiency_plots_dir / f"efficiency_vs_phi_{tech_name}_{wp_name.replace(' ', '_').replace('.', '_')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        print(f"Individual {tech_name} phi efficiency plots saved to: {efficiency_plots_dir}")

    def run_full_evaluation(self, skip_individual_plots=True, skip_technology_plots=True, skip_eta_phi_plots=True):
        """Run complete evaluation pipeline with all tracks, baseline filtered tracks, ML region tracks, time region tracks, and rejected tracks.

        Note: The rejected tracks region is now defined as tracks that fall outside the ML region
        (not outside the baseline region).

        Parameters:
        -----------
        skip_individual_plots : bool, default True
            Skip individual working point plots to save time/space
        skip_technology_plots : bool, default True
            Skip technology-specific plots to save time/space
        skip_eta_phi_plots : bool, default True
            Skip eta and phi binned plots to save time/space
        """
        print("Starting full evaluation of ATLAS muon hit filter with baseline, ML region, time region, and rejected tracks comparison...")

        # Monitor memory throughout
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        print(f"Starting memory usage: {start_memory:.1f} MB")

        # Collect data
        if not self.collect_data():
            print("Data collection failed, aborting evaluation")
            return None

        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after data collection: {current_memory:.1f} MB (+{current_memory - start_memory:.1f} MB)")

        # Backup original data for filtering
        self._backup_original_data()

        # Create baseline track filter
        baseline_mask, _baseline_rejected_mask, baseline_filter_stats = self.create_baseline_track_filter()

        # Create ML region track filter (defines the new rejected region)
        ml_region_mask, rejected_mask, ml_region_filter_stats = self.create_ml_region_track_filter()

        # Create time region track filter
        time_region_mask, _time_rejected_mask, time_region_filter_stats = self.create_time_region_track_filter()

        # Store statistics for all evaluations
        all_tracks_stats = {}
        baseline_stats = baseline_filter_stats.copy()  # Include baseline filtering stats
        ml_region_stats = ml_region_filter_stats.copy()  # Include ML region filtering stats
        time_region_stats = time_region_filter_stats.copy()  # Include time region filtering stats
        rejected_stats = ml_region_filter_stats.copy()  # Include ML region filtering stats (since rejected is based on ML region)

        # ===================================================================
        # PHASE 1: Evaluate ALL TRACKS
        # ===================================================================
        print("\n" + "=" * 80)
        print("PHASE 1: EVALUATING ALL TRACKS")
        print("=" * 80)

        # Generate core plots for all tracks
        print("\n=== Generating core evaluation plots (ALL TRACKS) ===")

        # Clear any cached ROC curves
        if hasattr(self, "_cached_roc"):
            delattr(self, "_cached_roc")

        # ROC curve
        all_roc_auc = self.plot_roc_curve(output_subdir=self.all_tracks_dir)
        all_tracks_stats["roc_auc"] = all_roc_auc
        gc.collect()

        # Efficiency vs pT (main plots only)
        self.plot_efficiency_vs_pt(skip_individual_plots=skip_individual_plots, output_subdir=self.all_tracks_dir)
        gc.collect()

        # Working point performance
        self.plot_working_point_performance(output_subdir=self.all_tracks_dir)
        gc.collect()

        # Track lengths (lightweight, always include)
        self.plot_track_lengths(output_subdir=self.all_tracks_dir)
        gc.collect()

        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after all tracks core plots: {current_memory:.1f} MB")

        # Store all tracks data statistics
        all_tracks_stats.update({
            "total_hits": len(self.all_logits),
            "true_hits": np.sum(self.all_true_labels),
            "noise_hits": np.sum(~self.all_true_labels),
            "unique_tracks": len(
                np.unique(np.column_stack([self.all_event_ids[self.all_true_labels], self.all_particle_ids[self.all_true_labels]]), axis=0)
            ),
        })

        # Optional plots (can be skipped for speed/space)
        if not skip_eta_phi_plots:
            print("\n=== Generating eta/phi plots (ALL TRACKS) ===")
            self.plot_efficiency_vs_eta(skip_individual_plots=skip_individual_plots, output_subdir=self.all_tracks_dir)
            self.plot_efficiency_vs_phi(skip_individual_plots=skip_individual_plots, output_subdir=self.all_tracks_dir)
            gc.collect()
        else:
            print("Skipping eta/phi plots (use --include-eta-phi to enable)")

        if not skip_technology_plots:
            print("\n=== Generating technology-specific plots (ALL TRACKS) ===")
            self._plot_efficiency_vs_pt_by_technology(DEFAULT_WORKING_POINTS, output_subdir=self.all_tracks_dir)
            gc.collect()
        else:
            print("Skipping technology-specific plots (use --include-tech to enable)")

        # ===================================================================
        # PHASE 2: Evaluate BASELINE FILTERED TRACKS
        # ===================================================================
        print("\n" + "=" * 80)
        print("PHASE 2: EVALUATING BASELINE FILTERED TRACKS")
        print("=" * 80)

        # Apply baseline filter to data
        self._apply_hit_filter(baseline_mask)

        # Clear any cached ROC curves for baseline evaluation
        if hasattr(self, "_cached_roc"):
            delattr(self, "_cached_roc")

        print(f"Baseline filtered data: {len(self.all_logits):,} hits ({len(self.all_logits) / len(self._original_logits) * 100:.1f}% of original)")

        # Generate core plots for baseline filtered tracks
        print("\n=== Generating core evaluation plots (BASELINE FILTERED) ===")

        # ROC curve
        baseline_roc_auc = self.plot_roc_curve(output_subdir=self.baseline_filtered_dir)
        baseline_stats["roc_auc"] = baseline_roc_auc
        gc.collect()

        # Efficiency vs pT (main plots only)
        self.plot_efficiency_vs_pt(skip_individual_plots=skip_individual_plots, output_subdir=self.baseline_filtered_dir)
        gc.collect()

        # Working point performance
        self.plot_working_point_performance(output_subdir=self.baseline_filtered_dir)
        gc.collect()

        # Track lengths (lightweight, always include)
        self.plot_track_lengths(output_subdir=self.baseline_filtered_dir)
        gc.collect()

        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after baseline filtered plots: {current_memory:.1f} MB")

        # Store baseline filtered data statistics
        baseline_stats.update({
            "total_hits": len(self.all_logits),
            "true_hits": np.sum(self.all_true_labels),
            "noise_hits": np.sum(~self.all_true_labels),
            "unique_tracks": len(
                np.unique(np.column_stack([self.all_event_ids[self.all_true_labels], self.all_particle_ids[self.all_true_labels]]), axis=0)
            ),
        })

        # Optional plots (can be skipped for speed/space)
        if not skip_eta_phi_plots:
            print("\n=== Generating eta/phi plots (BASELINE FILTERED) ===")
            self.plot_efficiency_vs_eta(skip_individual_plots=skip_individual_plots, output_subdir=self.baseline_filtered_dir)
            self.plot_efficiency_vs_phi(skip_individual_plots=skip_individual_plots, output_subdir=self.baseline_filtered_dir)
            gc.collect()
        else:
            print("Skipping eta/phi plots (use --include-eta-phi to enable)")

        if not skip_technology_plots:
            print("\n=== Generating technology-specific plots (BASELINE FILTERED) ===")
            self._plot_efficiency_vs_pt_by_technology(DEFAULT_WORKING_POINTS, output_subdir=self.baseline_filtered_dir)
            gc.collect()
        else:
            print("Skipping technology-specific plots (use --include-tech to enable)")

        # ===================================================================
        # PHASE 3: Evaluate ML REGION TRACKS
        # ===================================================================
        print("\n" + "=" * 80)
        print("PHASE 3: EVALUATING ML REGION TRACKS")
        print("=" * 80)

        # Restore original data and apply ML region filter
        self._restore_original_data()
        self._apply_hit_filter(ml_region_mask)

        # Clear any cached ROC curves for ML region evaluation
        if hasattr(self, "_cached_roc"):
            delattr(self, "_cached_roc")

        print(f"ML region data: {len(self.all_logits):,} hits ({len(self.all_logits) / len(self._original_logits) * 100:.1f}% of original)")

        # Generate core plots for ML region tracks
        print("\n=== Generating core evaluation plots (ML REGION) ===")

        # ROC curve
        ml_region_roc_auc = self.plot_roc_curve(output_subdir=self.ml_region_dir)
        ml_region_stats["roc_auc"] = ml_region_roc_auc
        gc.collect()

        # Efficiency vs pT (main plots only)
        self.plot_efficiency_vs_pt(skip_individual_plots=skip_individual_plots, output_subdir=self.ml_region_dir)
        gc.collect()

        # Working point performance
        self.plot_working_point_performance(output_subdir=self.ml_region_dir)
        gc.collect()

        # Track lengths (lightweight, always include)
        self.plot_track_lengths(output_subdir=self.ml_region_dir)
        gc.collect()

        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after ML region plots: {current_memory:.1f} MB")

        # Store ML region data statistics
        ml_region_stats.update({
            "total_hits": len(self.all_logits),
            "true_hits": np.sum(self.all_true_labels),
            "noise_hits": np.sum(~self.all_true_labels),
            "unique_tracks": len(
                np.unique(np.column_stack([self.all_event_ids[self.all_true_labels], self.all_particle_ids[self.all_true_labels]]), axis=0)
            ),
        })

        # Optional plots (can be skipped for speed/space)
        if not skip_eta_phi_plots:
            print("\n=== Generating eta/phi plots (ML REGION) ===")
            self.plot_efficiency_vs_eta(skip_individual_plots=skip_individual_plots, output_subdir=self.ml_region_dir)
            self.plot_efficiency_vs_phi(skip_individual_plots=skip_individual_plots, output_subdir=self.ml_region_dir)
            gc.collect()
        else:
            print("Skipping eta/phi plots (use --include-eta-phi to enable)")

        if not skip_technology_plots:
            print("\n=== Generating technology-specific plots (ML REGION) ===")
            self._plot_efficiency_vs_pt_by_technology(DEFAULT_WORKING_POINTS, output_subdir=self.ml_region_dir)
            gc.collect()
        else:
            print("Skipping technology-specific plots (use --include-tech to enable)")

        # ===================================================================
        # PHASE 5: Evaluate TIME REGION TRACKS (hit-level time filtering)
        # ===================================================================
        print("\n" + "=" * 80)
        print("PHASE 5: EVALUATING TIME REGION (ML tracks + time < 0.02 hit filter)")
        print("=" * 80)

        # Restore original data and apply time region filter
        self._restore_original_data()
        self._apply_hit_filter(time_region_mask)

        # Clear any cached ROC curves for time region evaluation
        if hasattr(self, "_cached_roc"):
            delattr(self, "_cached_roc")

        print(f"Time region data: {len(self.all_logits):,} hits ({len(self.all_logits) / len(self._original_logits) * 100:.1f}% of original)")

        # Generate core plots for time region tracks
        print("\n=== Generating core evaluation plots (TIME REGION) ===")

        # ROC curve
        time_region_roc_auc = self.plot_roc_curve(output_subdir=self.time_region_dir)
        time_region_stats["roc_auc"] = time_region_roc_auc
        gc.collect()

        # Efficiency vs pT (main plots only)
        self.plot_efficiency_vs_pt(skip_individual_plots=skip_individual_plots, output_subdir=self.time_region_dir)
        gc.collect()

        # Working point performance
        self.plot_working_point_performance(output_subdir=self.time_region_dir)
        gc.collect()

        # Track lengths (lightweight, always include)
        self.plot_track_lengths(output_subdir=self.time_region_dir)
        gc.collect()

        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after time region plots: {current_memory:.1f} MB")

        # Store time region data statistics
        time_region_stats.update({
            "total_hits": len(self.all_logits),
            "true_hits": np.sum(self.all_true_labels),
            "noise_hits": np.sum(~self.all_true_labels),
            "unique_tracks": len(
                np.unique(np.column_stack([self.all_event_ids[self.all_true_labels], self.all_particle_ids[self.all_true_labels]]), axis=0)
            ),
        })

        # Optional plots (can be skipped for speed/space)
        if not skip_eta_phi_plots:
            print("\n=== Generating eta/phi plots (TIME REGION) ===")
            self.plot_efficiency_vs_eta(skip_individual_plots=skip_individual_plots, output_subdir=self.time_region_dir)
            self.plot_efficiency_vs_phi(skip_individual_plots=skip_individual_plots, output_subdir=self.time_region_dir)
            gc.collect()
        else:
            print("Skipping eta/phi plots (use --include-eta-phi to enable)")

        if not skip_technology_plots:
            print("\n=== Generating technology-specific plots (TIME REGION) ===")
            self._plot_efficiency_vs_pt_by_technology(DEFAULT_WORKING_POINTS, output_subdir=self.time_region_dir)
            gc.collect()
        else:
            print("Skipping technology-specific plots (use --include-tech to enable)")

        # ===================================================================
        # PHASE 4: Evaluate REJECTED TRACKS (tracks outside ML region)
        # ===================================================================
        print("\n" + "=" * 80)
        print("PHASE 4: EVALUATING REJECTED TRACKS (outside ML region)")
        print("=" * 80)

        # Restore original data and apply rejected tracks filter
        self._restore_original_data()
        self._apply_hit_filter(rejected_mask)

        # Clear any cached ROC curves for rejected evaluation
        if hasattr(self, "_cached_roc"):
            delattr(self, "_cached_roc")

        print(f"Rejected tracks data: {len(self.all_logits):,} hits ({len(self.all_logits) / len(self._original_logits) * 100:.1f}% of original)")

        # Generate core plots for rejected tracks
        print("\n=== Generating core evaluation plots (REJECTED TRACKS) ===")

        # ROC curve
        rejected_roc_auc = self.plot_roc_curve(output_subdir=self.rejected_tracks_dir)
        rejected_stats["roc_auc"] = rejected_roc_auc
        gc.collect()

        # Efficiency vs pT (main plots only)
        self.plot_efficiency_vs_pt(skip_individual_plots=skip_individual_plots, output_subdir=self.rejected_tracks_dir)
        gc.collect()

        # Working point performance
        self.plot_working_point_performance(output_subdir=self.rejected_tracks_dir)
        gc.collect()

        # Track lengths (lightweight, always include)
        self.plot_track_lengths(output_subdir=self.rejected_tracks_dir)
        gc.collect()

        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after rejected tracks plots: {current_memory:.1f} MB")

        # Store rejected tracks data statistics
        rejected_stats.update({
            "total_hits": len(self.all_logits),
            "true_hits": np.sum(self.all_true_labels),
            "noise_hits": np.sum(~self.all_true_labels),
            "unique_tracks": len(
                np.unique(np.column_stack([self.all_event_ids[self.all_true_labels], self.all_particle_ids[self.all_true_labels]]), axis=0)
            ),
        })

        # Optional plots (can be skipped for speed/space)
        if not skip_eta_phi_plots:
            print("\n=== Generating eta/phi plots (REJECTED TRACKS) ===")
            self.plot_efficiency_vs_eta(skip_individual_plots=skip_individual_plots, output_subdir=self.rejected_tracks_dir)
            self.plot_efficiency_vs_phi(skip_individual_plots=skip_individual_plots, output_subdir=self.rejected_tracks_dir)
            gc.collect()
        else:
            print("Skipping eta/phi plots (use --include-eta-phi to enable)")

        if not skip_technology_plots:
            print("\n=== Generating technology-specific plots (REJECTED TRACKS) ===")
            self._plot_efficiency_vs_pt_by_technology(DEFAULT_WORKING_POINTS, output_subdir=self.rejected_tracks_dir)
            gc.collect()
        else:
            print("Skipping technology-specific plots (use --include-tech to enable)")

        # Restore original data for final cleanup
        self._restore_original_data()

        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"\nEvaluation complete! Final memory: {final_memory:.1f} MB")
        print(f"Peak memory usage: +{final_memory - start_memory:.1f} MB")
        print("Results saved to:")
        print(f"  All tracks: {self.all_tracks_dir}")
        print(f"  Baseline filtered: {self.baseline_filtered_dir}")
        print(f"  ML region: {self.ml_region_dir}")
        print(f"  Time region: {self.time_region_dir}")
        print(f"  Rejected tracks: {self.rejected_tracks_dir}")

        # Write comprehensive summary file
        self._write_comparative_evaluation_summary(
            all_tracks_stats,
            baseline_stats,
            ml_region_stats,
            time_region_stats,
            rejected_stats,
            skip_individual_plots,
            skip_technology_plots,
            skip_eta_phi_plots,
        )

        print("\nCOMPARISON SUMMARY:")
        print(f"All tracks AUC: {all_tracks_stats['roc_auc']:.4f}")
        print(f"Baseline filtered AUC: {baseline_stats['roc_auc']:.4f}")
        print(f"ML region AUC: {ml_region_stats['roc_auc']:.4f}")
        print(f"Time region AUC: {time_region_stats['roc_auc']:.4f}")
        print(f"Rejected tracks AUC: {rejected_stats['roc_auc']:.4f}")

        return all_tracks_stats, baseline_stats, ml_region_stats, time_region_stats, rejected_stats

    def _write_comparative_evaluation_summary(
        self,
        all_tracks_stats,
        baseline_stats,
        ml_region_stats,
        time_region_stats,
        rejected_stats,
        skip_individual_plots,
        skip_technology_plots,
        skip_eta_phi_plots,
    ):
        """Write a comprehensive summary comparing all tracks vs baseline filtered vs ML region vs time region vs rejected tracks."""
        summary_path = self.output_dir / "evaluation_summary_comparison.txt"

        with summary_path.open("w") as f:
            f.write("=" * 80 + "\n")
            f.write("ATLAS MUON HIT FILTER EVALUATION - COMPREHENSIVE SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max events processed: {self.max_events}\n\n")

            f.write("EVALUATION REGIONS:\n")
            f.write("1. ALL TRACKS: All tracks in the dataset (baseline for comparison)\n")
            f.write("2. BASELINE FILTERED: High-quality tracks meeting strict criteria\n")
            f.write("3. ML REGION: Tracks meeting ML training criteria (preprocessing filters)\n")
            f.write("4. TIME REGION: Tracks meeting ML criteria + time constraint (time < 0.02)\n")
            f.write("5. REJECTED TRACKS: Tracks outside the ML region\n\n")

            f.write("BASELINE FILTERING CRITERIA:\n")
            f.write("- Tracks must have hits in at least 3 different stations\n")
            f.write("- Each station must have at least 3 hits from the track\n")
            f.write("- This ensures tracks have at least 9 hits total\n")
            f.write("- Detector acceptance: 0.1 <= |eta| <= 2.7\n")
            f.write(f"- Minimum pT threshold: >= {self.min_pt} GeV\n\n")

            f.write("ML REGION FILTERING CRITERIA (matches preprocessing):\n")
            f.write("- Minimum 3 hits per track\n")
            f.write("- Detector acceptance: |eta| <= 2.7\n")
            f.write("- Minimum pT threshold: >= 5.0 GeV\n\n")

            f.write("TIME REGION FILTERING CRITERIA (ML + hit-level time filter):\n")
            f.write("- All ML region criteria (above) PLUS:\n")
            f.write("- Hit-level filtering: only keep hits with time < 2000*0.00001 (0.02)\n")
            f.write("- This removes high-time noise hits while keeping all qualifying tracks\n")
            f.write("- Improves signal-to-noise ratio based on discriminating time feature\n\n")

            f.write("NOTE: The rejected tracks region is defined as tracks that fall\n")
            f.write("outside the ML region (not outside the baseline region).\n\n")

            f.write("FILTERING STRATEGY:\n")
            f.write("- Keep ALL noise hits from all events in all categories\n")
            f.write("- Each category keeps true hits only from tracks meeting its criteria\n")
            f.write("- This maintains realistic signal-to-noise ratio for all evaluations\n\n")

            # Add detailed baseline filtering statistics
            if "total_tracks_checked" in baseline_stats:
                f.write("=" * 50 + "\n")
                f.write("BASELINE FILTERING BREAKDOWN\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total tracks evaluated: {baseline_stats['total_tracks_checked']:,}\n")
                f.write(
                    f"Tracks failing minimum hits (>=9): {baseline_stats['tracks_failed_min_hits']:,} ({baseline_stats['tracks_failed_min_hits'] / baseline_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks failing eta cuts (0.1 <= |eta| <= 2.7): {baseline_stats['tracks_failed_eta_cuts']:,} ({baseline_stats['tracks_failed_eta_cuts'] / baseline_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks failing pT cuts (pT >= {self.min_pt} GeV): {baseline_stats['tracks_failed_pt_cuts']:,} ({baseline_stats['tracks_failed_pt_cuts'] / baseline_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks failing station cuts (>=3 stations, >=3 hits/station): {baseline_stats['tracks_failed_station_cuts']:,} ({baseline_stats['tracks_failed_station_cuts'] / baseline_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks passing ALL criteria: {baseline_stats['tracks_passed_all_cuts']:,} ({baseline_stats['tracks_passed_all_cuts'] / baseline_stats['total_tracks_checked'] * 100:.2f}%)\n\n"
                )

                f.write("BASELINE FILTERING EFFICIENCY:\n")
                f.write(f"Track retention rate: {baseline_stats['tracks_passed_all_cuts'] / baseline_stats['total_tracks_checked'] * 100:.2f}%\n")
                f.write(f"Hit retention rate: {baseline_stats['baseline_hit_count'] / baseline_stats['total_hits'] * 100:.2f}%\n\n")

            # Add detailed ML region filtering statistics
            if "total_tracks_checked" in ml_region_stats:
                f.write("=" * 50 + "\n")
                f.write("ML REGION FILTERING BREAKDOWN\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total tracks evaluated: {ml_region_stats['total_tracks_checked']:,}\n")
                f.write(
                    f"Tracks failing minimum hits (>=3): {ml_region_stats['tracks_failed_min_hits']:,} ({ml_region_stats['tracks_failed_min_hits'] / ml_region_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks failing eta cuts (|eta| <= 2.7): {ml_region_stats['tracks_failed_eta_cuts']:,} ({ml_region_stats['tracks_failed_eta_cuts'] / ml_region_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks failing pT cuts (pT >= 5.0 GeV): {ml_region_stats['tracks_failed_pt_cuts']:,} ({ml_region_stats['tracks_failed_pt_cuts'] / ml_region_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks passing ALL criteria: {ml_region_stats['tracks_passed_all_cuts']:,} ({ml_region_stats['tracks_passed_all_cuts'] / ml_region_stats['total_tracks_checked'] * 100:.2f}%)\n\n"
                )

                f.write("ML REGION FILTERING EFFICIENCY:\n")
                f.write(f"Track retention rate: {ml_region_stats['tracks_passed_all_cuts'] / ml_region_stats['total_tracks_checked'] * 100:.2f}%\n")
                f.write(f"Hit retention rate: {ml_region_stats['ml_region_hit_count'] / ml_region_stats['total_hits'] * 100:.2f}%\n\n")

            # Add detailed time region filtering statistics
            if "total_tracks_checked" in time_region_stats:
                f.write("=" * 50 + "\n")
                f.write("TIME REGION FILTERING BREAKDOWN\n")
                f.write("=" * 50 + "\n")
                f.write("Hit-level time filtering (time < 0.02):\n")
                f.write(f"Hits removed by time filter: {time_region_stats.get('hits_removed_by_time_filter', 0):,}\n")
                f.write(f"True hits removed: {time_region_stats.get('true_hits_removed_by_time_filter', 0):,}\n")
                f.write(f"Noise hits removed: {time_region_stats.get('noise_hits_removed_by_time_filter', 0):,}\n\n")
                f.write("Track filtering (same as ML region):\n")
                f.write(f"Total tracks evaluated: {time_region_stats['total_tracks_checked']:,}\n")
                f.write(
                    f"Tracks failing minimum hits (>=3): {time_region_stats['tracks_failed_min_hits']:,} ({time_region_stats['tracks_failed_min_hits'] / time_region_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks failing eta cuts (|eta| <= 2.7): {time_region_stats['tracks_failed_eta_cuts']:,} ({time_region_stats['tracks_failed_eta_cuts'] / time_region_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks failing pT cuts (pT >= 5.0 GeV): {time_region_stats['tracks_failed_pt_cuts']:,} ({time_region_stats['tracks_failed_pt_cuts'] / time_region_stats['total_tracks_checked'] * 100:.2f}%)\n"
                )
                f.write(
                    f"Tracks passing ALL criteria: {time_region_stats['tracks_passed_all_cuts']:,} ({time_region_stats['tracks_passed_all_cuts'] / time_region_stats['total_tracks_checked'] * 100:.2f}%)\n\n"
                )

                f.write("TIME REGION FILTERING EFFICIENCY:\n")
                f.write(
                    f"Track retention rate: {time_region_stats['tracks_passed_all_cuts'] / time_region_stats['total_tracks_checked'] * 100:.2f}%\n"
                )
                f.write(f"Hit retention rate: {time_region_stats['time_region_hit_count'] / time_region_stats['total_hits'] * 100:.2f}%\n\n")

            f.write("=" * 50 + "\n")
            f.write("ALL TRACKS ANALYSIS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total hits analyzed: {all_tracks_stats['total_hits']:,}\n")
            f.write(f"True hits: {all_tracks_stats['true_hits']:,}\n")
            f.write(f"Noise hits: {all_tracks_stats['noise_hits']:,}\n")
            f.write(f"Unique tracks: {all_tracks_stats['unique_tracks']:,}\n")
            f.write(f"Truth hit ratio: {all_tracks_stats['true_hits'] / all_tracks_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"AUC Score: {all_tracks_stats['roc_auc']:.4f}\n\n")

            f.write("=" * 50 + "\n")
            f.write("BASELINE FILTERED TRACKS ANALYSIS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total hits analyzed: {baseline_stats['total_hits']:,}\n")
            f.write(f"True hits: {baseline_stats['true_hits']:,}\n")
            f.write(f"Noise hits: {baseline_stats['noise_hits']:,}\n")
            f.write(f"Unique tracks: {baseline_stats['unique_tracks']:,}\n")
            f.write(f"Truth hit ratio: {baseline_stats['true_hits'] / baseline_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"AUC Score: {baseline_stats['roc_auc']:.4f}\n\n")

            f.write("=" * 50 + "\n")
            f.write("ML REGION TRACKS ANALYSIS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total hits analyzed: {ml_region_stats['total_hits']:,}\n")
            f.write(f"True hits: {ml_region_stats['true_hits']:,}\n")
            f.write(f"Noise hits: {ml_region_stats['noise_hits']:,}\n")
            f.write(f"Unique tracks: {ml_region_stats['unique_tracks']:,}\n")
            f.write(f"Truth hit ratio: {ml_region_stats['true_hits'] / ml_region_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"AUC Score: {ml_region_stats['roc_auc']:.4f}\n\n")

            f.write("=" * 50 + "\n")
            f.write("TIME REGION TRACKS ANALYSIS (time < 0.02 hit filter)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total hits analyzed: {time_region_stats['total_hits']:,}\n")
            f.write(f"True hits: {time_region_stats['true_hits']:,}\n")
            f.write(f"Noise hits: {time_region_stats['noise_hits']:,}\n")
            f.write(f"Unique tracks: {time_region_stats['unique_tracks']:,}\n")
            f.write(f"Truth hit ratio: {time_region_stats['true_hits'] / time_region_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"AUC Score: {time_region_stats['roc_auc']:.4f}\n\n")

            f.write("=" * 50 + "\n")
            f.write("REJECTED TRACKS ANALYSIS (outside ML region)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total hits analyzed: {rejected_stats['total_hits']:,}\n")
            f.write(f"True hits: {rejected_stats['true_hits']:,}\n")
            f.write(f"Noise hits: {rejected_stats['noise_hits']:,}\n")
            f.write(f"Unique tracks: {rejected_stats['unique_tracks']:,}\n")
            f.write(f"Truth hit ratio: {rejected_stats['true_hits'] / rejected_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"AUC Score: {rejected_stats['roc_auc']:.4f}\n\n")

            f.write("=" * 50 + "\n")
            f.write("COMPARISON METRICS\n")
            f.write("=" * 50 + "\n")
            f.write("BASELINE vs ALL TRACKS:\n")
            f.write(f"Hit retention rate: {baseline_stats['total_hits'] / all_tracks_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"True hit retention: {baseline_stats['true_hits'] / all_tracks_stats['true_hits'] * 100:.2f}%\n")
            f.write(f"Noise hit retention: {baseline_stats['noise_hits'] / all_tracks_stats['noise_hits'] * 100:.2f}%\n")
            f.write(f"Track retention rate: {baseline_stats['unique_tracks'] / all_tracks_stats['unique_tracks'] * 100:.2f}%\n")
            f.write(f"AUC improvement: {baseline_stats['roc_auc'] - all_tracks_stats['roc_auc']:.4f}\n\n")

            f.write("ML REGION vs ALL TRACKS:\n")
            f.write(f"Hit retention rate: {ml_region_stats['total_hits'] / all_tracks_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"True hit retention: {ml_region_stats['true_hits'] / all_tracks_stats['true_hits'] * 100:.2f}%\n")
            f.write(f"Noise hit retention: {ml_region_stats['noise_hits'] / all_tracks_stats['noise_hits'] * 100:.2f}%\n")
            f.write(f"Track retention rate: {ml_region_stats['unique_tracks'] / all_tracks_stats['unique_tracks'] * 100:.2f}%\n")
            f.write(f"AUC improvement: {ml_region_stats['roc_auc'] - all_tracks_stats['roc_auc']:.4f}\n\n")

            f.write("TIME REGION vs ALL TRACKS:\n")
            f.write(f"Hit retention rate: {time_region_stats['total_hits'] / all_tracks_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"True hit retention: {time_region_stats['true_hits'] / all_tracks_stats['true_hits'] * 100:.2f}%\n")
            f.write(f"Noise hit retention: {time_region_stats['noise_hits'] / all_tracks_stats['noise_hits'] * 100:.2f}%\n")
            f.write(f"Track retention rate: {time_region_stats['unique_tracks'] / all_tracks_stats['unique_tracks'] * 100:.2f}%\n")
            f.write(f"AUC improvement: {time_region_stats['roc_auc'] - all_tracks_stats['roc_auc']:.4f}\n\n")

            f.write("TIME REGION vs ML REGION:\n")
            f.write(f"Hit retention rate: {time_region_stats['total_hits'] / ml_region_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"True hit retention: {time_region_stats['true_hits'] / ml_region_stats['true_hits'] * 100:.2f}%\n")
            f.write(f"Noise hit retention: {time_region_stats['noise_hits'] / ml_region_stats['noise_hits'] * 100:.2f}%\n")
            f.write(f"Track retention rate: {time_region_stats['unique_tracks'] / ml_region_stats['unique_tracks'] * 100:.2f}%\n")
            f.write(f"AUC improvement: {time_region_stats['roc_auc'] - ml_region_stats['roc_auc']:.4f}\n\n")

            f.write("REJECTED vs ALL TRACKS:\n")
            f.write(f"Hit retention rate: {rejected_stats['total_hits'] / all_tracks_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"True hit retention: {rejected_stats['true_hits'] / all_tracks_stats['true_hits'] * 100:.2f}%\n")
            f.write(f"Noise hit retention: {rejected_stats['noise_hits'] / all_tracks_stats['noise_hits'] * 100:.2f}%\n")
            f.write(f"Track retention rate: {rejected_stats['unique_tracks'] / all_tracks_stats['unique_tracks'] * 100:.2f}%\n")
            f.write(f"AUC improvement: {rejected_stats['roc_auc'] - all_tracks_stats['roc_auc']:.4f}\n\n")

            f.write("ML REGION vs BASELINE:\n")
            f.write(f"Hit retention rate: {ml_region_stats['total_hits'] / baseline_stats['total_hits'] * 100:.2f}%\n")
            f.write(f"True hit retention: {ml_region_stats['true_hits'] / baseline_stats['true_hits'] * 100:.2f}%\n")
            f.write(f"Noise hit retention: {ml_region_stats['noise_hits'] / baseline_stats['noise_hits'] * 100:.2f}%\n")
            f.write(f"Track retention rate: {ml_region_stats['unique_tracks'] / baseline_stats['unique_tracks'] * 100:.2f}%\n")
            f.write(f"AUC improvement: {ml_region_stats['roc_auc'] - baseline_stats['roc_auc']:.4f}\n\n")

            baseline_purity = baseline_stats["true_hits"] / baseline_stats["total_hits"]
            ml_region_purity = ml_region_stats["true_hits"] / ml_region_stats["total_hits"]
            time_region_purity = time_region_stats["true_hits"] / time_region_stats["total_hits"]
            rejected_purity = rejected_stats["true_hits"] / rejected_stats["total_hits"]
            all_tracks_purity = all_tracks_stats["true_hits"] / all_tracks_stats["total_hits"]
            f.write(f"Dataset purity (all tracks): {all_tracks_purity:.4f}\n")
            f.write(f"Dataset purity (baseline): {baseline_purity:.4f}\n")
            f.write(f"Dataset purity (ML region): {ml_region_purity:.4f}\n")
            f.write(f"Dataset purity (time region): {time_region_purity:.4f}\n")
            f.write(f"Dataset purity (rejected): {rejected_purity:.4f}\n")
            f.write(f"Baseline purity improvement: {baseline_purity - all_tracks_purity:.4f}\n")
            f.write(f"ML region purity improvement: {ml_region_purity - all_tracks_purity:.4f}\n")
            f.write(f"Time region purity improvement: {time_region_purity - all_tracks_purity:.4f}\n")
            f.write(f"Rejected purity change: {rejected_purity - all_tracks_purity:.4f}\n\n")

            f.write("PLOTS GENERATED:\n")
            f.write("- ROC curve (all five datasets)\n")
            f.write("- Efficiency vs pT (all five datasets)\n")
            f.write("- Working point performance (all five datasets)\n")
            f.write("- Track lengths (all five datasets)\n")

            if not skip_eta_phi_plots:
                f.write("- Efficiency vs eta/phi (all five datasets)\n")
            else:
                f.write("- Efficiency vs eta/phi (SKIPPED)\n")

            if not skip_technology_plots:
                f.write("- Technology-specific plots (all five datasets)\n")
            else:
                f.write("- Technology-specific plots (SKIPPED)\n")

            if not skip_individual_plots:
                f.write("- Individual working point plots (all five datasets)\n")
            else:
                f.write("- Individual working point plots (SKIPPED)\n")

            f.write("\nOutputs saved to:\n")
            f.write(f"  All tracks: {self.all_tracks_dir}\n")
            f.write(f"  Baseline filtered: {self.baseline_filtered_dir}\n")
            f.write(f"  ML region: {self.ml_region_dir}\n")
            f.write(f"  Time region: {self.time_region_dir}\n")
            f.write(f"  Rejected tracks: {self.rejected_tracks_dir}\n")

        print(f"Comparative evaluation summary saved to: {summary_path}")

    def _write_evaluation_summary(self, roc_auc, skip_individual_plots, skip_technology_plots, skip_eta_phi_plots):
        """Write a summary of the evaluation run."""
        summary_path = self.output_dir / "evaluation_summary.txt"

        with summary_path.open("w") as f:
            f.write("=" * 60 + "\n")
            f.write("ATLAS MUON HIT FILTER EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max events processed: {self.max_events}\n")
            f.write(f"Total hits analyzed: {len(self.all_logits):,}\n")
            f.write(f"True hits: {np.sum(self.all_true_labels):,}\n")
            f.write(f"AUC Score: {roc_auc:.4f}\n\n")

            f.write("PLOTS GENERATED:\n")
            f.write("- ROC curve\n")
            f.write("- Efficiency vs pT (main)\n")
            f.write("- Working point performance\n")
            f.write("- Track lengths\n")

            if not skip_eta_phi_plots:
                f.write("- Efficiency vs eta\n")
                f.write("- Efficiency vs phi\n")
            else:
                f.write("- Efficiency vs eta/phi (SKIPPED)\n")

            if not skip_technology_plots:
                f.write("- Technology-specific plots\n")
            else:
                f.write("- Technology-specific plots (SKIPPED)\n")

            if not skip_individual_plots:
                f.write("- Individual working point plots\n")
            else:
                f.write("- Individual working point plots (SKIPPED)\n")

            f.write(f"\nOutputs saved to: {self.output_dir}\n")

        print(f"Evaluation summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ATLAS muon hit filter using DataLoader (OPTIMIZED)")
    parser.add_argument(
        "--eval_path",
        "-e",
        type=str,
        default="/scratch/epoch=041-val_loss=0.00402_ml_test_data_156000_hdf5_eval_small_cuts.h5",
        help="Path to evaluation HDF5 file",
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="/scratch/ml_test_data_156000_hdf5",
        help="Path to processed test data directory",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument("--output_dir", "-o", type=str, default="./evaluation_results", help="Output directory for plots and results")
    parser.add_argument("--max_events", "-m", type=int, default=-1, help="Maximum number of events to process (for testing)")
    parser.add_argument(
        "--min_pt", type=float, default=5.0, help="Minimum pT threshold in GeV for track filtering (0.0 = no filter, 10.0 = exclude tracks < 10 GeV)"
    )
    parser.add_argument(
        "--max_pt",
        type=float,
        default=float("inf"),
        help="Maximum pT threshold in GeV for track filtering (inf = no filter, 50.0 = exclude tracks > 50 GeV)",
    )

    # Performance and output control options
    parser.add_argument("--include-individual-plots", action="store_true", help="Generate individual working point plots (slower, more files)")
    parser.add_argument("--include-tech", action="store_true", help="Generate technology-specific plots (slower)")
    parser.add_argument("--include-eta-phi", action="store_true", help="Generate eta and phi binned plots (slower)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip all optional plots (equivalent to default)")

    args = parser.parse_args()

    # Handle fast mode
    if args.fast:
        args.include_individual_plots = False
        args.include_tech = False
        args.include_eta_phi = False

    print("=" * 80)
    print("ATLAS MUON HIT FILTER EVALUATION (OPTIMIZED)")
    print("=" * 80)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Config file: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events if args.max_events > 0 else 'ALL'}")
    print(f"Minimum pT threshold: {args.min_pt} GeV {'(no filter)' if args.min_pt <= 0 else ''}")
    print(f"Include individual plots: {args.include_individual_plots}")
    print(f"Include technology plots: {args.include_tech}")
    print(f"Include eta/phi plots: {args.include_eta_phi}")
    print("=" * 80)

    # Enable stdout buffering for nohup
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    try:
        # Create evaluator and run
        evaluator = AtlasMuonEvaluatorDataLoader(
            eval_path=args.eval_path,
            data_dir=args.data_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_events=args.max_events,
            min_pt=args.min_pt,
            max_pt=args.max_pt,
        )

        evaluator.run_full_evaluation(
            skip_individual_plots=not args.include_individual_plots,
            skip_technology_plots=not args.include_tech,
            skip_eta_phi_plots=not args.include_eta_phi,
        )

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"\nERROR: Evaluation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
