#!/usr/bin/env python3
# ruff: noqa: E501,DTZ005,SIM108,EXE001
"""Evaluation script for Task 2: Track Validity Classification (track_valid).

This script evaluates the performance of the track validity classification task with:
1. Three categories: all tracks, baseline tracks, rejected tracks
2. On-the-fly data processing (no large memory storage)
3. ROC curves using track validity logits
4. Efficiency and fake rate plots over pt, eta, phi
5. Baseline filtering criteria from Task 1

Based on lessons learned from simple_task1_metrics.py
"""

import sys
import traceback

import h5py
import matplotlib as mpl
import numpy as np

mpl.use("Agg")  # Set backend before importing pyplot
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm

# Add the source directory to Python path
sys.path.append(str(Path(__file__).parent / "../../src"))
from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

warnings.filterwarnings("ignore")

# Set matplotlib style
plt.style.use("default")
plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (10, 6),
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "errorbar.capsize": 3,
})


class Task2TrackValidityEvaluator:
    """Evaluator for track validity classification task with baseline filtering."""

    def __init__(self, eval_path, data_dir, output_dir, max_events=None, random_seed=42):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.max_events = max_events
        self.random_seed = random_seed

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / f"task2_evaluation_{timestamp}"

        # Create output directory and subdirectories for categories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_tracks_dir = self.output_dir / "all_tracks"
        self.baseline_dir = self.output_dir / "baseline_tracks"
        self.ml_region_dir = self.output_dir / "ml_region_tracks"
        self.rejected_dir = self.output_dir / "rejected_tracks"

        for subdir in [self.all_tracks_dir, self.baseline_dir, self.ml_region_dir, self.rejected_dir]:
            subdir.mkdir(parents=True, exist_ok=True)

        print("Task 2 Evaluator initialized")
        print(f"Evaluation file: {eval_path}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max events: {max_events}")
        print(f"Random seed: {random_seed}")

    def setup_data_module(self):
        """Setup the data module for loading truth information."""
        print("Setting up data module...")

        self.data_module = AtlasMuonDataModule(
            train_dir=self.data_dir,
            val_dir=self.data_dir,
            test_dir=self.data_dir,
            num_workers=1,  # Reduced to avoid threading issues
            num_train=1,
            num_val=1,
            num_test=self.max_events or -1,
            batch_size=1,
            event_max_num_particles=2,
            inputs={
                "hit": [
                    "spacePoint_globEdgeHighX",
                    "spacePoint_globEdgeHighY",
                    "spacePoint_globEdgeHighZ",
                    "spacePoint_globEdgeLowX",
                    "spacePoint_globEdgeLowY",
                    "spacePoint_globEdgeLowZ",
                    "spacePoint_time",
                    "spacePoint_driftR",
                    "spacePoint_covXX",
                    "spacePoint_covXY",
                    "spacePoint_covYX",
                    "spacePoint_covYY",
                    "spacePoint_channel",
                    "spacePoint_layer",
                    "spacePoint_stationPhi",
                    "spacePoint_stationEta",
                    "spacePoint_stationIndex",
                    "spacePoint_technology",
                    "r",
                    "s",
                    "theta",
                    "phi",
                ]
            },
            targets={"particle": ["truthMuon_pt", "truthMuon_q", "truthMuon_eta", "truthMuon_phi"]},
        )

        self.data_module.setup("test")
        self.test_dataloader = self.data_module.test_dataloader(shuffle=False)

        # Create random sampling order for reproducible random evaluation
        dataset_size = len(self.test_dataloader.dataset)
        num_events_to_process = min(self.max_events, dataset_size) if self.max_events else dataset_size

        # Create random indices for sampling
        rng = np.random.default_rng(self.random_seed)  # Use configurable random seed
        self.random_indices = rng.permutation(dataset_size)[:num_events_to_process]

        print(f"Data module setup complete. Test dataset size: {dataset_size}")
        print(f"Random sampling {num_events_to_process} events with seed {self.random_seed}")
        print(f"Random indices: {self.random_indices[:10]}{'...' if len(self.random_indices) > 10 else ''}")

    def collect_and_process_data(self):
        """Collect and process data with random sampling, applying baseline and ML region filtering."""
        print("Collecting and processing data with random sampling and filtering...")

        # Data storage for all four categories
        all_data = {"logits": [], "true_validity": [], "predictions": [], "track_pts": [], "track_etas": [], "track_phis": [], "track_info": []}
        baseline_data = {"logits": [], "true_validity": [], "predictions": [], "track_pts": [], "track_etas": [], "track_phis": [], "track_info": []}
        ml_region_data = {"logits": [], "true_validity": [], "predictions": [], "track_pts": [], "track_etas": [], "track_phis": [], "track_info": []}
        rejected_data = {"logits": [], "true_validity": [], "predictions": [], "track_pts": [], "track_etas": [], "track_phis": [], "track_info": []}

        # Baseline filtering statistics (more granular)
        baseline_stats = {
            "total_tracks_checked": 0,
            "tracks_failed_min_hits": 0,
            "tracks_failed_eta_cuts": 0,
            "tracks_failed_pt_cuts": 0,
            "tracks_failed_insufficient_stations": 0,  # < 3 stations total
            "tracks_failed_hits_per_station": 0,  # >= 3 stations but < 3 stations with >= 3 hits
            "tracks_passed_all_cuts": 0,
        }

        # ML region filtering statistics
        ml_region_stats = {
            "total_tracks_checked": 0,
            "tracks_failed_min_hits": 0,
            "tracks_failed_eta_cuts": 0,
            "tracks_failed_pt_cuts": 0,
            "tracks_passed_all_cuts": 0,
        }

        with h5py.File(self.eval_path, "r") as pred_file:
            # Use the randomly sampled indices to access both dataset and eval file
            for _i, dataset_idx in enumerate(tqdm(self.random_indices, desc="Processing events")):
                # Get the batch from dataset using the random index
                batch = self.test_dataloader.dataset[dataset_idx]

                # Use the dataset index as the event_id for the eval file
                event_id = str(dataset_idx)

                if event_id not in pred_file:
                    print(f"Warning: Event {event_id} not found in predictions file")
                    continue

                # Get truth information from batch (dataset still has batch dimension of 1)
                inputs, targets = batch
                true_station_index = inputs["hit_spacePoint_stationIndex"][0].numpy().astype(np.int32)  # Remove batch dimension

                # Get predictions and truth
                pred_group = pred_file[event_id]

                # Get predictions
                track_valid_pred = pred_group["preds/final/track_valid/track_valid"][...]  # Shape: (1, 2)
                track_valid_logits = pred_group["outputs/final/track_valid/track_logit"][...]  # Shape: (1, 2)

                # Get predicted track parameters for binning (available for both real and fake tracks)
                pred_eta = pred_group["preds/final/parameter_regression/track_truthMuon_eta"][...]  # Shape: (1, 2)
                pred_phi = pred_group["preds/final/parameter_regression/track_truthMuon_phi"][...]  # Shape: (1, 2)

                # Check if pt predictions are available
                has_pt_pred = "preds/final/parameter_regression/track_truthMuon_pt" in pred_group
                if has_pt_pred:
                    pred_pt = pred_group["preds/final/parameter_regression/track_truthMuon_pt"][...]  # Shape: (1, 2)
                else:
                    pred_pt = None

                # Get truth
                true_particle_valid = targets["particle_valid"][0]  # Remove batch dimension
                true_hit_assignments = targets["particle_hit_valid"][0]  # Remove batch dimension

                # Extract true particle parameters (only for valid particles)
                valid_particles = true_particle_valid.numpy()
                int(valid_particles.sum())

                # Process both potential tracks (max 2 tracks per event)
                for track_idx in range(2):
                    # Get track validity predictions and truth
                    predicted_track_valid = bool(track_valid_pred[0, track_idx])
                    true_particle_exists = bool(true_particle_valid[track_idx])
                    track_logit = float(track_valid_logits[0, track_idx])

                    # Use predicted kinematic parameters for binning (works for both real and fake tracks)
                    track_eta = pred_eta[0, track_idx].item()
                    track_phi = pred_phi[0, track_idx].item()
                    track_pt = pred_pt[0, track_idx].item() if has_pt_pred else 0.0

                    # For filtering, we need truth parameters if the particle exists
                    if true_particle_exists:
                        # Get true track parameters (handle dataset structure with batch dimension)
                        eta_tensor = targets["particle_truthMuon_eta"][0, track_idx]  # Remove batch dimension
                        phi_tensor = targets["particle_truthMuon_phi"][0, track_idx]  # Remove batch dimension
                        pt_tensor = targets["particle_truthMuon_pt"][0, track_idx]  # Remove batch dimension

                        # Handle potential multi-dimensional tensors by taking the first element if needed
                        true_eta = eta_tensor.item() if eta_tensor.numel() == 1 else eta_tensor[0].item()
                        true_phi = phi_tensor.item() if phi_tensor.numel() == 1 else phi_tensor[0].item()
                        true_pt = pt_tensor.item() if pt_tensor.numel() == 1 else pt_tensor[0].item()
                    else:
                        # For fake tracks, use predicted parameters for consistency
                        true_eta = track_eta
                        true_phi = track_phi
                        true_pt = track_pt

                    track_info = {
                        "pt": true_pt,
                        "eta": true_eta,
                        "phi": true_phi,
                        "event_id": dataset_idx,
                        "track_id": track_idx,
                        "true_particle_exists": true_particle_exists,
                    }

                    # Add to all tracks
                    all_data["logits"].append(track_logit)
                    all_data["true_validity"].append(true_particle_exists)
                    all_data["predictions"].append(predicted_track_valid)
                    all_data["track_pts"].append(track_pt)
                    all_data["track_etas"].append(track_eta)
                    all_data["track_phis"].append(track_phi)
                    all_data["track_info"].append(track_info)

                    baseline_passed = False
                    ml_region_passed = False

                    # Only apply filtering to true particles
                    if true_particle_exists:
                        # Apply baseline filtering
                        baseline_stats["total_tracks_checked"] += 1

                        # Apply ML region filtering (in parallel with baseline)
                        ml_region_stats["total_tracks_checked"] += 1

                        # Get hit assignments for filtering
                        true_hits = true_hit_assignments[track_idx].numpy()
                        track_mask = true_hits.astype(bool)

                        # Check baseline criteria
                        total_hits = np.sum(true_hits)

                        baseline_passed = True
                        ml_region_passed = True

                        # Baseline criterion 1: >= 9 hits
                        if total_hits < 9:
                            baseline_stats["tracks_failed_min_hits"] += 1
                            baseline_passed = False

                        # Baseline criterion 2: eta cuts (0.1 <= |eta| <= 2.7)
                        if np.abs(true_eta) < 0.1 or np.abs(true_eta) > 2.7:
                            baseline_stats["tracks_failed_eta_cuts"] += 1
                            baseline_passed = False

                        # Baseline criterion 3: pt >= 3.0 GeV
                        if true_pt < 3.0:
                            baseline_stats["tracks_failed_pt_cuts"] += 1
                            baseline_passed = False

                        # Baseline criterion 4: Station requirements (>= 3 stations, >= 3 hits per station)
                        unique_stations, station_counts = np.unique(true_station_index[track_mask], return_counts=True)
                        if len(unique_stations) < 3:
                            baseline_stats["tracks_failed_insufficient_stations"] += 1
                            baseline_passed = False
                        elif np.sum(station_counts >= 3) < 3:  # Need at least 3 stations with >= 3 hits each
                            baseline_stats["tracks_failed_hits_per_station"] += 1
                            baseline_passed = False

                        # ML region criteria (pt >= 5.0 GeV, |eta| <= 2.7, >= 3 hits)
                        if total_hits < 3:
                            ml_region_stats["tracks_failed_min_hits"] += 1
                            ml_region_passed = False

                        if np.abs(true_eta) > 2.7:
                            ml_region_stats["tracks_failed_eta_cuts"] += 1
                            ml_region_passed = False

                        if true_pt < 5.0:
                            ml_region_stats["tracks_failed_pt_cuts"] += 1
                            ml_region_passed = False

                        # Update passed counts
                        if baseline_passed:
                            baseline_stats["tracks_passed_all_cuts"] += 1

                        if ml_region_passed:
                            ml_region_stats["tracks_passed_all_cuts"] += 1

                    # Add to appropriate categories (including fake tracks in all categories for ROC calculation)
                    # Note: fake tracks are always included so we can compute ROC curves properly

                    # For baseline: include if real track passes baseline OR if fake track
                    if (true_particle_exists and baseline_passed) or not true_particle_exists:
                        baseline_data["logits"].append(track_logit)
                        baseline_data["true_validity"].append(true_particle_exists)
                        baseline_data["predictions"].append(predicted_track_valid)
                        baseline_data["track_pts"].append(track_pt)
                        baseline_data["track_etas"].append(track_eta)
                        baseline_data["track_phis"].append(track_phi)
                        baseline_data["track_info"].append(track_info)

                    # For ML region: include if real track passes ML region OR if fake track
                    if (true_particle_exists and ml_region_passed) or not true_particle_exists:
                        ml_region_data["logits"].append(track_logit)
                        ml_region_data["true_validity"].append(true_particle_exists)
                        ml_region_data["predictions"].append(predicted_track_valid)
                        ml_region_data["track_pts"].append(track_pt)
                        ml_region_data["track_etas"].append(track_eta)
                        ml_region_data["track_phis"].append(track_phi)
                        ml_region_data["track_info"].append(track_info)

                    # For rejected: include only if real track fails ML region criteria
                    if true_particle_exists and not ml_region_passed:
                        rejected_data["logits"].append(track_logit)
                        rejected_data["true_validity"].append(true_particle_exists)
                        rejected_data["predictions"].append(predicted_track_valid)
                        rejected_data["track_pts"].append(track_pt)
                        rejected_data["track_etas"].append(track_eta)
                        rejected_data["track_phis"].append(track_phi)
                        rejected_data["track_info"].append(track_info)

        # Convert to numpy arrays
        for data_dict in [all_data, baseline_data, ml_region_data, rejected_data]:
            for key in data_dict:
                if key != "track_info":  # Don't convert track_info to numpy array
                    data_dict[key] = np.array(data_dict[key])

        print("\nData collection complete!")
        print(f"Total events processed: {len(self.random_indices)}")
        print()
        print("Baseline Filtering Statistics:")
        print(f"  Total tracks checked: {baseline_stats['total_tracks_checked']}")
        print(
            f"  Failed minimum hits (>=9): {baseline_stats['tracks_failed_min_hits']} ({baseline_stats['tracks_failed_min_hits'] / baseline_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed eta cuts (0.1 <= |eta| <= 2.7): {baseline_stats['tracks_failed_eta_cuts']} ({baseline_stats['tracks_failed_eta_cuts'] / baseline_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed pt cuts (pt >= 3.0 GeV): {baseline_stats['tracks_failed_pt_cuts']} ({baseline_stats['tracks_failed_pt_cuts'] / baseline_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed insufficient stations (<3 stations): {baseline_stats['tracks_failed_insufficient_stations']} ({baseline_stats['tracks_failed_insufficient_stations'] / baseline_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed hits per station (<3 stations with >=3 hits): {baseline_stats['tracks_failed_hits_per_station']} ({baseline_stats['tracks_failed_hits_per_station'] / baseline_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Tracks passing all cuts: {baseline_stats['tracks_passed_all_cuts']} ({baseline_stats['tracks_passed_all_cuts'] / baseline_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print()
        print("ML Region Filtering Statistics:")
        print(f"  Total tracks checked: {ml_region_stats['total_tracks_checked']}")
        print(
            f"  Failed minimum hits (>=3): {ml_region_stats['tracks_failed_min_hits']} ({ml_region_stats['tracks_failed_min_hits'] / ml_region_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed eta cuts (|eta| <= 2.7): {ml_region_stats['tracks_failed_eta_cuts']} ({ml_region_stats['tracks_failed_eta_cuts'] / ml_region_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Failed pt cuts (pt >= 5.0 GeV): {ml_region_stats['tracks_failed_pt_cuts']} ({ml_region_stats['tracks_failed_pt_cuts'] / ml_region_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print(
            f"  Tracks passing all cuts: {ml_region_stats['tracks_passed_all_cuts']} ({ml_region_stats['tracks_passed_all_cuts'] / ml_region_stats['total_tracks_checked'] * 100:.1f}%)"
        )
        print()
        print("Category Statistics:")
        print(f"  All tracks: {len(all_data['logits'])}")
        print(f"  Baseline tracks: {len(baseline_data['logits'])}")
        print(f"  ML region tracks: {len(ml_region_data['logits'])}")
        print(f"  Rejected tracks: {len(rejected_data['logits'])}")

        return all_data, baseline_data, ml_region_data, rejected_data, baseline_stats, ml_region_stats

    def _calculate_data_driven_bins(self, data, variable, num_bins=21):
        """Calculate data-driven bins for a kinematic variable.

        Uses truth data ranges to determine bin boundaries, but the bins
        will be applied to prediction data for plotting.
        """
        # Extract the truth values from track_info for range determination
        if len(data["track_info"]) == 0:
            # Fallback to fixed bins if no data
            if variable == "pt":
                return np.linspace(0, 200, num_bins)
            if variable == "eta":
                return np.linspace(-3, 3, num_bins)
            if variable == "phi":
                return np.linspace(-np.pi, np.pi, num_bins)

        # Get truth values from track_info for range calculation
        # Only use real tracks (where true_particle_exists is True) for truth range
        truth_values = np.array([track[variable] for track in data["track_info"] if track["true_particle_exists"]])

        if len(truth_values) == 0:
            # Fallback to fixed bins if no real tracks
            if variable == "pt":
                return np.linspace(0, 200, num_bins)
            if variable == "eta":
                return np.linspace(-3, 3, num_bins)
            if variable == "phi":
                return np.linspace(-np.pi, np.pi, num_bins)

        if variable == "pt":
            # PT: min of truth data to 200
            min_val = np.min(truth_values)
            print(f"PT min from REAL truth data: {min_val}", "+" * 80)
            max_val = 200.0
            return np.linspace(min_val, max_val, num_bins)
        if variable == "eta":
            # Eta: min to max of truth data
            min_val = np.min(truth_values)
            max_val = np.max(truth_values)
            print(f"ETA range from REAL truth data: {min_val:.3f} to {max_val:.3f}", "+" * 80)
            return np.linspace(min_val, max_val, num_bins)
        if variable == "phi":
            # Phi: keep standard range
            return np.linspace(-np.pi, np.pi, num_bins)

        return np.linspace(0, 1, num_bins)  # Fallback

    def calculate_efficiency_fakerate_by_variable(self, data, variable="pt", bins=None):
        """Calculate efficiency and fake rate binned by a kinematic variable.

        Bins are calculated from truth data ranges but applied to prediction data.
        """
        if bins is None:
            bins = self._calculate_data_driven_bins(data, variable)

        # Extract the prediction variable values for binning
        if variable == "pt":
            var_values = data["track_pts"]  # prediction data
        elif variable == "eta":
            var_values = data["track_etas"]  # prediction data
        elif variable == "phi":
            var_values = data["track_phis"]  # prediction data

        predictions = data["predictions"]
        true_validity = data["true_validity"]

        # Calculate bin indices
        bin_indices = np.digitize(var_values, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        efficiencies = []
        fake_rates = []
        bin_centers = []
        eff_errors = []
        fake_errors = []

        for i in range(len(bins) - 1):
            mask = bin_indices == i

            if mask.sum() == 0:
                continue

            # Get predictions and truth for this bin
            bin_predictions = predictions[mask]
            bin_truth = true_validity[mask]
            n_total = len(bin_predictions)

            # Calculate efficiency and fake rate
            n_true = bin_truth.sum()  # Number of true tracks
            (~bin_truth).sum()  # Number of fake tracks

            # True positives: correctly identified real tracks
            true_positives = (bin_predictions & bin_truth).sum()
            # False positives: incorrectly identified fake tracks as real
            false_positives = (bin_predictions & ~bin_truth).sum()
            # False negatives: missed real tracks
            (~bin_predictions & bin_truth).sum()

            # Efficiency = TP / (TP + FN) = TP / n_true
            if n_true > 0:
                efficiency = true_positives / n_true
                eff_error = np.sqrt(efficiency * (1 - efficiency) / n_true)
            else:
                efficiency = 0.0
                eff_error = 0.0

            # Fake rate = FP / Total Predictions = false positives / all predictions
            if n_total > 0:
                fake_rate = false_positives / n_total
                fake_error = np.sqrt(fake_rate * (1 - fake_rate) / n_total)
            else:
                fake_rate = 0.0
                fake_error = 0.0

            efficiencies.append(efficiency)
            fake_rates.append(fake_rate)
            eff_errors.append(eff_error)
            fake_errors.append(fake_error)
            bin_centers.append((bins[i] + bins[i + 1]) / 2)

        return np.array(bin_centers), np.array(efficiencies), np.array(fake_rates), np.array(eff_errors), np.array(fake_errors)

    def plot_efficiency_fakerate_vs_variable(self, data, variable="pt", output_dir=None, category_name=""):
        """Plot efficiency and fake rate vs a kinematic variable."""
        # Use data-driven bins
        bins = self._calculate_data_driven_bins(data, variable)

        bin_centers, efficiencies, fake_rates, eff_errors, fake_errors = self.calculate_efficiency_fakerate_by_variable(data, variable, bins)

        if len(bin_centers) == 0:
            print(f"Warning: No data points for {variable} plots in {category_name}")
            return

        # Create the plot with step style and error bands
        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Efficiency plot
        for i, (lhs, rhs, eff_val, eff_err) in enumerate(zip(bins[:-1], bins[1:], efficiencies, eff_errors, strict=False)):
            color = "blue"

            # Create error band
            if eff_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(eff_val + eff_err, 1.0)
                y_lower = max(eff_val - eff_err, 0.0)
                ax1.fill_between(point_in_range, y_upper, y_lower, color=color, alpha=0.3, label="binomial err - Efficiency" if i == 0 else "")

            # Step plot
            ax1.step([lhs, rhs], [eff_val, eff_val], color=color, linewidth=2.5, label="Efficiency" if i == 0 else "")

        ax1.set_ylabel("Track Validity Efficiency")
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Track Validity Efficiency vs {variable.capitalize()} - {category_name}")

        # Fake rate plot
        for i, (lhs, rhs, fake_val, fake_err) in enumerate(zip(bins[:-1], bins[1:], fake_rates, fake_errors, strict=False)):
            color = "red"

            # Create error band
            if fake_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(fake_val + fake_err, 1.0)
                y_lower = max(fake_val - fake_err, 0.0)
                ax2.fill_between(point_in_range, y_upper, y_lower, color=color, alpha=0.3, label="binomial err - Fake Rate" if i == 0 else "")

            # Step plot
            ax2.step([lhs, rhs], [fake_val, fake_val], color=color, linewidth=2.5, label="Fake Rate" if i == 0 else "")

        ax2.set_ylabel("Track Validity Fake Rate")
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"Track Validity Fake Rate vs {variable.capitalize()} - {category_name}")

        # Set x-axis labels
        if variable == "pt":
            ax2.set_xlabel("$p_T$ [GeV]")
        elif variable == "eta":
            ax2.set_xlabel("$\\eta$")
        elif variable == "phi":
            ax2.set_xlabel("$\\phi$ [rad]")

        plt.tight_layout()

        # Save the plot
        output_path = output_dir / f"track_validity_efficiency_fakerate_vs_{variable}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved {category_name} efficiency/fake rate vs {variable} plot to {output_path}")

    def plot_roc_curve(self, data, output_dir=None, category_name=""):
        """Plot ROC curve using track validity logits."""
        logits = data["logits"]
        true_validity = data["true_validity"]

        if len(logits) == 0:
            print(f"Warning: No logits available for ROC curve in {category_name}")
            return None

        # Check if we have both positive and negative examples
        n_positive = true_validity.sum()
        n_negative = (~true_validity).sum()

        if n_positive == 0 or n_negative == 0:
            print(f"Warning: Cannot create ROC curve for {category_name} - need both positive and negative examples")
            print(f"  Positive examples: {n_positive}, Negative examples: {n_negative}")
            return None

        # Calculate ROC curve
        fpr, tpr, _thresholds = roc_curve(true_validity, logits)
        roc_auc = auc(fpr, tpr)

        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for Track Validity Classification - {category_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Save the plot
        output_path = output_dir / "roc_curve_track_validity.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved {category_name} ROC curve to {output_path} (AUC: {roc_auc:.4f})")

        return roc_auc

    def plot_logit_distributions(self, data, output_dir=None, category_name=""):
        """Plot distributions of logits for true and fake tracks."""
        logits = data["logits"]
        true_validity = data["true_validity"]

        if len(logits) == 0:
            print(f"Warning: No logits available for distributions in {category_name}")
            return

        true_track_logits = logits[true_validity]
        fake_track_logits = logits[~true_validity]

        plt.figure(figsize=(10, 6))

        if len(fake_track_logits) > 0:
            plt.hist(fake_track_logits, bins=50, alpha=0.7, label=f"Fake tracks (n={len(fake_track_logits)})", color="red", density=False)

        if len(true_track_logits) > 0:
            plt.hist(true_track_logits, bins=50, alpha=0.7, label=f"True tracks (n={len(true_track_logits)})", color="blue", density=False)

        plt.xlabel("Track Validity Logit")
        plt.ylabel("Count")
        plt.title(f"Distribution of Track Validity Logits - {category_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        output_path = output_dir / "track_validity_logit_distributions.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved {category_name} logit distributions to {output_path}")

    def write_category_summary(self, data, output_dir, category_name, roc_auc):
        """Write evaluation summary for a category."""
        summary_path = output_dir / f"{category_name.lower().replace(' ', '_')}_summary.txt"

        data["logits"]
        predictions = data["predictions"]
        true_validity = data["true_validity"]

        if len(predictions) == 0:
            print(f"Warning: No data for {category_name} summary")
            return

        # Calculate overall statistics
        n_total = len(predictions)
        n_true_tracks = true_validity.sum()
        n_fake_tracks = n_total - n_true_tracks
        n_pred_valid = predictions.sum()
        n_pred_invalid = n_total - n_pred_valid

        # Calculate confusion matrix elements
        true_positives = (predictions & true_validity).sum()
        false_positives = (predictions & ~true_validity).sum()
        true_negatives = (~predictions & ~true_validity).sum()
        false_negatives = (~predictions & true_validity).sum()

        # Calculate metrics
        efficiency = true_positives / n_true_tracks if n_true_tracks > 0 else 0  # TP / (TP + FN)
        fake_rate = false_positives / n_total if n_total > 0 else 0  # FP / Total predictions

        with summary_path.open("w") as f:
            f.write(f"TASK 2: TRACK VALIDITY CLASSIFICATION - {category_name.upper()} SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Category: {category_name}\n\n")

            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total tracks analyzed: {n_total:,}\n")
            f.write(f"True valid tracks: {n_true_tracks:,}\n")
            f.write(f"True invalid tracks: {n_fake_tracks:,}\n")
            f.write(f"Predicted valid tracks: {n_pred_valid:,}\n")
            f.write(f"Predicted invalid tracks: {n_pred_invalid:,}\n\n")

            f.write("CONFUSION MATRIX\n")
            f.write("-" * 16 + "\n")
            f.write(f"True Positives: {true_positives:,}\n")
            f.write(f"False Positives: {false_positives:,}\n")
            f.write(f"True Negatives: {true_negatives:,}\n")
            f.write(f"False Negatives: {false_negatives:,}\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Efficiency: {efficiency:.4f}\n")
            f.write(f"Fake Rate: {fake_rate:.4f}\n")

            if roc_auc is not None:
                f.write(f"ROC AUC: {roc_auc:.4f}\n")
            else:
                f.write("ROC AUC: N/A (insufficient data)\n")

            f.write(f"\nGenerated at: {datetime.now()}\n")

        print(f"Summary for {category_name} written to {summary_path}")

    def write_comparative_summary(self, all_results, baseline_stats, ml_region_stats):
        """Write comprehensive summary comparing all categories."""
        summary_path = self.output_dir / "task2_comparative_summary.txt"

        with summary_path.open("w") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TASK 2: TRACK VALIDITY CLASSIFICATION - COMPARATIVE SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluation file: {self.eval_path}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Max events processed: {self.max_events}\n\n")

            # Write filtering statistics
            f.write("BASELINE FILTERING STATISTICS\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total tracks checked: {baseline_stats.get('total_tracks_checked', 0):,}\n")
            f.write(f"Failed minimum hits (>=9): {baseline_stats.get('tracks_failed_min_hits', 0):,}\n")
            f.write(f"Failed eta cuts (0.1 <= |eta| <= 2.7): {baseline_stats.get('tracks_failed_eta_cuts', 0):,}\n")
            f.write(f"Failed pt cuts (pt >= 3.0 GeV): {baseline_stats.get('tracks_failed_pt_cuts', 0):,}\n")
            f.write(f"Failed insufficient stations (<3 stations): {baseline_stats.get('tracks_failed_insufficient_stations', 0):,}\n")
            f.write(f"Failed hits per station (<3 stations with >=3 hits): {baseline_stats.get('tracks_failed_hits_per_station', 0):,}\n")
            f.write(f"Tracks passing all cuts: {baseline_stats.get('tracks_passed_all_cuts', 0):,}\n")

            f.write("\nML REGION FILTERING STATISTICS\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total tracks checked: {ml_region_stats.get('total_tracks_checked', 0):,}\n")
            f.write(f"Failed minimum hits (>=3): {ml_region_stats.get('tracks_failed_min_hits', 0):,}\n")
            f.write(f"Failed eta cuts (|eta| <= 2.7): {ml_region_stats.get('tracks_failed_eta_cuts', 0):,}\n")
            f.write(f"Failed pt cuts (pt >= 5.0 GeV): {ml_region_stats.get('tracks_failed_pt_cuts', 0):,}\n")
            f.write(f"Tracks passing all cuts: {ml_region_stats.get('tracks_passed_all_cuts', 0):,}\n")

            # Write track counts per category
            f.write("\nTRACKS ANALYZED PER CATEGORY\n")
            f.write("-" * 30 + "\n")
            categories = ["All Tracks", "Baseline Tracks", "ML Region Tracks", "Rejected Tracks"]
            for category in categories:
                if category in all_results:
                    num_tracks = all_results[category].get("num_tracks", 0)
                    f.write(f"{category}: {num_tracks:,} tracks\n")
                else:
                    f.write(f"{category}: 0 tracks\n")
            f.write("\n")

            # Write comparative metrics
            f.write("COMPARATIVE METRICS\n")
            f.write("-" * 20 + "\n")

            f.write(f"{'Category':<20}{'Num Tracks':<15}{'Efficiency':<12}{'Fake Rate':<12}{'ROC AUC':<10}\n")
            f.write("-" * 70 + "\n")

            for category in categories:
                if category in all_results:
                    result = all_results[category]
                    f.write(f"{category:<20}{result['num_tracks']:<15}{result['efficiency']:<12.4f}{result['fake_rate']:<12.4f}")
                    if result.get("roc_auc") is not None:
                        f.write(f"{result['roc_auc']:<10.4f}\n")
                    else:
                        f.write(f"{'N/A':<10}\n")
                else:
                    f.write(f"{category:<20}{'0':<15}{'N/A':<12}{'N/A':<12}{'N/A':<10}\n")

            f.write("\n")

        print(f"Comparative summary written to {summary_path}")

    def evaluate_category(self, data, category_name):
        """Evaluate a single category of tracks."""
        # Create output directory for this category
        category_dir = self.output_dir / category_name.lower().replace(" ", "_")
        category_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        true_validity = np.array(data["true_validity"], dtype=bool)
        predictions = np.array(data["predictions"], dtype=bool)
        logits = np.array(data["logits"])

        # Calculate basic metrics
        n_total = len(predictions)
        n_true_tracks = true_validity.sum()
        n_fake_tracks = n_total - n_true_tracks

        if n_total == 0:
            return {"num_tracks": 0, "efficiency": 0.0, "fake_rate": 0.0, "roc_auc": None}

        true_positives = (predictions & true_validity).sum()
        false_positives = (predictions & ~true_validity).sum()
        (~predictions & true_validity).sum()
        (~predictions & ~true_validity).sum()

        # Calculate efficiency and fake rate
        efficiency = true_positives / n_true_tracks if n_true_tracks > 0 else 0.0
        fake_rate = false_positives / n_total if n_total > 0 else 0.0

        # Calculate ROC AUC if we have both classes
        roc_auc = None
        if n_true_tracks > 0 and n_fake_tracks > 0:
            try:
                roc_auc = roc_auc_score(true_validity, logits)  # Use logits, not binary predictions
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"Could not calculate ROC AUC for {category_name}: {e}")

        # Generate plots
        print("Generating plots...")

        # ROC curve
        try:
            self.plot_roc_curve(data, category_dir, category_name)
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error creating ROC curve: {e}")

        # Logit distributions
        try:
            self.plot_logit_distributions(data, category_dir, category_name)
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error creating logit distributions: {e}")

        # Efficiency/fake rate vs kinematic variables
        for variable in ["pt", "eta", "phi"]:
            try:
                self.plot_efficiency_fakerate_vs_variable(data, variable, category_dir, category_name)
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"Error creating efficiency/fake rate vs {variable} plot: {e}")

        # Write individual summary
        try:
            self.write_category_summary(data, category_dir, category_name, roc_auc)
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error writing category summary: {e}")

        return {"num_tracks": n_total, "efficiency": efficiency, "fake_rate": fake_rate, "roc_auc": roc_auc}

    def run_evaluation_with_categories(self):
        """Run evaluation for all track categories."""
        print("\n" + "=" * 80)
        print("STARTING TASK 2 TRACK VALIDITY EVALUATION")
        print("=" * 80)

        # Setup data module first
        self.setup_data_module()

        # Collect and process data for all categories
        print("\nCollecting and processing track data...")
        all_data, baseline_data, ml_region_data, rejected_data, baseline_stats, ml_region_stats = self.collect_and_process_data()

        # Reorganize data into the expected structure
        organized_data = {
            "All Tracks": all_data,
            "Baseline Tracks": baseline_data,
            "ML Region Tracks": ml_region_data,
            "Rejected Tracks": rejected_data,
        }

        # Initialize results dictionary
        all_results = {}

        # Evaluate each category
        categories = ["All Tracks", "Baseline Tracks", "ML Region Tracks", "Rejected Tracks"]

        for category in categories:
            if category in organized_data and organized_data[category]["logits"] is not None and len(organized_data[category]["logits"]) > 0:
                print(f"\n{'=' * 20} Evaluating {category} {'=' * 20}")
                print(f"Total tracks: {len(organized_data[category]['predictions']):,}")

                # Run evaluation for this category
                result = self.evaluate_category(organized_data[category], category)

                all_results[category] = result

                # Print summary for this category
                print(f"  Efficiency: {result['efficiency']:.4f}")
                print(f"  Fake Rate: {result['fake_rate']:.4f}")
                if result["roc_auc"] is not None:
                    print(f"  ROC AUC: {result['roc_auc']:.4f}")
            else:
                print(f"\n{'=' * 20} Skipping {category} {'=' * 20}")
                print("No tracks found for this category")
                all_results[category] = {"num_tracks": 0, "efficiency": 0.0, "fake_rate": 0.0, "roc_auc": None}

        # Write comparative summary
        self.write_comparative_summary(all_results, baseline_stats, ml_region_stats)

        print("\n" + "=" * 80)
        print("TASK 2 EVALUATION COMPLETED")
        print("=" * 80)

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 2: Track Validity Classification")
    parser.add_argument(
        "--eval_path",
        type=str,
        #    default="/scratch/epoch=069-val_loss=2.87600_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600_eval.h5",
        default="/scratch/epoch=139-val_loss=2.74982_ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600_eval.h5",
        #    default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/TRK-ATLAS-Muon-smallModel-better-run_20250925-T202923/ckpts/epoch=017-val_loss=4.78361_ml_test_data_156000_hdf5_filtered_mild_cuts_eval.h5",
        help="Path to evaluation HDF5 file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        #    default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts",
        default="/scratch/ml_test_data_156000_hdf5_filtered_wp0990_maxtrk2_maxhit600",
        help="Path to processed test data directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./tracking_evaluation_results/task2_track_validity", help="Base output directory for plots and results"
    )
    parser.add_argument("--max_events", "-m", type=int, default=1000, help="Maximum number of events to process (for testing)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible sampling")

    args = parser.parse_args()

    print("Task 2: Track Validity Classification Evaluation with Categories")
    print("=" * 70)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    print(f"Random seed: {args.random_seed}")

    try:
        evaluator = Task2TrackValidityEvaluator(
            eval_path=args.eval_path, data_dir=args.data_dir, output_dir=args.output_dir, max_events=args.max_events, random_seed=args.random_seed
        )

        evaluator.run_evaluation_with_categories()

    except (ValueError, KeyError, RuntimeError) as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
