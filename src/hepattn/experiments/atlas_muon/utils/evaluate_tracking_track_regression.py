#!/usr/bin/env python3
# ruff: noqa: PLC0415,E501,DTZ005,EXE001
"""Evaluation script for Task 3: Regression Outputs (parameter_regression) with Categories.

This script evaluates the performance of the regression outputs by:
1. Creating truth-normalized residual plots for eta, phi, pt, and q
2. Creating correlation plots between predictions and truth
3. Analyzing the performance with three categories: all tracks, baseline tracks, rejected tracks
4. Applying baseline filtering criteria from Task 1

Based on lessons learned from Task 1 and Task 2 evaluation improvements.
"""

import sys

import h5py
import matplotlib as mpl
import numpy as np

mpl.use("Agg")  # Set backend before importing pyplot
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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


class Task3RegressionEvaluator:
    """Evaluator for regression outputs task with baseline filtering and categories."""

    def __init__(self, eval_path, data_dir, output_dir, max_events=None, random_seed=42):
        self.eval_path = eval_path
        self.data_dir = data_dir
        self.max_events = max_events
        self.random_seed = random_seed

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / f"task3_evaluation_{timestamp}"

        # Create output directory and subdirectories for categories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_tracks_dir = self.output_dir / "all_tracks"
        self.baseline_dir = self.output_dir / "baseline_tracks"
        self.ml_region_dir = self.output_dir / "ml_region_tracks"
        self.rejected_dir = self.output_dir / "rejected_tracks"

        for subdir in [self.all_tracks_dir, self.baseline_dir, self.ml_region_dir, self.rejected_dir]:
            subdir.mkdir(parents=True, exist_ok=True)

        print("Task 3 Evaluator initialized")
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
        all_data = {"phi": [], "eta": [], "pt": [], "q": [], "phi_truth": [], "eta_truth": [], "pt_truth": [], "q_truth": [], "track_info": []}
        baseline_data = {"phi": [], "eta": [], "pt": [], "q": [], "phi_truth": [], "eta_truth": [], "pt_truth": [], "q_truth": [], "track_info": []}
        ml_region_data = {"phi": [], "eta": [], "pt": [], "q": [], "phi_truth": [], "eta_truth": [], "pt_truth": [], "q_truth": [], "track_info": []}
        rejected_data = {"phi": [], "eta": [], "pt": [], "q": [], "phi_truth": [], "eta_truth": [], "pt_truth": [], "q_truth": [], "track_info": []}

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

                # Get regression predictions
                pred_phi = pred_group["preds/final/parameter_regression/track_truthMuon_phi"][...]  # Shape: (1, 2)
                pred_eta = pred_group["preds/final/parameter_regression/track_truthMuon_eta"][...]  # Shape: (1, 2)
                pred_pt = pred_group["preds/final/parameter_regression/track_truthMuon_pt"][...] * 200  # Shape: (1, 2)
                pred_q = pred_group["preds/final/parameter_regression/track_truthMuon_q"][...]  # Shape: (1, 2)

                # Truth track parameters (dataset still has batch dimension of 1)
                true_particle_valid = targets["particle_valid"][0]  # Remove batch dimension
                true_hit_assignments = targets["particle_hit_valid"][0]  # Remove batch dimension

                # Extract true particle parameters (only for valid particles)
                valid_particles = true_particle_valid.numpy()
                num_valid = int(valid_particles.sum())

                if num_valid == 0:
                    continue

                for track_idx in range(num_valid):
                    true_hits = true_hit_assignments[track_idx].numpy()
                    track_mask = true_hits.astype(bool)

                    # Get predicted parameters
                    pred_phi_val = float(pred_phi[0, track_idx])
                    pred_eta_val = float(pred_eta[0, track_idx])
                    pred_pt_val = float(pred_pt[0, track_idx])
                    pred_q_val = float(pred_q[0, track_idx])

                    # Get true track parameters (handle dataset structure with batch dimension)
                    eta_tensor = targets["particle_truthMuon_eta"][0, track_idx]  # Remove batch dimension
                    phi_tensor = targets["particle_truthMuon_phi"][0, track_idx]  # Remove batch dimension
                    pt_tensor = targets["particle_truthMuon_pt"][0, track_idx]  # Remove batch dimension
                    q_tensor = targets["particle_truthMuon_q"][0, track_idx]  # Remove batch dimension

                    # Handle potential multi-dimensional tensors by taking the first element if needed
                    true_eta = eta_tensor.item() if eta_tensor.numel() == 1 else eta_tensor[0].item()
                    true_phi = phi_tensor.item() if phi_tensor.numel() == 1 else phi_tensor[0].item()
                    true_pt = pt_tensor.item() if pt_tensor.numel() == 1 else pt_tensor[0].item()
                    true_q = q_tensor.item() if q_tensor.numel() == 1 else q_tensor[0].item()

                    track_info = {"pt": true_pt, "eta": true_eta, "phi": true_phi, "event_id": dataset_idx, "track_id": track_idx}

                    # Add to all tracks
                    all_data["phi"].append(pred_phi_val)
                    all_data["eta"].append(pred_eta_val)
                    all_data["pt"].append(pred_pt_val)
                    all_data["q"].append(pred_q_val)
                    all_data["phi_truth"].append(true_phi)
                    all_data["eta_truth"].append(true_eta)
                    all_data["pt_truth"].append(true_pt)
                    all_data["q_truth"].append(true_q)
                    all_data["track_info"].append(track_info)

                    # Apply baseline filtering
                    baseline_stats["total_tracks_checked"] += 1

                    # Apply ML region filtering (in parallel with baseline)
                    ml_region_stats["total_tracks_checked"] += 1

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

                    # Add to appropriate categories
                    if baseline_passed:
                        baseline_stats["tracks_passed_all_cuts"] += 1
                        baseline_data["phi"].append(pred_phi_val)
                        baseline_data["eta"].append(pred_eta_val)
                        baseline_data["pt"].append(pred_pt_val)
                        baseline_data["q"].append(pred_q_val)
                        baseline_data["phi_truth"].append(true_phi)
                        baseline_data["eta_truth"].append(true_eta)
                        baseline_data["pt_truth"].append(true_pt)
                        baseline_data["q_truth"].append(true_q)
                        baseline_data["track_info"].append(track_info)

                    if ml_region_passed:
                        ml_region_stats["tracks_passed_all_cuts"] += 1
                        ml_region_data["phi"].append(pred_phi_val)
                        ml_region_data["eta"].append(pred_eta_val)
                        ml_region_data["pt"].append(pred_pt_val)
                        ml_region_data["q"].append(pred_q_val)
                        ml_region_data["phi_truth"].append(true_phi)
                        ml_region_data["eta_truth"].append(true_eta)
                        ml_region_data["pt_truth"].append(true_pt)
                        ml_region_data["q_truth"].append(true_q)
                        ml_region_data["track_info"].append(track_info)

                    # Add to rejected if it doesn't pass ML region (keeping original logic for compatibility)
                    if not ml_region_passed:
                        rejected_data["phi"].append(pred_phi_val)
                        rejected_data["eta"].append(pred_eta_val)
                        rejected_data["pt"].append(pred_pt_val)
                        rejected_data["q"].append(pred_q_val)
                        rejected_data["phi_truth"].append(true_phi)
                        rejected_data["eta_truth"].append(true_eta)
                        rejected_data["pt_truth"].append(true_pt)
                        rejected_data["q_truth"].append(true_q)
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
        print(f"  All tracks: {len(all_data['phi'])}")
        print(f"  Baseline tracks: {len(baseline_data['phi'])}")
        print(f"  ML region tracks: {len(ml_region_data['phi'])}")
        print(f"  Rejected tracks: {len(rejected_data['phi'])}")

        return all_data, baseline_data, ml_region_data, rejected_data, baseline_stats, ml_region_stats

    def calculate_statistics(self, data, category_name):
        """Calculate regression statistics for a category."""
        statistics = {}

        params = ["phi", "eta", "pt", "q"]
        for param in params:
            if len(data[param]) > 0:
                predictions = data[param]
                truth = data[param + "_truth"]

                if param == "q":
                    # For charge, use classification approach with 0 cutoff
                    # Convert predictions to discrete classifications
                    pred_charge_discrete = np.where(predictions >= 0, 1, -1)

                    # Calculate accuracy
                    correct_predictions = (pred_charge_discrete == truth).sum()
                    total_predictions = len(truth)
                    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

                    # Store both accuracy and raw data for plotting
                    statistics[param] = {
                        "accuracy": accuracy,
                        "num_tracks": total_predictions,
                        "raw_predictions": predictions,  # Keep raw predictions for distribution plots
                        "discrete_predictions": pred_charge_discrete,
                        "truth": truth,
                        "correct_predictions": correct_predictions,
                    }
                    print(f"{category_name} - {param}: Accuracy = {accuracy:.4f} ({correct_predictions}/{total_predictions})")
                else:
                    # For other parameters, use regression approach with normalized residuals
                    # Raw residuals
                    residuals = predictions - truth
                    # Compute truth-normalized residuals using absolute truth to avoid sign flips
                    residuals = np.abs(truth)

                    if residuals.size > 0:
                        mean_residual = np.mean(residuals)
                        std_residual = np.std(residuals)
                        num_tracks = residuals.size
                    else:
                        mean_residual = np.nan
                        std_residual = np.nan
                        num_tracks = 0

                    statistics[param] = {
                        "mean_residual": mean_residual,
                        "std_residual": std_residual,
                        "num_tracks": num_tracks,
                        "residuals": residuals,
                    }
                    print(f"{category_name} - {param}: Mean residuals = {mean_residual:.6f}, STD residuals = {std_residual:.6f}")
            elif param == "q":
                statistics[param] = {
                    "accuracy": 0.0,
                    "num_tracks": 0,
                    "raw_predictions": np.array([]),
                    "discrete_predictions": np.array([]),
                    "truth": np.array([]),
                    "correct_predictions": 0,
                }
            else:
                statistics[param] = {"mean_residual": np.nan, "std_residual": np.nan, "num_tracks": 0, "residuals": np.array([])}

        return statistics

    def _calculate_data_driven_bins(self, data, variable, num_bins=21):
        """Calculate data-driven bins for a kinematic variable."""
        # Extract variable values from track info
        if len(data["track_info"]) == 0:
            # Fallback to fixed bins if no data
            if variable == "pt":
                return np.linspace(0, 200, num_bins)
            if variable == "eta":
                return np.linspace(-3, 3, num_bins)
            if variable == "phi":
                return np.linspace(-np.pi, np.pi, num_bins)

        var_values = np.array([track[variable] for track in data["track_info"]])

        if variable == "pt":
            # PT: min of data to 200
            min_val = np.min(var_values)
            max_val = 200.0
            return np.linspace(min_val, max_val, num_bins)
        if variable == "eta":
            # Eta: min to max of data
            min_val = np.min(var_values)
            max_val = np.max(var_values)
            return np.linspace(min_val, max_val, num_bins)
        if variable == "phi":
            # Phi: keep standard range
            return np.linspace(-np.pi, np.pi, num_bins)

        return np.linspace(0, 1, num_bins)  # Fallback

    def calculate_charge_accuracy_by_variable(self, data, variable="pt", bins=None):
        """Calculate charge accuracy binned by a kinematic variable."""
        if bins is None:
            bins = self._calculate_data_driven_bins(data, variable)

        # Extract the variable values from track_info
        var_values = np.array([track[variable] for track in data["track_info"]])

        # Get charge predictions and truth
        predictions = data["q"]
        truth = data["q_truth"]

        # Convert predictions to discrete classifications
        pred_charge_discrete = np.where(predictions >= 0, 1, -1)

        # Calculate bin indices
        bin_indices = np.digitize(var_values, bins) - 1

        bin_accuracies = []
        bin_centers = []
        acc_errors = []

        for i in range(len(bins) - 1):
            mask = bin_indices == i

            if mask.sum() == 0:
                continue

            # Get predictions and truth for this bin
            bin_pred = pred_charge_discrete[mask]
            bin_truth = truth[mask]

            # Calculate accuracy for this bin
            correct = (bin_pred == bin_truth).sum()
            total = len(bin_truth)

            if total > 0:
                accuracy = correct / total
                # Binomial error
                acc_error = np.sqrt(accuracy * (1 - accuracy) / total)
            else:
                accuracy = 0
                acc_error = 0

            bin_accuracies.append(accuracy)
            acc_errors.append(acc_error)
            bin_centers.append((bins[i] + bins[i + 1]) / 2)

        return np.array(bin_centers), np.array(bin_accuracies), np.array(acc_errors)

    def plot_charge_accuracy_vs_variable(self, data, variable="pt", output_subdir=None, category_name=""):
        """Plot charge classification accuracy vs a kinematic variable."""
        if len(data["q"]) == 0:
            print(f"Warning: No data for {variable} charge accuracy plot in {category_name}")
            return

        # Use data-driven bins
        bins = self._calculate_data_driven_bins(data, variable)

        bin_centers, accuracies, acc_errors = self.calculate_charge_accuracy_by_variable(data, variable, bins)

        if len(bin_centers) == 0:
            print(f"Warning: No data points for {variable} charge accuracy plot")
            return

        # Create the plot with step style and error bands
        _fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Accuracy plot with step style and error bands
        for i, (lhs, rhs, acc_val, acc_err) in enumerate(zip(bins[:-1], bins[1:], accuracies, acc_errors, strict=False)):
            color = "green"

            # Create error band
            if acc_err > 0:
                point_in_range = np.linspace(lhs, rhs, 100)
                y_upper = min(acc_val + acc_err, 1.0)  # Cap at 1.0
                y_lower = max(acc_val - acc_err, 0.0)  # Floor at 0.0
                ax.fill_between(point_in_range, y_upper, y_lower, color=color, alpha=0.3, label="binomial err - Accuracy" if i == 0 else "")

            # Step plot
            ax.step([lhs, rhs], [acc_val, acc_val], color=color, linewidth=2.5, label="Charge Classification Accuracy" if i == 0 else "")

        ax.set_ylabel("Charge Classification Accuracy")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Charge Classification Accuracy vs {variable.capitalize()} - {category_name}\n(Cutoff: pred >= 0 → +1, pred < 0 → -1)")

        # Set x-axis labels
        if variable == "pt":
            ax.set_xlabel("$p_T$ [GeV]")
        elif variable == "eta":
            ax.set_xlabel("$\\eta$")
        elif variable == "phi":
            ax.set_xlabel("$\\phi$ [rad]")

        plt.tight_layout()

        # Save the plot
        output_path = output_subdir / f"charge_accuracy_vs_{variable}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved {category_name} charge accuracy vs {variable} plot to {output_path}")

    def plot_residuals(self, data, param, output_dir, category_name, statistics):
        """Plot residual distributions using step histogram style."""
        if len(data[param]) == 0:
            print(f"Warning: No data for {param} residuals in {category_name}")
            return

        if param == "q":
            # For charge, we don't plot residuals since it's now classification
            # Instead, we could plot a confusion matrix or accuracy metrics
            print("Skipping residual plot for charge parameter (classification mode)")
            return

        # Get both normalized and absolute residuals
        residuals = statistics[param]["residuals"]
        if residuals.size == 0:
            print(f"Warning: No finite normalized residuals for {param} in {category_name}")
            return

        # Calculate absolute residuals (pred - truth)
        predictions = np.array(data[param])
        truth = np.array(data[f"{param}_truth"])
        absolute_residuals = predictions - truth

        # Plot 1: Normalized Residuals
        # norm_mean = np.mean(residuals / np.abs(truth))
        # norm_std = np.std(residuals / np.abs(truth))
        # self._plot_single_residual_type(residuals, param, output_dir, category_name,
        #                               "normalized", norm_mean, norm_std)

        # Plot 2: Absolute Residuals
        abs_mean = np.mean(absolute_residuals)
        abs_std = np.std(absolute_residuals)
        self._plot_single_residual_type(absolute_residuals, param, output_dir, category_name, "absolute", abs_mean, abs_std)

    def _plot_single_residual_type(self, residuals, param, output_dir, category_name, residual_type, mean_val, std_val):
        """Plot a single type of residual (normalized or absolute)."""
        # Create step histogram plot
        plt.figure(figsize=(10, 6))

        # Calculate mean and std for 3-sigma range
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        # Define range as mean ± 3*sigma
        lower_bound = mean_residual - 3 * std_residual
        upper_bound = mean_residual + 3 * std_residual

        # Create bins centered on mean, extending 3 sigma to each side
        n_bins = 50
        bins = np.linspace(lower_bound, upper_bound, n_bins + 1)

        # Clip outliers into edge bins
        clipped_residuals = np.clip(residuals, lower_bound, upper_bound)

        # Calculate histogram with clipped data
        counts, bin_edges = np.histogram(clipped_residuals, bins=bins, density=False)

        # Create step histogram that properly displays the last bin
        # Use bin centers for plotting to ensure all bins are visible
        (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_edges[1] - bin_edges[0]

        # Create step histogram extending the last point to show the final bin
        x_step = np.concatenate([bin_edges[:-1], [bin_edges[-1]]])
        y_step = np.concatenate([counts, [counts[-1] if len(counts) > 0 else 0]])

        plt.step(x_step, y_step, where="post", linewidth=1, color="blue", label=f"{param.capitalize()} {residual_type.capitalize()} Residuals")

        # Add vertical lines for statistics
        plt.axvline(mean_residual, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_residual:.6f}")
        plt.axvline(mean_residual - std_residual, color="orange", linestyle=":", alpha=0.7, label=f"+-1sigma: {std_residual:.6f}")
        plt.axvline(mean_residual + std_residual, color="orange", linestyle=":", alpha=0.7)
        plt.axvline(mean_residual - 2 * std_residual, color="yellow", linestyle=":", alpha=0.7, label="+-2sigma")
        plt.axvline(mean_residual + 2 * std_residual, color="yellow", linestyle=":", alpha=0.7)
        # plt.axvline(lower_bound, color='purple', linestyle='-', alpha=0.7,
        #            label=f'+-3sigma (plot range)')
        # plt.axvline(upper_bound, color='purple', linestyle='-', alpha=0.7)
        plt.axvline(0, color="black", linestyle="-", alpha=0.7, linewidth=1, label="Perfect agreement")

        # Formatting
        if residual_type == "normalized":
            xlabel = f"{param.capitalize()} Normalized Residual (Pred - Truth)/|Truth|"
            title_suffix = "Truth-Normalized"
        else:
            xlabel = f"{param.capitalize()} Absolute Residual (Pred - Truth)"
            title_suffix = "Absolute"

        plt.xlabel(xlabel)
        plt.title(
            f"{param.capitalize()} {title_suffix} Residual Distribution - {category_name}\n(Centered on mean +-3sigma, outliers clipped to edge bins)"
        )
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add text box with statistics
        n_total = len(residuals)
        n_clipped = np.sum((residuals < lower_bound) | (residuals > upper_bound))
        clipped_pct = 100 * n_clipped / n_total if n_total > 0 else 0

        stats_text = f"Mean: {mean_residual:.6f}\nSTD: {std_residual:.6f}\nN: {n_total}\nClipped: {n_clipped} ({clipped_pct:.1f}%)"
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()

        # Save plot
        output_path = output_dir / f"{param}_{residual_type}_residuals.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved {category_name} {param} {residual_type} residual plot to {output_path} (mean +-3sigma range, {n_clipped} outliers clipped)")

    def plot_overlaid_distributions(self, data, param, output_dir, category_name):
        """Plot overlaid distribution plots comparing model predictions with ground truth."""
        if len(data[param]) == 0:
            print(f"Warning: No data for {param} distribution plots in {category_name}")
            return

        plt.figure(figsize=(10, 6))

        predictions = data[param]
        truth = data[param + "_truth"]

        # Calculate appropriate bins with more resolution
        all_values = np.concatenate([predictions, truth])

        # Set axis ranges to match filter evaluation
        if param == "eta":
            # Use data-driven bins for eta (min to max of actual data)
            min_val = np.min(all_values)
            max_val = np.max(all_values)
            bins = np.linspace(min_val, max_val, 100)  # More bins for better resolution
            plt.xlim(min_val, max_val)
        elif param == "phi":
            bins = np.linspace(-np.pi, np.pi, 100)
            plt.xlim(-np.pi, np.pi)
        elif param == "pt":
            # For pt, use data min to 200
            min_val = np.min(all_values)
            max_val = min(200.0, np.max(all_values) * 1.1)  # Cap at 200 or slightly above data max
            bins = np.linspace(min_val, max_val, 100)
        elif param == "q":
            # For charge, use way more bins for better resolution of logit distributions
            bins = np.linspace(-1.5, 1.5, 301)  # 300 bins for high resolution
            plt.xlim(-1.5, 1.5)
        else:
            bins = np.linspace(np.percentile(all_values, 1), np.percentile(all_values, 99), 100)

        # Plot both distributions with transparency (alpha) to make them properly overlaid
        plt.hist(truth, bins=bins, alpha=0.6, density=False, label="Ground Truth", color="blue", histtype="stepfilled")
        plt.hist(predictions, bins=bins, alpha=0.6, density=False, label="Model Predictions", color="red", histtype="stepfilled")

        plt.xlabel(f"{param.capitalize()}")
        plt.ylabel("Count")
        plt.title(f"{param.capitalize()} Distribution: Predictions vs Ground Truth - {category_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_path = output_dir / f"{param}_distribution_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved {category_name} {param} distribution comparison to {output_path}")

    def write_category_summary(self, data, statistics, output_dir, category_name):
        """Write evaluation summary for a category."""
        summary_path = output_dir / f"{category_name.lower().replace(' ', '_')}_summary.txt"

        with summary_path.open("w") as f:
            f.write(f"TASK 3: REGRESSION EVALUATION - {category_name.upper()} SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Category: {category_name}\n\n")

            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total tracks analyzed: {len(data['phi']):,}\n\n")

            f.write("REGRESSION AND CLASSIFICATION STATISTICS\n")
            f.write("-" * 40 + "\n")
            for param in ["phi", "eta", "pt", "q"]:
                stats = statistics.get(param, {})
                if stats.get("num_tracks", 0) > 0:
                    f.write(f"{param.upper()}:\n")
                    if param == "q":
                        f.write(f"  Classification accuracy: {stats['accuracy']:.4f}\n")
                        f.write(f"  Correct predictions: {stats['correct_predictions']}/{stats['num_tracks']}\n")
                    else:
                        f.write(f"  Mean normalized residual: {stats['mean_residual']:.6f}\n")
                        f.write(f"  STD normalized residual:  {stats['std_residual']:.6f}\n")
                    f.write(f"  Number of tracks: {stats['num_tracks']}\n\n")
                else:
                    f.write(f"{param.upper()}: No data available\n\n")

            f.write(f"\nGenerated at: {datetime.now()}\n")

        print(f"Summary for {category_name} written to {summary_path}")

    def calculate_overall_charge_accuracy(self, data):
        """Calculate overall charge accuracy across all tracks."""
        if len(data["q"]) == 0:
            return 0.0

        predictions = data["q"]
        truth = data["q_truth"]

        # Convert predictions to discrete classifications
        pred_charge_discrete = np.where(predictions >= 0, 1, -1)

        # Calculate accuracy
        correct_predictions = (pred_charge_discrete == truth).sum()
        total_predictions = len(truth)
        return correct_predictions / total_predictions if total_predictions > 0 else 0

    def write_comparative_summary(self, all_results, baseline_stats, ml_region_stats):
        """Write comprehensive summary comparing all categories."""
        summary_path = self.output_dir / "task3_comparative_summary.txt"

        with summary_path.open("w") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TASK 3: REGRESSION EVALUATION - COMPARATIVE SUMMARY\n")
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
                    num_tracks = all_results[category]["phi"]["num_tracks"] if "phi" in all_results[category] else 0
                    f.write(f"{category}: {num_tracks:,} tracks\n")
                else:
                    f.write(f"{category}: 0 tracks\n")
            f.write("\n")

            # Write comparative metrics for each parameter
            params = ["phi", "eta", "pt", "q"]

            for param in params:
                f.write(f"{param.upper()} COMPARATIVE METRICS\n")
                f.write("-" * 30 + "\n")

                if param == "q":
                    f.write(f"{'Category':<20}{'Num Tracks':<15}{'Accuracy':<15}\n")
                    f.write("-" * 50 + "\n")

                    for category in categories:
                        if category in all_results and param in all_results[category]:
                            stats = all_results[category][param]
                            f.write(f"{category:<20}{stats['num_tracks']:<15}{stats['accuracy']:<15.4f}\n")
                        else:
                            f.write(f"{category:<20}{'0':<15}{'N/A':<15}\n")
                else:
                    f.write(f"{'Category':<20}{'Num Tracks':<15}{'Mean Residual':<15}{'STD Residual':<15}\n")
                    f.write("-" * 65 + "\n")

                    for category in categories:
                        if category in all_results and param in all_results[category]:
                            stats = all_results[category][param]
                            f.write(f"{category:<20}{stats['num_tracks']:<15}{stats['mean_residual']:<15.6f}{stats['std_residual']:<15.6f}\n")
                        else:
                            f.write(f"{category:<20}{'0':<15}{'N/A':<15}{'N/A':<15}\n")
                f.write("\n")

        print(f"Comparative summary written to {summary_path}")

    def run_evaluation_with_categories(self):
        """Run evaluation for all categories."""
        print("=" * 80)
        print("TASK 3: REGRESSION EVALUATION WITH CATEGORIES")
        print("=" * 80)

        # Setup and collect data
        self.setup_data_module()
        all_data, baseline_data, ml_region_data, rejected_data, baseline_stats, ml_region_stats = self.collect_and_process_data()

        # Store results for comparative summary
        all_results = {}

        # Process each category
        categories = [
            ("All Tracks", all_data, self.all_tracks_dir),
            ("Baseline Tracks", baseline_data, self.baseline_dir),
            ("ML Region Tracks", ml_region_data, self.ml_region_dir),
            ("Rejected Tracks", rejected_data, self.rejected_dir),
        ]

        for category_name, data, output_dir in categories:
            print("\n" + "=" * 50)
            print(f"EVALUATING {category_name.upper()}")
            print("=" * 50)

            if len(data["phi"]) == 0:
                print(f"Warning: No data for {category_name}")
                continue

            # Calculate statistics
            statistics = self.calculate_statistics(data, category_name)
            all_results[category_name] = statistics

            # Generate plots
            print("Generating plots...")

            # Residual plots for each parameter (skip charge since it's now classification)
            for param in ["phi", "eta", "pt"]:
                try:
                    self.plot_residuals(data, param, output_dir, category_name, statistics)
                except (ValueError, KeyError, RuntimeError) as e:
                    print(f"Error creating {param} residual plots: {e}")

            # Charge accuracy plots by kinematic variables
            for variable in ["pt", "eta", "phi"]:
                try:
                    self.plot_charge_accuracy_vs_variable(data, variable, output_dir, category_name)
                except (ValueError, KeyError, RuntimeError) as e:
                    print(f"Error creating charge accuracy vs {variable} plots: {e}")

            # Overlaid distribution plots
            for param in ["phi", "eta", "pt", "q"]:
                try:
                    self.plot_overlaid_distributions(data, param, output_dir, category_name)
                except (ValueError, KeyError, RuntimeError) as e:
                    print(f"Error creating {param} distribution plots: {e}")

            # Write individual summary
            try:
                self.write_category_summary(data, statistics, output_dir, category_name)
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"Error writing summary for {category_name}: {e}")

        # Write comparative summary
        self.write_comparative_summary(all_results, baseline_stats, ml_region_stats)

        print(f"\nTask 3 evaluation with categories complete. Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 3: Regression Outputs with Categories")
    parser.add_argument(
        "--eval_path",
        type=str,
        default="/path/to/evaluation_file.h5",
        help="Path to evaluation HDF5 file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        #    default="/home/iwsatlas1/jrenusch/master_thesis/tracking/data/tracking_eval/ml_test_data_156000_hdf5_filtered_mild_cuts",
        default="/path/to/processed_test_data_directory",
        help="Path to processed test data directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./tracking_evaluation_results/task3_regression", help="Base output directory for plots and results"
    )
    parser.add_argument("--max_events", "-m", type=int, default=1000, help="Maximum number of events to process (for testing)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible random sampling")

    args = parser.parse_args()

    print("Task 3: Regression Evaluation with Categories")
    print("=" * 50)
    print(f"Evaluation file: {args.eval_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max events: {args.max_events}")
    print(f"Random seed: {args.random_seed}")

    try:
        evaluator = Task3RegressionEvaluator(
            eval_path=args.eval_path, data_dir=args.data_dir, output_dir=args.output_dir, max_events=args.max_events, random_seed=args.random_seed
        )

        evaluator.run_evaluation_with_categories()

    except (ValueError, KeyError, RuntimeError) as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
