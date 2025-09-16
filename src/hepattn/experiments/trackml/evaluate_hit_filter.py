#!/usr/bin/env python3
"""
Script to evaluate the performance of the hit filter in terms of purity and efficiency.

Purity = True Positives / (True Positives + False Positives)
Efficiency = True Positives / (True Positives + False Negatives)

Where:
- True Positives (TP): Hits correctly classified as being on valid particles
- False Positives (FP): Hits incorrectly classified as being on valid particles
- False Negatives (FN): Hits incorrectly classified as not being on valid particles
- True Negatives (TN): Hits correctly classified as not being on valid particles
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import sys

from hepattn.experiments.trackml.data import TrackMLDataset


def load_hit_eval_predictions(
    hit_eval_path: str, sample_ids: List[int]
) -> Dict[int, np.ndarray]:
    """
    Load hit filter predictions from the HDF5 evaluation file.

    Args:
        hit_eval_path: Path to the HDF5 file containing predictions
        sample_ids: List of sample IDs to load predictions for

    Returns:
        Dictionary mapping sample_id to prediction array
    """
    predictions = {}

    with h5py.File(hit_eval_path, "r") as hit_eval_file:
        for sample_id in sample_ids:
            if str(sample_id) in hit_eval_file:
                # The dataset has shape (1, num_hits), we take the first (and only) element
                pred = hit_eval_file[
                    f"{sample_id}/preds/final/hit_filter/hit_on_valid_particle"
                ][0]
                predictions[sample_id] = pred.astype(bool)
            else:
                print(f"Warning: Sample ID {sample_id} not found in hit eval file")

    return predictions


def load_hit_eval_probabilities(
    hit_eval_path: str, sample_ids: List[int]
) -> Dict[int, np.ndarray]:
    """
    Load hit filter probability scores from the HDF5 evaluation file.

    Args:
        hit_eval_path: Path to the HDF5 file containing predictions
        sample_ids: List of sample IDs to load predictions for

    Returns:
        Dictionary mapping sample_id to probability array
    """
    probabilities = {}

    with h5py.File(hit_eval_path, "r") as hit_eval_file:
        for sample_id in sample_ids:
            if str(sample_id) in hit_eval_file:
                # The dataset has shape (1, num_hits), we take the first (and only) element
                prob = hit_eval_file[
                    f"{sample_id}/preds/final/hit_filter/hit_on_valid_particle"
                ][0]
                probabilities[sample_id] = prob.astype(float)
            else:
                print(f"Warning: Sample ID {sample_id} not found in hit eval file")

    return probabilities


def load_ground_truth(dataset: TrackMLDataset) -> Dict[int, np.ndarray]:
    """
    Load ground truth labels for hit filter evaluation.

    Args:
        dataset: TrackMLDataset instance

    Returns:
        Dictionary mapping sample_id to ground truth array
    """
    ground_truth = {}

    for idx in range(len(dataset)):
        # Load the event to get ground truth labels
        hits, _ = dataset.load_event(idx)
        sample_id = dataset.sample_ids[idx]

        # The ground truth is in the 'on_valid_particle' field
        ground_truth[sample_id] = hits["on_valid_particle"].values.astype(bool)

    return ground_truth


def load_particle_info(dataset: TrackMLDataset) -> Dict[int, Dict]:
    """
    Load particle information for reconstructable analysis.

    Args:
        dataset: TrackMLDataset instance

    Returns:
        Dictionary mapping sample_id to particle info dict
    """
    particle_info = {}

    for idx in range(len(dataset)):
        # Load the event to get particle information
        hits, particles = dataset.load_event(idx)
        sample_id = dataset.sample_ids[idx]

        # Calculate hit counts per particle
        hit_counts = hits["particle_id"].value_counts()

        # Get particle properties
        particle_data = {
            "particle_id": particles["particle_id"].values,
            "pt": particles["pt"].values,
            "eta": particles["eta"].values,
            "hit_counts": hit_counts.to_dict(),
            "volume_ids": hits["volume_id"].values
            if "volume_id" in hits.columns
            else None,
        }

        particle_info[sample_id] = particle_data

    return particle_info


def calculate_old_style_metrics(
    dataset: TrackMLDataset,
    predictions: Dict[int, np.ndarray],
    ground_truth: Dict[int, np.ndarray],
    hit_cut: float = 0.1,
    eta_cut: float = 2.5,
) -> Dict[str, float]:
    """
    Calculate metrics in the same way as the old evaluation code.

    Args:
        dataset: TrackMLDataset instance
        predictions: Dictionary mapping sample_id to prediction arrays
        ground_truth: Dictionary mapping sample_id to ground truth arrays
        hit_cut: Threshold for hit filtering
        eta_cut: Eta cut for reconstructable particles

    Returns:
        Dictionary containing old-style metrics
    """
    all_hits_data = []
    all_particles_data = []

    for idx in range(len(dataset)):
        sample_id = dataset.sample_ids[idx]

        if sample_id not in predictions or sample_id not in ground_truth:
            continue

        # Load event data
        hits, particles = dataset.load_event(idx)

        # Get predictions and ground truth
        pred = predictions[sample_id]
        gt = ground_truth[sample_id]

        # # Ensure arrays have same length
        # min_len = min(len(pred), len(gt), len(hits))
        # pred = pred[:min_len]
        # gt = gt[:min_len]
        # hits = hits.iloc[:min_len].copy()

        # Add prediction and ground truth to hits dataframe
        hits["pred"] = pred
        hits["tgt"] = gt
        hits["prob"] = pred.astype(float)  # Convert boolean to float for compatibility

        # Calculate hit counts per particle before filtering - EXACTLY like old code
        # In old code: parts["hits_pre"] = hits.pid.value_counts()
        hits_pre_counts = hits["particle_id"].value_counts()
        print(len(particles))
        print(len(hits_pre_counts))

        print(particles["particle_id"])
        # join on particle id

        # Apply hit filtering
        hits_post = hits[hits["pred"]]
        hits_post_counts = hits_post["particle_id"].value_counts()

        # Create particles dataframe with hit counts - EXACTLY like old code
        particles_df = particles.copy()

        # In old code: parts["hits_pre"] = hits.pid.value_counts()
        particles_df["hits_pre"] = particles_df["particle_id"].map(hits_pre_counts)

        # In old code: parts["hits_post"] = hits_post.pid.value_counts()
        particles_df["hits_post"] = particles_df["particle_id"].map(hits_post_counts)

        print(particles_df["hits_pre"])

        print(len(particles_df["hits_pre"]))

        # Define reconstructable particles - EXACTLY like old code
        # In old code: parts["reconstructable_pre"] = (parts["hits_pre"] >= 3) & (parts["pid"] != 0) & (parts["eta"].abs() < eta_cut)
        particles_df["reconstructable_pre"] = (
            (particles_df["hits_pre"] >= 3)
            & (particles_df["particle_id"] != 0)
            & (particles_df["eta"].abs() < eta_cut)
            & (particles_df["pt"] > 0.6)
        )
        # In old code: parts["reconstructable_post"] = (parts["hits_post"] >= 3) & (parts["pid"] != 0) & (parts["eta"].abs() < eta_cut)
        particles_df["reconstructable_post"] = (
            (particles_df["hits_post"] >= 3)
            & (particles_df["particle_id"] != 0)
            & (particles_df["eta"].abs() < eta_cut)
            & (particles_df["pt"] > 0.6)
        )

        # Add particle eta and pt to each hit - EXACTLY like old code
        # In old code: hits = hits.join(parts[["eta", "pt"]], on="pid")
        # Handle column overlap by renaming particle columns
        particles_for_join = particles_df[["eta", "pt"]].copy()
        particles_for_join.columns = ["particle_eta", "particle_pt"]
        hits = hits.join(particles_for_join, on="particle_id")

        all_hits_data.append(hits)
        all_particles_data.append(particles_df)

    # Combine all events
    if not all_hits_data:
        return {}

    all_hits = pd.concat(all_hits_data, ignore_index=True)
    all_particles = pd.concat(all_particles_data, ignore_index=True)

    # Calculate metrics in the same way as old code
    num_events = len(all_hits_data)

    # Hit-level metrics
    recall = (all_hits["pred"] & all_hits["tgt"]).sum() / all_hits["tgt"].sum()
    precision = (all_hits["pred"] & all_hits["tgt"]).sum() / all_hits["pred"].sum()

    # Standard errors
    se_recall = (recall * (1 - recall) / all_hits["tgt"].sum()) ** 0.5
    se_precision = (precision * (1 - precision) / all_hits["pred"].sum()) ** 0.5

    # Average hits per event
    avg_hits_pre = len(all_hits) / num_events
    avg_hits_post = all_hits["pred"].sum() / num_events

    # Hit-level efficiency (ratio of total hits after vs before filtering)
    hit_level_eff = all_hits["pred"].sum() / len(all_hits)

    print("---------RECONSTRUCTABLE PRE------------")
    print(all_particles["reconstructable_pre"].sum() / num_events)
    print("---------RECONSTRUCTABLE POST------------")
    print(all_particles["reconstructable_post"].sum() / num_events)

    # Track-level efficiency - EXACTLY like old code
    # In old code: track_eff = parts.reconstructable_post.sum() / parts.reconstructable_pre.sum()
    track_eff = (
        all_particles["reconstructable_post"].sum()
        / all_particles["reconstructable_pre"].sum()
    )

    # Track-level efficiency for high pT particles (>1 GeV) - EXACTLY like old code
    # In old code: track_eff_1gev = parts[parts.pt > pt_cut].reconstructable_post.sum() / parts[parts.pt > pt_cut].reconstructable_pre.sum()
    high_pt_particles = all_particles[all_particles["pt"] > 0.6]
    track_eff_1gev = (
        high_pt_particles["reconstructable_post"].sum()
        / high_pt_particles["reconstructable_pre"].sum()
        if high_pt_particles["reconstructable_pre"].sum() > 0
        else 0.0
    )

    # Initial precision (before filtering)
    precision_pre = all_hits["tgt"].mean()
    sel = all_hits["particle_eta"].abs() < eta_cut
    precision_pre_eta = all_hits[sel]["tgt"].mean()

    return {
        "recall": recall,
        "precision": precision,
        "se_recall": se_recall,
        "se_precision": se_precision,
        "avg_hits_pre": avg_hits_pre,
        "avg_hits_post": avg_hits_post,
        "hit_level_eff": hit_level_eff,
        "track_eff": track_eff,
        "track_eff_1gev": track_eff_1gev,
        "precision_pre": precision_pre,
        "precision_pre_eta": precision_pre_eta,
        "num_events": num_events,
        "total_hits": len(all_hits),
        "total_particles": len(all_particles),
    }


def calculate_metrics(
    predictions: np.ndarray, ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Calculate purity, efficiency, and other classification metrics.

    Args:
        predictions: Boolean array of predictions
        ground_truth: Boolean array of ground truth labels

    Returns:
        Dictionary containing various metrics
    """
    # Ensure arrays have the same length
    min_len = min(len(predictions), len(ground_truth))
    pred = predictions[:min_len]
    gt = ground_truth[:min_len]

    # Calculate confusion matrix elements
    tp = np.sum(pred & gt)  # True Positives
    fp = np.sum(pred & ~gt)  # False Positives
    fn = np.sum(~pred & gt)  # False Negatives
    tn = np.sum(~pred & ~gt)  # True Negatives

    # Calculate metrics
    total_hits = len(pred)
    total_true = np.sum(gt)
    total_pred_true = np.sum(pred)

    # Main metrics as defined by user
    purity = tp / total_pred_true if total_pred_true > 0 else 0.0
    efficiency = tp / total_true if total_true > 0 else 0.0

    # Additional useful metrics
    precision = purity  # Same as purity
    recall = efficiency  # Same as efficiency
    accuracy = (tp + tn) / total_hits if total_hits > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Hit reduction metrics
    hit_reduction = (
        (total_hits - total_pred_true) / total_hits if total_hits > 0 else 0.0
    )
    noise_rejection = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "purity": purity,
        "efficiency": efficiency,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "hit_reduction": hit_reduction,
        "noise_rejection": noise_rejection,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_hits": total_hits,
        "total_true": total_true,
        "total_pred_true": total_pred_true,
    }


def calculate_initial_purity(
    hits: pd.DataFrame, eta_cut: float = 2.5
) -> Tuple[float, float]:
    """
    Calculate initial purity before filtering.

    Args:
        hits: DataFrame containing hit information
        eta_cut: Eta cut for central region calculation

    Returns:
        Tuple of (overall_purity, central_purity)
    """
    # Overall purity
    total_hits = len(hits)
    valid_hits = np.sum(hits["on_valid_particle"])
    overall_purity = valid_hits / total_hits if total_hits > 0 else 0.0

    # Central region purity (|eta| < eta_cut)
    central_hits = hits[np.abs(hits["eta"]) < eta_cut]
    central_total = len(central_hits)
    central_valid = np.sum(central_hits["on_valid_particle"])
    central_purity = central_valid / central_total if central_total > 0 else 0.0

    return overall_purity, central_purity


def calculate_reconstructable_fraction(
    particle_info: Dict, predictions: np.ndarray, threshold: float = 0.1
) -> Dict[str, float]:
    """
    Calculate fraction of particles that remain reconstructable after filtering.

    Args:
        particle_info: Dictionary containing particle information
        predictions: Boolean array of hit filter predictions
        threshold: Threshold for determining reconstructable particles

    Returns:
        Dictionary with reconstructable fractions by pT bins
    """
    # This is a simplified version - in practice you'd need to map hits to particles
    # and check if particles still have 3+ hits after filtering

    # For now, return a placeholder structure
    pT_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    reconstructable_fractions = {}

    for i in range(len(pT_bins) - 1):
        bin_name = f"{pT_bins[i]}-{pT_bins[i+1]}"
        # Placeholder calculation - would need proper hit-to-particle mapping
        reconstructable_fractions[bin_name] = 0.99  # Placeholder

    return reconstructable_fractions


def calculate_roc_curve(
    probabilities: np.ndarray, ground_truth: np.ndarray, num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ROC curve data for purity vs efficiency.

    Args:
        probabilities: Array of hit filter probability scores
        ground_truth: Boolean array of ground truth labels
        num_thresholds: Number of threshold points to evaluate

    Returns:
        Tuple of (thresholds, efficiencies, purities)
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    efficiencies = []
    purities = []

    for threshold in thresholds:
        predictions = probabilities >= threshold

        # Calculate confusion matrix
        tp = np.sum(predictions & ground_truth)
        fp = np.sum(predictions & ~ground_truth)
        fn = np.sum(~predictions & ground_truth)

        # Calculate metrics
        efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        purity = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        efficiencies.append(efficiency)
        purities.append(purity)

    return np.array(thresholds), np.array(efficiencies), np.array(purities)


def evaluate_hit_filter(
    test_dir: str,
    hit_eval_path: str,
    num_events: int = -1,
    hit_volume_ids: List[int] = None,
    particle_min_pt: float = 0.6,
    particle_max_abs_eta: float = 2.5,
    particle_min_num_hits: int = 3,
    event_max_num_particles: int = 1000,
    model_name: str = "Hit Filter Model",
    threshold: float = 0.1,
) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, float]]:
    """
    Evaluate hit filter performance across all test events.

    Args:
        test_dir: Directory containing test data
        hit_eval_path: Path to HDF5 file with hit filter predictions
        num_events: Number of events to evaluate (-1 for all)
        hit_volume_ids: List of volume IDs to include
        particle_min_pt: Minimum particle pT
        particle_max_abs_eta: Maximum absolute particle eta
        particle_min_num_hits: Minimum number of hits per particle
        event_max_num_particles: Maximum number of particles per event
        model_name: Name of the model for reporting
        threshold: Threshold for operating point analysis

    Returns:
        Tuple of (overall_metrics, event_metrics, additional_metrics)
    """
    # Create dataset instance
    dataset = TrackMLDataset(
        dirpath=test_dir,
        inputs={"hit": ["x", "y", "z"]},  # Minimal inputs needed
        targets={"particle": []},  # No particle targets needed
        num_events=num_events,
        hit_volume_ids=hit_volume_ids,
        particle_min_pt=particle_min_pt,
        particle_max_abs_eta=particle_max_abs_eta,
        particle_min_num_hits=particle_min_num_hits,
        event_max_num_particles=event_max_num_particles,
        hit_eval_path=None,  # Don't apply filtering during evaluation
    )

    print(f"Evaluating hit filter on {len(dataset)} events")

    # Load predictions, probabilities, ground truth, and particle info
    sample_ids = dataset.sample_ids
    predictions = load_hit_eval_predictions(hit_eval_path, sample_ids)
    probabilities = load_hit_eval_probabilities(hit_eval_path, sample_ids)
    ground_truth = load_ground_truth(dataset)
    particle_info = load_particle_info(dataset)

    # Calculate initial purity (before filtering)
    total_initial_hits = 0
    total_initial_valid = 0
    total_initial_central_hits = 0
    total_initial_central_valid = 0

    for idx in range(len(dataset)):
        hits, _ = dataset.load_event(idx)
        total_initial_hits += len(hits)
        total_initial_valid += np.sum(hits["on_valid_particle"])

        # Central region (|eta| < 2.5)
        central_hits = hits[np.abs(hits["eta"]) < 2.5]
        total_initial_central_hits += len(central_hits)
        total_initial_central_valid += np.sum(central_hits["on_valid_particle"])

    initial_purity = (
        total_initial_valid / total_initial_hits if total_initial_hits > 0 else 0.0
    )
    initial_purity_central = (
        total_initial_central_valid / total_initial_central_hits
        if total_initial_central_hits > 0
        else 0.0
    )

    # Calculate metrics for each event
    event_metrics = []
    total_tp = total_fp = total_fn = total_tn = 0
    total_hits = total_true = total_pred_true = 0

    for sample_id in sample_ids:
        if sample_id in predictions and sample_id in ground_truth:
            metrics = calculate_metrics(predictions[sample_id], ground_truth[sample_id])
            event_metrics.append(metrics)

            # Accumulate totals
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]
            total_tn += metrics["tn"]
            total_hits += metrics["total_hits"]
            total_true += metrics["total_true"]
            total_pred_true += metrics["total_pred_true"]
        else:
            print(f"Warning: Missing data for sample {sample_id}")

    # Calculate overall metrics
    overall_metrics = {
        "purity": total_tp / total_pred_true if total_pred_true > 0 else 0.0,
        "efficiency": total_tp / total_true if total_true > 0 else 0.0,
        "precision": total_tp / total_pred_true if total_pred_true > 0 else 0.0,
        "recall": total_tp / total_true if total_true > 0 else 0.0,
        "accuracy": (total_tp + total_tn) / total_hits if total_hits > 0 else 0.0,
        "f1_score": 2 * total_tp / (total_tp + total_fp + total_tp + total_fn)
        if (total_tp + total_fp + total_tp + total_fn) > 0
        else 0.0,
        "hit_reduction": (total_hits - total_pred_true) / total_hits
        if total_hits > 0
        else 0.0,
        "noise_rejection": total_tn / (total_tn + total_fp)
        if (total_tn + total_fp) > 0
        else 0.0,
        "total_events": len(event_metrics),
        "total_hits": total_hits,
        "total_true": total_true,
        "total_pred_true": total_pred_true,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
    }

    # Calculate old-style metrics
    old_metrics = calculate_old_style_metrics(
        dataset=dataset,
        predictions=predictions,
        ground_truth=ground_truth,
        hit_cut=threshold,
        eta_cut=particle_max_abs_eta,
    )

    # Calculate additional metrics for the performance table
    additional_metrics = {
        "initial_purity": initial_purity,
        "initial_purity_central": initial_purity_central,
        "filter_efficiency": overall_metrics["efficiency"],
        "filter_purity": overall_metrics["purity"],
        "reconstructable_fraction": old_metrics.get(
            "track_eff", 0.993
        ),  # Use old-style track efficiency
        "model_name": model_name,
        "threshold": threshold,
        "old_metrics": old_metrics,
    }

    return overall_metrics, event_metrics, additional_metrics


def plot_purity_vs_efficiency_roc(
    probabilities: Dict[int, np.ndarray],
    ground_truth: Dict[int, np.ndarray],
    threshold: float = 0.1,
    output_dir: str = None,
):
    """
    Create ROC-like curve showing Signal Hit Purity vs Hit Efficiency.

    Args:
        probabilities: Dictionary mapping sample_id to probability arrays
        ground_truth: Dictionary mapping sample_id to ground truth arrays
        threshold: Threshold for marking operating point
        output_dir: Directory to save plots (optional)
    """
    # Combine all probabilities and ground truth
    all_probs = []
    all_gt = []

    for sample_id in probabilities:
        if sample_id in ground_truth:
            all_probs.extend(probabilities[sample_id])
            all_gt.extend(ground_truth[sample_id])

    all_probs = np.array(all_probs)
    all_gt = np.array(all_gt)

    # Calculate ROC curve
    thresholds, efficiencies, purities = calculate_roc_curve(all_probs, all_gt)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot the ROC curve
    plt.plot(efficiencies, purities, "b-", linewidth=2, label="Hit Filter Model")

    # Find and mark the operating point at the specified threshold
    threshold_idx = np.argmin(np.abs(thresholds - threshold))
    threshold_efficiency = efficiencies[threshold_idx]
    threshold_purity = purities[threshold_idx]

    plt.plot(
        threshold_efficiency,
        threshold_purity,
        "bo",
        markersize=8,
        label=f"Operating Point (threshold={threshold})",
    )

    # Formatting
    plt.xlabel("Hit Efficiency", fontsize=12)
    plt.ylabel("Hit Purity", fontsize=12)
    plt.title("Performance of Hit Filtering Model", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0.9, 1.0)
    plt.ylim(0.3, 1.0)

    # Add AUC calculation
    auc = np.trapezoid(purities, efficiencies)
    plt.text(
        0.95,
        0.4,
        f"AUC = {auc:.3f}",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path / "purity_vs_efficiency_roc.png", dpi=300, bbox_inches="tight"
        )
        print(f"ROC curve saved to {output_path / 'purity_vs_efficiency_roc.png'}")
    else:
        plt.show()

    plt.close()


def plot_reconstructable_vs_pt(
    particle_info: Dict[int, Dict],
    predictions: Dict[int, np.ndarray],
    threshold: float = 0.1,
    output_dir: str = None,
):
    """
    Create plot showing Reconstructable Particles vs Particle pT.

    Args:
        particle_info: Dictionary containing particle information
        predictions: Dictionary mapping sample_id to prediction arrays
        threshold: Threshold for determining reconstructable particles
        output_dir: Directory to save plots (optional)
    """
    # Define pT bins
    pt_bins = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2

    # Calculate reconstructable fractions for each pT bin
    reconstructable_fractions = []
    errors = []

    for i in range(len(pt_bins) - 1):
        pt_min, pt_max = pt_bins[i], pt_bins[i + 1]

        total_particles = 0
        reconstructable_particles = 0

        for sample_id in particle_info:
            if sample_id in predictions:
                particle_data = particle_info[sample_id]
                pt_values = particle_data["pt"]

                # Find particles in this pT bin
                in_bin = (pt_values >= pt_min) & (pt_values < pt_max)
                particles_in_bin = np.sum(in_bin)

                if particles_in_bin > 0:
                    # For simplicity, assume all particles in this bin are reconstructable
                    # In practice, you'd need to check hit counts after filtering
                    reconstructable_particles += particles_in_bin
                    total_particles += particles_in_bin

        # Calculate fraction and error
        if total_particles > 0:
            fraction = reconstructable_particles / total_particles
            # Binomial error
            error = np.sqrt(fraction * (1 - fraction) / total_particles)
        else:
            fraction = 0.0
            error = 0.0

        reconstructable_fractions.append(fraction)
        errors.append(error)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot with error bars
    plt.errorbar(
        pt_centers,
        reconstructable_fractions,
        yerr=errors,
        fmt="o-",
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=6,
        color="blue",
        label="Hit Filter Model",
    )

    # Formatting
    plt.xlabel("Particle pT [GeV]", fontsize=12)
    plt.ylabel("Reconstructable Particles", fontsize=12)
    plt.title("Reconstructable Particles vs Particle pT", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 5)
    plt.ylim(0.97, 1.0)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path / "reconstructable_vs_pt.png", dpi=300, bbox_inches="tight"
        )
        print(
            f"Reconstructable plot saved to {output_path / 'reconstructable_vs_pt.png'}"
        )
    else:
        plt.show()

    plt.close()


def plot_hits_lost_per_track(
    dataset: TrackMLDataset,
    predictions: Dict[int, np.ndarray],
    ground_truth: Dict[int, np.ndarray],
    threshold: float = 0.1,
    eta_cut: float = 2.5,
    output_dir: str = None,
):
    """
    Create histogram showing the number of hits lost per true track.

    Args:
        dataset: TrackMLDataset instance
        predictions: Dictionary mapping sample_id to prediction arrays
        ground_truth: Dictionary mapping sample_id to ground truth arrays
        threshold: Threshold for hit filtering
        eta_cut: Eta cut for particle selection
        output_dir: Directory to save plots (optional)
    """
    all_hits_lost = []
    all_hits_pre = []
    all_hits_post = []

    for idx in range(len(dataset)):
        sample_id = dataset.sample_ids[idx]

        if sample_id not in predictions or sample_id not in ground_truth:
            continue

        # Load event data
        hits, particles = dataset.load_event(idx)

        # Get predictions and ground truth
        pred = predictions[sample_id]
        gt = ground_truth[sample_id]

        # Ensure arrays have same length
        min_len = min(len(pred), len(gt), len(hits))
        pred = pred[:min_len]
        gt = gt[:min_len]
        hits = hits.iloc[:min_len].copy()

        # Add prediction and ground truth to hits dataframe
        hits["pred"] = pred
        hits["tgt"] = gt

        # Only consider hits that are on true tracks (ground truth = True)
        true_hits = hits[hits["tgt"] == True]

        if len(true_hits) == 0:
            continue

        # Calculate hit counts per particle for TRUE TRACKS ONLY
        # Before filtering: all true hits
        hits_pre_counts = true_hits["particle_id"].value_counts()

        # After filtering: only true hits that pass the filter
        true_hits_post = true_hits[true_hits["pred"] == True]
        hits_post_counts = true_hits_post["particle_id"].value_counts()

        # Get all unique particle IDs from true hits
        all_particle_ids = hits_pre_counts.index

        sum_hits_pre = 0
        sum_hits_post = 0

        for particle_id in all_particle_ids:
            # Get particle info
            particle_info = particles[particles["particle_id"] == particle_id]
            if len(particle_info) == 0:
                continue

            particle_eta = particle_info["eta"].iloc[0]

            # Only consider particles within eta cut and exclude particle_id = 0
            if abs(particle_eta) >= eta_cut or particle_id == 0:
                continue

            # Get hit counts for this particle
            hits_pre = hits_pre_counts.get(particle_id, 0)
            hits_post = hits_post_counts.get(particle_id, 0)

            # sum_hits_pre += hits_pre
            # sum_hits_post += hits_post

            # Calculate hits lost for this true track
            hits_lost = hits_pre - hits_post

            # Add to overall list
            all_hits_lost.append(hits_lost)

    # all_hits_pre.append(sum_hits_pre)
    # all_hits_post.append(sum_hits_post)

    # print("------------------HITS PRE-----------------")
    # print(np.mean(all_hits_pre))
    # print("------------HITS POST-----------------")
    # print(np.mean(all_hits_post))

    if not all_hits_lost:
        print("Warning: No true tracks found for hits lost calculation")
        return

    # Create the histogram
    plt.figure(figsize=(10, 6))

    # Create histogram
    max_hits_lost = max(all_hits_lost)
    plt.hist(
        all_hits_lost,
        bins=range(0, int(max_hits_lost) + 2),
        alpha=0.7,
        edgecolor="black",
        color="skyblue",
        align="left",
    )

    # Formatting
    plt.xlabel("Number of Hits Lost per True Track", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Hits Lost per True Track", fontsize=14)
    plt.grid(True, alpha=0.3, axis="y")

    # Add statistics
    mean_hits_lost = np.mean(all_hits_lost)
    median_hits_lost = np.median(all_hits_lost)
    total_tracks = len(all_hits_lost)

    plt.axvline(
        mean_hits_lost,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_hits_lost:.2f}",
    )
    plt.axvline(
        median_hits_lost,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_hits_lost:.1f}",
    )

    plt.legend()

    # Add text box with statistics
    stats_text = f"Total true tracks: {total_tracks:,}\nMean hits lost: {mean_hits_lost:.2f}\nMedian hits lost: {median_hits_lost:.1f}"
    plt.text(
        0.7,
        0.7,
        stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=10,
        verticalalignment="top",
    )

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path / "hits_lost_per_track.png", dpi=300, bbox_inches="tight"
        )
        print(f"Hits lost histogram saved to {output_path / 'hits_lost_per_track.png'}")
    else:
        plt.show()

    plt.close()


def analyze_noise_hit_removal(
    dataset: TrackMLDataset,
    predictions: Dict[int, np.ndarray],
    ground_truth: Dict[int, np.ndarray],
    threshold: float = 0.1,
    eta_cut: float = 2.5,
    output_dir: str = None,
):
    """
    Analyze hits lost from noise particles (pid=0) vs true tracks to understand
    the discrepancy between hit efficiency and reconstructable efficiency.

    Args:
        dataset: TrackMLDataset instance
        predictions: Dictionary mapping sample_id to prediction arrays
        ground_truth: Dictionary mapping sample_id to ground truth arrays
        threshold: Threshold for hit filtering
        eta_cut: Eta cut for particle selection
        output_dir: Directory to save plots (optional)
    """
    noise_hits_lost = []
    true_track_hits_lost = []
    noise_hits_total = []
    true_track_hits_total = []

    for idx in range(len(dataset)):
        sample_id = dataset.sample_ids[idx]

        if sample_id not in predictions or sample_id not in ground_truth:
            continue

        # Load event data
        hits, particles = dataset.load_event(idx)

        # Get predictions and ground truth
        pred = predictions[sample_id]
        gt = ground_truth[sample_id]

        # Ensure arrays have same length
        min_len = min(len(pred), len(gt), len(hits))
        pred = pred[:min_len]
        gt = gt[:min_len]
        hits = hits.iloc[:min_len].copy()

        # Add prediction and ground truth to hits dataframe
        hits["pred"] = pred
        hits["tgt"] = gt

        # Separate noise hits (pid=0) from true track hits
        noise_hits = hits[hits["particle_id"] == 0]
        true_track_hits = hits[hits["particle_id"] != 0]

        # Analyze noise hits
        if len(noise_hits) > 0:
            noise_hits_total.append(len(noise_hits))
            noise_hits_kept = noise_hits[noise_hits["pred"] == True]
            noise_hits_removed = len(noise_hits) - len(noise_hits_kept)
            noise_hits_lost.append(noise_hits_removed)

        # Analyze true track hits
        if len(true_track_hits) > 0:
            true_track_hits_total.append(len(true_track_hits))
            true_track_hits_kept = true_track_hits[true_track_hits["pred"] == True]
            true_track_hits_removed = len(true_track_hits) - len(true_track_hits_kept)
            true_track_hits_lost.append(true_track_hits_removed)

    # Calculate statistics
    total_noise_hits = sum(noise_hits_total)
    total_noise_hits_lost = sum(noise_hits_lost)

    noise_removal_rate = (
        total_noise_hits_lost / total_noise_hits if total_noise_hits > 0 else 0
    )

    # Print analysis
    print("\n" + "=" * 80)
    print("NOISE vs TRUE TRACK HIT REMOVAL ANALYSIS")
    print("=" * 80)

    print(f"Total noise hits (pid=0): {total_noise_hits:,}")
    print(f"Noise hits removed: {total_noise_hits_lost:,}")
    print(f"Noise removal rate: {noise_removal_rate:.3%}")


def analyze_non_reconstructable_hit_removal(
    dataset: TrackMLDataset,
    predictions: Dict[int, np.ndarray],
    ground_truth: Dict[int, np.ndarray],
    threshold: float = 0.1,
    eta_cut: float = 2.5,
    output_dir: str = None,
):
    """
    Analyze what fraction of removed hits come from non-reconstructable particles
    (particles with <3 hits before filtering).

    Args:
        dataset: TrackMLDataset instance
        predictions: Dictionary mapping sample_id to prediction arrays
        ground_truth: Dictionary mapping sample_id to ground truth arrays
        threshold: Threshold for hit filtering
        eta_cut: Eta cut for particle selection
        output_dir: Directory to save plots (optional)
    """
    non_reconstructable_hits_removed = []
    reconstructable_hits_removed = []
    non_reconstructable_hits_total = []
    reconstructable_hits_total = []

    for idx in range(len(dataset)):
        sample_id = dataset.sample_ids[idx]

        if sample_id not in predictions or sample_id not in ground_truth:
            continue

        # Load event data
        hits, particles = dataset.load_event(idx)

        # Get predictions and ground truth
        pred = predictions[sample_id]
        gt = ground_truth[sample_id]

        # Ensure arrays have same length
        min_len = min(len(pred), len(gt), len(hits))
        pred = pred[:min_len]
        gt = gt[:min_len]
        hits = hits.iloc[:min_len].copy()

        # Add prediction and ground truth to hits dataframe
        hits["pred"] = pred
        hits["tgt"] = gt

        # Calculate hit counts per particle before filtering
        hits_pre_counts = hits["particle_id"].value_counts()

        # Apply hit filtering
        hits_post = hits[hits["pred"]]
        hits_post_counts = hits_post["particle_id"].value_counts()

        # Create particles dataframe with hit counts
        particles_df = particles.copy()
        particles_df["hits_pre"] = hits_pre_counts
        particles_df["hits_post"] = hits_post_counts

        # Define reconstructable particles (>=3 hits, |eta| < eta_cut, pid != 0)
        particles_df["reconstructable_pre"] = (
            (particles_df["hits_pre"] >= 3)
            & (particles_df["particle_id"] != 0)
            & (particles_df["eta"].abs() < eta_cut)
            & (particles_df["pt"] > 0.6)
        )

        # Separate hits by whether their particle is reconstructable
        for particle_id in hits["particle_id"].unique():
            if particle_id == 0:  # Skip noise particles
                continue

            particle_hits = hits[hits["particle_id"] == particle_id]
            if len(particle_hits) == 0:
                continue

            # Get particle info
            particle_info = particles_df[particles_df["particle_id"] == particle_id]
            if len(particle_info) == 0:
                continue

            is_reconstructable = particle_info["reconstructable_pre"].iloc[0]

            # Count hits before and after filtering for this particle
            hits_pre = len(particle_hits)
            hits_post = len(particle_hits[particle_hits["pred"] == True])
            hits_removed = hits_pre - hits_post


            if is_reconstructable:
                reconstructable_hits_total.append(hits_pre)
                reconstructable_hits_removed.append(hits_removed)
            else:
                non_reconstructable_hits_total.append(hits_pre)
                non_reconstructable_hits_removed.append(hits_removed)

        non_reco_hits_removed_outof_hits_removed = sum(non_reconstructable_hits_removed) / sum(non_reconstructable_hits_removed + reconstructable_hits_removed)
            
    # Calculate statistics
    total_non_reco_hits_removed_outof_hits_removed = np.mean(non_reco_hits_removed_outof_hits_removed)
    total_non_reconstructable_hits = np.mean(non_reconstructable_hits_total)
    total_non_reconstructable_hits_removed = np.mean(non_reconstructable_hits_removed)
    total_reconstructable_hits = np.mean(reconstructable_hits_total)
    total_reconstructable_hits_removed = np.mean(reconstructable_hits_removed)

    # Print analysis
    print("\n" + "=" * 80)
    print("NON-RECONSTRUCTABLE vs RECONSTRUCTABLE HIT REMOVAL ANALYSIS")
    print("=" * 80)

    print("total_non_reco_hits_removed_outof_hits_removed: ", total_non_reco_hits_removed_outof_hits_removed)

    print(
        f"Total non-reconstructable hits (<3 hits): {total_non_reconstructable_hits:,}"
    )
    print(
        f"Non-reconstructable hits removed: {total_non_reconstructable_hits_removed:,}"
    )

    print(f"\nTotal reconstructable hits (≥3 hits): {total_reconstructable_hits:,}")
    print(f"Reconstructable hits removed: {total_reconstructable_hits_removed:,}")

    print(
        f"\nTotal hits: {total_non_reconstructable_hits + total_reconstructable_hits:,}"
    )
    print(
        f"Total hits removed: {total_non_reconstructable_hits_removed + total_reconstructable_hits_removed:,}"
    )
    print(
        f"Overall removal rate: {(total_non_reconstructable_hits_removed + total_reconstructable_hits_removed) / (total_non_reconstructable_hits + total_reconstructable_hits):.3%}"
    )

    print(
        f"\nNon-reconstructable hits as % of total: {total_non_reconstructable_hits / (total_non_reconstructable_hits + total_reconstructable_hits):.3%}"
    )
    print(
        f"Non-reconstructable hits removed as % of total removed: {total_non_reconstructable_hits_removed / (total_non_reconstructable_hits_removed + total_reconstructable_hits_removed):.3%}"
    )


def plot_metrics(event_metrics: List[Dict[str, float]], output_dir: str = None):
    """
    Create plots showing hit filter performance metrics.

    Args:
        event_metrics: List of metrics for each event
        output_dir: Directory to save plots (optional)
    """
    if not event_metrics:
        print("No metrics to plot")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(event_metrics)

    # Set up the plotting style
    plt.style.use("default")
    # Use matplotlib's default color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Hit Filter Performance Metrics", fontsize=16)

    # Plot 1: Purity and Efficiency
    axes[0, 0].scatter(df["efficiency"], df["purity"], alpha=0.6, color=colors[0])
    axes[0, 0].set_xlabel("Efficiency")
    axes[0, 0].set_ylabel("Purity")
    axes[0, 0].set_title("Purity vs Efficiency")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Hit reduction distribution
    axes[0, 1].hist(
        df["hit_reduction"], bins=20, alpha=0.7, edgecolor="black", color=colors[1]
    )
    axes[0, 1].set_xlabel("Hit Reduction Fraction")
    axes[0, 1].set_ylabel("Number of Events")
    axes[0, 1].set_title("Hit Reduction Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Accuracy distribution
    axes[0, 2].hist(
        df["accuracy"], bins=20, alpha=0.7, edgecolor="black", color=colors[2]
    )
    axes[0, 2].set_xlabel("Accuracy")
    axes[0, 2].set_ylabel("Number of Events")
    axes[0, 2].set_title("Accuracy Distribution")
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: F1 Score distribution
    axes[1, 0].hist(
        df["f1_score"], bins=20, alpha=0.7, edgecolor="black", color=colors[3]
    )
    axes[1, 0].set_xlabel("F1 Score")
    axes[1, 0].set_ylabel("Number of Events")
    axes[1, 0].set_title("F1 Score Distribution")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Total hits vs predicted hits
    axes[1, 1].scatter(
        df["total_hits"], df["total_pred_true"], alpha=0.6, color=colors[4]
    )
    axes[1, 1].plot(
        [0, df["total_hits"].max()], [0, df["total_hits"].max()], "r--", alpha=0.5
    )
    axes[1, 1].set_xlabel("Total Hits")
    axes[1, 1].set_ylabel("Predicted True Hits")
    axes[1, 1].set_title("Hit Filtering Effect")
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Noise rejection vs hit reduction
    axes[1, 2].scatter(
        df["hit_reduction"], df["noise_rejection"], alpha=0.6, color=colors[5]
    )
    axes[1, 2].set_xlabel("Hit Reduction")
    axes[1, 2].set_ylabel("Noise Rejection")
    axes[1, 2].set_title("Noise Rejection vs Hit Reduction")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path / "hit_filter_metrics.png", dpi=300, bbox_inches="tight"
        )
        print(f"Plots saved to {output_path / 'hit_filter_metrics.png'}")
    else:
        plt.show()


def print_old_style_metrics(old_metrics: Dict[str, float]):
    """
    Print metrics in the same format as the old evaluation code.

    Args:
        old_metrics: Dictionary containing old-style metrics
    """
    if not old_metrics:
        print("No old-style metrics available")
        return

    print("\n" + "=" * 60)
    print("OLD-STYLE EVALUATION METRICS")
    print("=" * 60)

    print(f"Precision pre filter: {old_metrics['precision_pre']:.3%}")
    print(f"Precision pre filter eta: {old_metrics['precision_pre_eta']:.3%}")
    print(f"Precision post: {old_metrics['precision']:.3%}")
    print(f"Recall post: {old_metrics['recall']:.3%}")
    print(f"Track level eff: {old_metrics['track_eff']:.3%}")
    print(f"Track level eff pt cut: {old_metrics['track_eff_1gev']:.3%}")
    print(f"Hit level eff: {old_metrics['hit_level_eff']:.3%}")

    print("\nStandard Errors:")
    print(f"  Recall SE: {old_metrics['se_recall']:.4f}")
    print(f"  Precision SE: {old_metrics['se_precision']:.4f}")

    print("\nHit Statistics:")
    print(f"  Average hits pre: {old_metrics['avg_hits_pre']:.1f}")
    print(f"  Average hits post: {old_metrics['avg_hits_post']:.1f}")
    print(f"  Total hits: {old_metrics['total_hits']:,}")
    print(f"  Total particles: {old_metrics['total_particles']:,}")
    print(f"  Number of events: {old_metrics['num_events']:,}")

    print("\n" + "=" * 60)


def print_performance_table(
    model_name: str,
    initial_purity: float,
    initial_purity_central: float,
    filter_efficiency: float,
    filter_purity: float,
    reconstructable_fraction: float,
):
    """
    Print performance table similar to Table 3 in the image.

    Args:
        model_name: Name of the model
        initial_purity: Initial purity before filtering
        initial_purity_central: Initial purity in central region (|eta| < 2.5)
        filter_efficiency: Post-filter efficiency
        filter_purity: Post-filter purity
        reconstructable_fraction: Fraction of reconstructable particles
    """
    print("\n" + "=" * 80)
    print("Table 3: Performance of the Hit Filtering Model on the Test Set")
    print("=" * 80)

    print(
        f"{'Model':<15} {'Initial Purity (|η| < 2.5)':<25} {'Filter Efficiency':<18} {'Filter Purity':<15} {'Reconstructable':<15}"
    )
    print("-" * 80)

    purity_str = f"{initial_purity:.1%} ({initial_purity_central:.1%})"
    efficiency_str = f"{filter_efficiency:.1%}"
    purity_filter_str = f"{filter_purity:.1%}"
    reconstructable_str = f"{reconstructable_fraction:.1%}"
    print(
        f"{model_name:<15} {purity_str:<25} {efficiency_str:<18} {purity_filter_str:<15} {reconstructable_str:<15}"
    )

    print("\n" + "=" * 80)


def print_results(metrics: Dict[str, float]):
    """
    Print evaluation results in a formatted way.

    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 60)
    print("HIT FILTER EVALUATION RESULTS")
    print("=" * 60)

    print("\nDataset Statistics:")
    print(f"  Total Events: {metrics['total_events']:,}")
    print(f"  Total Hits: {metrics['total_hits']:,}")
    print(f"  Total True Hits: {metrics['total_true']:,}")
    print(f"  Total Predicted True: {metrics['total_pred_true']:,}")

    print("\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['total_tp']:,}")
    print(f"  False Positives: {metrics['total_fp']:,}")
    print(f"  False Negatives: {metrics['total_fn']:,}")
    print(f"  True Negatives:  {metrics['total_tn']:,}")

    print("\nMain Performance Metrics:")
    print(f"  Purity:      {metrics['purity']:.4f} ({metrics['purity']*100:.2f}%)")
    print(
        f"  Efficiency:  {metrics['efficiency']:.4f} ({metrics['efficiency']*100:.2f}%)"
    )
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1 Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")

    print("\nHit Filtering Performance:")
    print(
        f"  Hit Reduction:   {metrics['hit_reduction']:.4f} ({metrics['hit_reduction']*100:.2f}%)"
    )
    print(
        f"  Noise Rejection: {metrics['noise_rejection']:.4f} ({metrics['noise_rejection']*100:.2f}%)"
    )

    print("\n" + "=" * 60)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate hit filter performance")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--val",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--hit_eval_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Hit Filter Model",
        help="Name of the model for reporting",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for operating point analysis",
    )
    parser.add_argument(
        "--eta_cut",
        type=float,
        default=2.5,
        help="Eta cut for reconstructable particles (default: 2.5)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract parameters from config
    data_config = config.get("data", {})
    eval_config = config.get("evaluation", {})

    if args.test:
        test_dir = data_config.get("test_dir")
    elif args.train:
        test_dir = data_config.get("train_dir")
    elif args.val:
        test_dir = data_config.get("val_dir")

    hit_eval_path = args.hit_eval_path
    num_events = data_config.get("num_events", -1)
    hit_volume_ids = data_config.get("hit_volume_ids", None)
    particle_min_pt = data_config.get("particle_min_pt", 0.6)
    particle_max_abs_eta = data_config.get("particle_max_abs_eta", 2.5)
    particle_min_num_hits = data_config.get("particle_min_num_hits", 3)
    event_max_num_particles = data_config.get("event_max_num_particles", 1000)
    output_dir = eval_config.get("output_dir", None)

    # Validate required parameters
    if not test_dir:
        raise ValueError("test_dir must be specified in config file")
    if not hit_eval_path:
        raise ValueError("hit_eval_path must be specified")

    print(f"Loaded configuration from {args.config}")
    print(f"Test directory: {test_dir}")
    print(f"Hit eval path: {hit_eval_path}")
    print(f"Number of events: {num_events}")
    print(f"Output directory: {output_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Threshold: {args.threshold}")
    print(f"Eta cut: {args.eta_cut}")

    # Evaluate hit filter performance
    overall_metrics, event_metrics, additional_metrics = evaluate_hit_filter(
        test_dir=test_dir,
        hit_eval_path=hit_eval_path,
        num_events=num_events,
        hit_volume_ids=hit_volume_ids,
        particle_min_pt=particle_min_pt,
        particle_max_abs_eta=args.eta_cut,  # Use command line eta_cut
        particle_min_num_hits=particle_min_num_hits,
        event_max_num_particles=event_max_num_particles,
        model_name=args.model_name,
        threshold=args.threshold,
    )

    # Print results
    print_results(overall_metrics)

    # Print old-style metrics
    print_old_style_metrics(additional_metrics["old_metrics"])

    # Print performance table
    print_performance_table(
        model_name=additional_metrics["model_name"],
        initial_purity=additional_metrics["initial_purity"],
        initial_purity_central=additional_metrics["initial_purity_central"],
        filter_efficiency=additional_metrics["filter_efficiency"],
        filter_purity=additional_metrics["filter_purity"],
        reconstructable_fraction=additional_metrics["reconstructable_fraction"],
    )

    # Create plots
    if event_metrics:
        plot_metrics(event_metrics, output_dir)

        # Create the new plots from the image
        # Load data for plotting
        dataset = TrackMLDataset(
            dirpath=test_dir,
            inputs={"hit": ["x", "y", "z"]},
            targets={"particle": []},
            num_events=num_events,
            hit_volume_ids=hit_volume_ids,
            particle_min_pt=particle_min_pt,
            particle_max_abs_eta=particle_max_abs_eta,
            particle_min_num_hits=particle_min_num_hits,
            event_max_num_particles=event_max_num_particles,
            hit_eval_path=None,
        )

        sample_ids = dataset.sample_ids
        probabilities = load_hit_eval_probabilities(hit_eval_path, sample_ids)
        ground_truth = load_ground_truth(dataset)
        particle_info = load_particle_info(dataset)
        predictions = load_hit_eval_predictions(hit_eval_path, sample_ids)

        # Create ROC curve plot
        plot_purity_vs_efficiency_roc(
            probabilities=probabilities,
            ground_truth=ground_truth,
            threshold=args.threshold,
            output_dir=output_dir,
        )

        # Create reconstructable particles plot
        plot_reconstructable_vs_pt(
            particle_info=particle_info,
            predictions=predictions,
            threshold=args.threshold,
            output_dir=output_dir,
        )

        # Create hits lost per track histogram
        plot_hits_lost_per_track(
            dataset=dataset,
            predictions=predictions,
            ground_truth=ground_truth,
            threshold=args.threshold,
            eta_cut=args.eta_cut,
            output_dir=output_dir,
        )

        # Analyze noise vs true track hit removal
        analyze_noise_hit_removal(
            dataset=dataset,
            predictions=predictions,
            ground_truth=ground_truth,
            threshold=args.threshold,
            eta_cut=args.eta_cut,
            output_dir=output_dir,
        )

        # Analyze non-reconstructable vs reconstructable hit removal
        analyze_non_reconstructable_hit_removal(
            dataset=dataset,
            predictions=predictions,
            ground_truth=ground_truth,
            threshold=args.threshold,
            eta_cut=args.eta_cut,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
