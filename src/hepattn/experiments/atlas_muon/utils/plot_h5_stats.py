#!/usr/bin/env python3
# ruff: noqa: DTZ005,PLR6201,RUF059,ICN001,EXE001
"""Script to generate HDF5 file analysis plots.

This script uses the h5Analyzer and h5TrackVisualizerMDTGeometry classes
to create hit distribution, track analysis, and event display plots
for HDF5 files.
"""

import argparse
import gc
from datetime import datetime
from pathlib import Path

import h5py

# Set matplotlib backend before any other matplotlib imports
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .data import AtlasMuonDataModule, AtlasMuonDataset
from .data_vis.h5_analyzer import h5Analyzer
from .data_vis.h5_config import DEFAULT_TREE_NAME, HISTOGRAM_SETTINGS
from .data_vis.track_visualizer_h5_MDTGeometry import h5TrackVisualizerMDTGeometry


def create_output_directory(base_output_dir: str) -> Path:
    """Create timestamped output directory for the plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"h5_analysis_plots_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_inputs_targets_from_config(config_path: str) -> tuple:
    """Load inputs and targets configuration from a YAML file."""
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)
    data_cfg = config.get("data", {})
    inputs = data_cfg.get("inputs", {})
    targets = data_cfg.get("targets", {})
    return inputs, targets


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    return "".join(c for c in filename if c.isalnum() or c in (" ", "-", "_")).rstrip()


def generate_plots_for_file(
    datamodule: AtlasMuonDataModule,
    config_key: str,
    output_dir: Path,
    num_events: int = 10,
    generate_histograms: bool = True,
    histogram_categories: list[str] | None = None,
    dataset: AtlasMuonDataset | None = None,
    hit_eval_path: str | None = None,
) -> dict[str, list[str] | int] | None:
    """Generate all plots for a single HDF5 file.

    Parameters
    ----------
    datamodule : AtlasMuonDataModule
        DataModule providing the data.
    config_key : str
        Configuration key for naming.
    output_dir : Path
        Output directory for plots.
    num_events : int
        Number of events to analyze.
    generate_histograms : bool
        Whether to generate branch histograms.
    histogram_categories : list, optional
        Categories of histograms to generate.
    dataset : AtlasMuonDataset, optional
        Dataset for event visualization.
    hit_eval_path : str, optional
        Path to hit evaluation file for filtered output naming.

    Returns:
    -------
    dict or None
        File info dictionary if successful, None otherwise.
    """
    if histogram_categories is None:
        histogram_categories = ["standard", "signal_background"]

    print(f"\n{'=' * 80}")
    print(f"Processing: {config_key}")
    print(f"{'=' * 80}")

    analyzer = h5Analyzer(datamodule, num_events)

    file_output_dir = output_dir / sanitize_filename(config_key)
    if hit_eval_path is not None:
        file_output_dir = file_output_dir.with_name(file_output_dir.name + "_FILTERED")

    file_output_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # Hits per event analysis
    print("\n--- Analyzing hits per event ---")
    hits_plot_path = file_output_dir / f"{config_key}_hits_distribution.png"
    hits_data = analyzer.analyze_hits_per_event(output_plot_path=str(hits_plot_path))

    if hits_data is None:
        print("ERROR: Failed to analyze hits per event")
        success = False
    else:
        print(f"✓ Hits distribution plot saved to: {hits_plot_path}")

    # Tracks and track lengths analysis
    print("\n--- Analyzing tracks and track lengths ---")
    tracks_plot_path = file_output_dir / f"{config_key}_tracks_analysis.png"
    tracks_data = analyzer.analyze_tracks_and_lengths(output_plot_path=str(tracks_plot_path))

    if tracks_data is None:
        print("ERROR: Failed to analyze tracks and lengths")
        success = False
    else:
        print(f"✓ Tracks analysis plot saved to: {tracks_plot_path}")

    # Generate histograms
    if generate_histograms:
        for histogram_category in histogram_categories:
            print(f"\n--- Generating histograms (category: {histogram_category}) ---")
            analyzer.generate_feature_histograms_with_categories(
                output_dir=file_output_dir / f"histograms_{histogram_category}",
                histogram_settings=HISTOGRAM_SETTINGS,
                category=histogram_category,
            )

    # Plot number of true hits
    if dataset is not None:
        track_analyzer = h5TrackVisualizerMDTGeometry(dataset=dataset)
        print("\n--- Plotting number of true hits ---")
        track_analyzer.plot_and_save_true_hits_histogram(
            dataloader=datamodule.test_dataloader(shuffle=True),
            num_events=num_events,
            save_path=file_output_dir / f"{config_key}_true_hits_histogram.png",
        )

        # Generate random event displays
        print("\n--- Generating random event displays ---")
        rng = np.random.default_rng(42)
        random_indices = rng.choice(min(num_events, len(dataset)), size=min(100, num_events), replace=False)
        events_dir = file_output_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)

        for idx in random_indices:
            print(f"Processing random event {idx + 1}/{num_events}")
            try:
                track_analyzer.plot_and_save_event(
                    event_index=idx,
                    save_path=events_dir / f"{config_key}_random_event_{idx}.png",
                )
                plt.close("all")
                gc.collect()
            except (ValueError, KeyError, OSError, RuntimeError) as e:
                print(f"Error processing event {idx}: {e}")
                continue

    return {"success": success} if success else None


def calculate_filter_metrics(
    test_dataloader,
    hit_eval_filepath: str,
    num_events: int,
) -> dict[str, float]:
    """Calculate hit filter performance metrics.

    Parameters
    ----------
    test_dataloader : DataLoader
        DataLoader for the test set.
    hit_eval_filepath : str
        Path to the hit evaluation HDF5 file.
    num_events : int
        Number of events to process.

    Returns:
    -------
    dict
        Dictionary of averaged metrics.
    """
    metrics_lists = {
        "nh_total_pre": [],
        "nh_total_post": [],
        "nh_pred_true": [],
        "nh_pred_false": [],
        "nh_valid_pre": [],
        "nh_valid_post": [],
        "nh_noise_pre": [],
        "nh_noise_post": [],
        "acc": [],
        "valid_recall": [],
        "valid_precision": [],
        "noise_recall": [],
        "noise_precision": [],
    }

    batch_size_metrics = 1000

    for batch_start in range(0, num_events, batch_size_metrics):
        batch_end = min(batch_start + batch_size_metrics, num_events)
        print(f"Processing metrics batch {batch_start}-{batch_end}/{num_events}")

        with h5py.File(hit_eval_filepath, "r") as hit_eval_file:
            for idx, batch in enumerate(
                tqdm(
                    test_dataloader,
                    total=batch_end - batch_start,
                    desc=f"Processing batch {batch_start // batch_size_metrics + 1}",
                )
            ):
                if idx < batch_start:
                    continue
                if idx >= batch_end:
                    break

                inputs_batch, targets_batch = batch

                pred = hit_eval_file[f"{idx}/preds/final/hit_filter/hit_on_valid_particle"][0]
                pred = torch.from_numpy(pred).to(targets_batch["hit_on_valid_particle"].device)
                if pred.dtype != torch.bool:
                    pred = pred.bool()

                true = targets_batch["hit_on_valid_particle"][targets_batch["hit_valid"]]
                tp = (pred & true).sum()
                tn = ((~pred) & (~true)).sum()

                event_metrics = {
                    "nh_total_pre": float(pred.numel()),
                    "nh_total_post": float(pred.sum().item()),
                    "nh_pred_true": pred.float().sum().item(),
                    "nh_pred_false": (~pred).float().sum().item(),
                    "nh_valid_pre": true.float().sum().item(),
                    "nh_valid_post": (pred & true).float().sum().item(),
                    "nh_noise_pre": (~true).float().sum().item(),
                    "nh_noise_post": (pred & ~true).float().sum().item(),
                    "acc": (pred == true).float().mean().item(),
                    "valid_recall": (tp / true.sum()).item() if true.sum() > 0 else float("nan"),
                    "valid_precision": (tp / pred.sum()).item() if pred.sum() > 0 else float("nan"),
                    "noise_recall": (tn / (~true).sum()).item() if (~true).sum() > 0 else float("nan"),
                    "noise_precision": (tn / (~pred).sum()).item() if (~pred).sum() > 0 else float("nan"),
                }

                for k, v in event_metrics.items():
                    metrics_lists[k].append(v)

        gc.collect()

    return {k: float(np.nanmean(v)) for k, v in metrics_lists.items()}


def save_metrics_report(
    output_dir: Path,
    averages: dict[str, float],
) -> None:
    """Save metrics report to a text file."""
    output_path = output_dir / "hit_filter_performance_metrics.txt"
    with output_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("HIT FILTER PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n")
        f.writelines(f"{k}: {v}\n" for k, v in averages.items())
        f.write("\n")
    print(f"Saved average metrics to {output_path}")


def save_detector_technology_stats(
    output_dir: Path,
    tech_stats: dict,
) -> None:
    """Save detector technology statistics to a text file."""
    tech_stats_path = output_dir / "detector_technology_statistics_filtered.txt"
    with tech_stats_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("DETECTOR TECHNOLOGY STATISTICS (FILTERED DATASET)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Events processed: {tech_stats['overall']['events_processed']}\n")
        f.write(f"Total hits: {tech_stats['overall']['total_hits']:,}\n")
        f.write(f"Total true hits: {tech_stats['overall']['total_true_hits']:,}\n\n")

        f.write("TECHNOLOGY DISTRIBUTION (ABSOLUTE NUMBERS):\n")
        f.write("-" * 50 + "\n")
        for tech_name in ["MDT", "RPC", "TGC", "STGC", "MM"]:
            if tech_name in tech_stats:
                stats = tech_stats[tech_name]
                f.write(f"{tech_name}:\n")
                f.write(f"  Total hits: {stats['total_hits']:,}\n")
                f.write(f"  True hits: {stats['true_hits']:,}\n\n")

        f.write("TECHNOLOGY DISTRIBUTION (PERCENTAGES):\n")
        f.write("-" * 50 + "\n")
        f.write("Percentage of total hits by technology:\n")
        for tech_name in ["MDT", "RPC", "TGC", "STGC", "MM"]:
            if tech_name in tech_stats:
                stats = tech_stats[tech_name]
                f.write(f"  {tech_name}: {stats['total_hits_percentage']:.2f}%\n")

        f.write("\nPercentage of true hits by technology:\n")
        for tech_name in ["MDT", "RPC", "TGC", "STGC", "MM"]:
            if tech_name in tech_stats:
                stats = tech_stats[tech_name]
                f.write(f"  {tech_name}: {stats['true_hits_percentage']:.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Saved detector technology statistics to {tech_stats_path}")


def main() -> None:
    """Main function to process HDF5 files."""
    parser = argparse.ArgumentParser(description="Generate HDF5 analysis plots for configured files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./h5_analysis_output",
        help="Base directory for output plots (default: ./h5_analysis_output)",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default=DEFAULT_TREE_NAME,
        help=f"Tree name (default: {DEFAULT_TREE_NAME})",
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="+",
        help="Specific config keys to process (default: all keys)",
    )
    parser.add_argument(
        "--num-events",
        "-n",
        type=int,
        default=-1,
        help="Number of events to use for statistics",
    )
    parser.add_argument(
        "--skip-histograms",
        action="store_true",
        help="Skip generation of branch histograms",
    )
    parser.add_argument(
        "--histogram-categories",
        type=str,
        nargs="+",
        default=["standard", "signal_background"],
        choices=["standard", "signal_background"],
        help="Categories of histograms to generate",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the HDF5 data directory",
    )
    parser.add_argument(
        "--hit-eval-path",
        type=str,
        default=None,
        help="Path to hit evaluation HDF5 file (optional, for filtered data analysis)",
    )

    args = parser.parse_args()

    output_dir = create_output_directory(args.output_dir)
    print(f"Output directory: {output_dir}")
    print(f"Data directory: {args.data_dir}")
    if args.hit_eval_path:
        print(f"Hit evaluation file: {args.hit_eval_path}")

    # Load inputs and targets
    inputs, targets = load_inputs_targets_from_config(args.config_path)

    # Process hit evaluation metrics if hit_eval_path is provided
    if args.hit_eval_path is not None:
        print("Running quick performance stats (HIT_EVAL_FILEPATH is set)")

        inputs_eval = {k: list(v) for k, v in inputs.items()}
        targets_eval = {k: list(v) for k, v in targets.items()}

        datamodule_eval = AtlasMuonDataModule(
            train_dir=args.data_dir,
            val_dir=args.data_dir,
            test_dir=args.data_dir,
            num_workers=10,
            num_train=-1,
            num_val=-1,
            num_test=-1,
            batch_size=1,
            inputs=inputs_eval,
            targets=targets_eval,
        )
        datamodule_eval.setup("test")
        test_dataloader_eval = datamodule_eval.test_dataloader(shuffle=True)

        # Calculate metrics
        averages = calculate_filter_metrics(test_dataloader_eval, args.hit_eval_path, args.num_events)
        save_metrics_report(output_dir, averages)

        # Calculate detector technology statistics
        print("\n--- Calculating detector technology statistics ---")
        dataset_eval = AtlasMuonDataset(
            dirpath=args.data_dir,
            inputs=inputs_eval,
            targets=targets_eval,
        )
        track_analyzer_eval = h5TrackVisualizerMDTGeometry(dataset=dataset_eval)
        tech_stats = track_analyzer_eval.calculate_detector_technology_statistics(test_dataloader_eval, args.num_events)
        if tech_stats:
            save_detector_technology_stats(output_dir, tech_stats)

        # Cleanup
        del test_dataloader_eval
        del datamodule_eval
        gc.collect()

    # Create main datamodule
    inputs_main = {k: list(v) for k, v in inputs.items()}
    targets_main = {k: list(v) for k, v in targets.items()}

    datamodule = AtlasMuonDataModule(
        train_dir=args.data_dir,
        val_dir=args.data_dir,
        test_dir=args.data_dir,
        num_workers=10,
        num_train=-1,
        num_val=-1,
        num_test=-1,
        batch_size=1,
        inputs=inputs_main,
        targets=targets_main,
        hit_eval_train=args.hit_eval_path,
        hit_eval_val=args.hit_eval_path,
        hit_eval_test=args.hit_eval_path,
    )

    dataset = AtlasMuonDataset(
        dirpath=args.data_dir,
        inputs=inputs_main,
        targets=targets_main,
        hit_eval_path=args.hit_eval_path,
    )

    datamodule.setup("test")

    # Generate plots
    config_key = Path(args.data_dir).name
    generate_plots_for_file(
        datamodule,
        config_key,
        output_dir,
        args.num_events,
        not args.skip_histograms,
        args.histogram_categories,
        dataset=dataset,
        hit_eval_path=args.hit_eval_path,
    )


if __name__ == "__main__":
    main()
