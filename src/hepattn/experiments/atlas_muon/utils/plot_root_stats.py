#!/usr/bin/env python3
# ruff: noqa: TRY300,DTZ005,PTH207,PLR6201,PLC0415,EXE001
"""Script to generate ROOT file analysis plots for all configured ROOT files.

This script uses the RootAnalyzer and TrackVisualizerMDTGeometry classes
to create hit distribution, track analysis, and event display plots.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from .data_vis.config import DEFAULT_TREE_NAME, HISTOGRAM_SETTINGS
from .data_vis.root_analyzer import RootAnalyzer
from .data_vis.track_visualizer_MDTGeometry import TrackVisualizerMDTGeometry


def create_output_directory(base_output_dir: str) -> Path:
    """Create timestamped output directory for the plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"root_analysis_plots_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    return "".join(c for c in filename if c.isalnum() or c in (" ", "-", "_")).rstrip()


def generate_plots_for_file(
    root_file_path: str,
    config_key: str,
    output_dir: Path,
    tree_name: str = DEFAULT_TREE_NAME,
    num_events: int = 10,
    min_tracks: int = 1,
    generate_histograms: bool = True,
) -> dict[str, list[str] | int] | None:
    """Generate all plots for a single ROOT file.

    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file.
    config_key : str
        Configuration key for naming.
    output_dir : Path
        Output directory for plots.
    tree_name : str
        Name of the ROOT tree.
    num_events : int
        Number of random events to display.
    min_tracks : int
        Minimum number of tracks required per event.
    generate_histograms : bool
        Whether to generate branch histograms.

    Returns:
    -------
    dict or None
        File info dictionary if successful, None otherwise.
    """
    print(f"\n{'=' * 80}")
    print(f"Processing: {config_key} -> {root_file_path}")
    print(f"{'=' * 80}")

    if not Path(root_file_path).exists():
        print(f"ERROR: File not found: {root_file_path}")
        return None

    try:
        analyzer = RootAnalyzer(root_file_path, tree_name)

        file_info = analyzer.get_file_info()
        print(f"File info: {file_info['num_entries']} entries")

        branches = file_info["branches"]
        if isinstance(branches, list):
            print(f"Available branches: {branches[:10]}...")
        else:
            print(f"Available branches: {branches}")

        file_output_dir = output_dir / sanitize_filename(config_key)
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

        # Generate branch histograms
        histogram_results = {}
        if generate_histograms:
            print("\n--- Generating branch histograms ---")
            histograms_output_dir = file_output_dir / "histograms"
            histogram_results = analyzer.generate_branch_histograms(output_dir=histograms_output_dir, histogram_settings=HISTOGRAM_SETTINGS)

            successful_histograms = sum(1 for s in histogram_results.values() if s)
            total_histograms = len(histogram_results)
            print(f"✓ Generated {successful_histograms}/{total_histograms} branch histograms")

            if successful_histograms == 0:
                print("WARNING: Failed to generate any branch histograms")
        else:
            print("\n--- Skipping histogram generation (disabled) ---")

        # Generate event displays
        event_displays_success = generate_event_displays(
            root_file_path,
            config_key,
            file_output_dir,
            tree_name,
            num_events=num_events,
            min_tracks=min_tracks,
        )

        if not event_displays_success:
            print("WARNING: Failed to generate event displays")

        # Save summary statistics
        if hits_data and tracks_data:
            _save_summary_statistics(
                file_output_dir,
                config_key,
                root_file_path,
                tree_name,
                file_info,
                hits_data,
                tracks_data,
                histogram_results,
                generate_histograms,
            )

        return file_info if success else None

    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"ERROR: Failed to process {config_key}: {e!s}")
        return None


def _save_summary_statistics(
    file_output_dir: Path,
    config_key: str,
    root_file_path: str,
    tree_name: str,
    file_info: dict,
    hits_data: dict,
    tracks_data: dict,
    histogram_results: dict,
    generate_histograms: bool,
) -> None:
    """Save summary statistics to a text file."""
    summary_path = file_output_dir / f"{config_key}_summary.txt"
    with summary_path.open("w") as f:
        f.write("ROOT File Analysis Summary\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"File: {root_file_path}\n")
        f.write(f"Config key: {config_key}\n")
        f.write(f"Tree name: {tree_name}\n")
        f.write(f"Analysis timestamp: {datetime.now().isoformat()}\n\n")

        f.write("File Statistics:\n")
        f.write(f"  Total entries: {file_info['num_entries']}\n")
        branches = file_info["branches"]
        if isinstance(branches, list):
            f.write(f"  Number of branches: {len(branches)}\n\n")
        else:
            f.write(f"  Number of branches: {branches}\n\n")

        f.write("Hits Analysis:\n")
        f.write(f"  Total events with hits: {len(hits_data)}\n")
        if hits_data:
            hit_counts = list(hits_data.values())
            f.write(f"  Mean hits per event: {sum(hit_counts) / len(hit_counts):.2f}\n")
            f.write(f"  Min hits per event: {min(hit_counts)}\n")
            f.write(f"  Max hits per event: {max(hit_counts)}\n\n")

        f.write("Tracks Analysis:\n")
        if tracks_data and "track_counts" in tracks_data:
            track_counts = tracks_data["track_counts"]
            f.write(f"  Total events with tracks: {len(track_counts)}\n")
            f.write(f"  Mean tracks per event: {sum(track_counts) / len(track_counts):.2f}\n")
            f.write(f"  Min tracks per event: {min(track_counts)}\n")
            f.write(f"  Max tracks per event: {max(track_counts)}\n")

        if tracks_data and "track_lengths" in tracks_data:
            track_lengths = tracks_data["track_lengths"]
            f.write(f"  Total track segments: {len(track_lengths)}\n")
            if track_lengths:
                f.write(f"  Mean track length: {sum(track_lengths) / len(track_lengths):.2f}\n")
                f.write(f"  Min track length: {min(track_lengths)}\n")
                f.write(f"  Max track length: {max(track_lengths)}\n\n")

        if generate_histograms and histogram_results:
            successful_histograms = sum(1 for s in histogram_results.values() if s)
            f.write("Branch Histograms:\n")
            f.write(f"  Total branches processed: {len(histogram_results)}\n")
            f.write(f"  Successfully generated: {successful_histograms}\n")
            f.write(f"  Failed: {len(histogram_results) - successful_histograms}\n")

            if histogram_results:
                f.write("  Histogram status:\n")
                for branch_name, success in histogram_results.items():
                    status = "✓" if success else "✗"
                    f.write(f"    {status} {branch_name}\n")
        elif not generate_histograms:
            f.write("Branch Histograms:\n")
            f.write("  Histogram generation was disabled\n")

    print(f"✓ Summary statistics saved to: {summary_path}")


def generate_event_displays(
    root_file_path: str,
    config_key: str,
    output_dir: Path,
    tree_name: str = DEFAULT_TREE_NAME,
    num_events: int = 10,
    min_tracks: int = 1,
) -> list[int] | None:
    """Generate event displays for randomly selected events that contain tracks.

    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file.
    config_key : str
        Configuration key for naming.
    output_dir : Path
        Output directory for plots.
    tree_name : str
        Name of the ROOT tree.
    num_events : int
        Number of random events to display.
    min_tracks : int
        Minimum number of tracks required per event.

    Returns:
    -------
    list or None
        List of successfully processed event indices, or None if failed.
    """
    print(f"\n--- Generating {num_events} random event displays (min {min_tracks} tracks) ---")

    try:
        events_with_tracks = find_events_with_tracks(root_file_path, tree_name, min_tracks=min_tracks)

        if not events_with_tracks:
            print("ERROR: No events with tracks found in the ROOT file")
            return None

        print(f"Found {len(events_with_tracks)} events with tracks")

        events_output_dir = output_dir / "events"
        events_output_dir.mkdir(exist_ok=True)

        selected_events = select_diverse_events(events_with_tracks, num_events)

        # True hits per event histogram
        visualizer = TrackVisualizerMDTGeometry(root_file_path, tree_name)
        visualizer.plot_and_save_true_hits_histogram(save_path=str(output_dir / f"{config_key}_true_hits_histogram.png"))

        successful_events = []

        for event_idx, num_tracks in selected_events:
            try:
                visualizer = TrackVisualizerMDTGeometry(root_file_path, tree_name)
                save_path = events_output_dir / f"{config_key}_eventidx_{event_idx}_{num_tracks}tracks.png"
                fig = visualizer.plot_and_save_event(event_idx, str(save_path))

                if fig is not None:
                    successful_events.append(event_idx)
                    print(f"✓ Event {event_idx} display saved (tracks: {num_tracks})")
                else:
                    print(f"✗ Failed to create display for event {event_idx}")

            except (ValueError, KeyError, OSError, RuntimeError) as e:
                print(f"✗ Error processing event {event_idx}: {e!s}")

        print(f"✓ Generated {len(successful_events)} event displays")
        return successful_events or None

    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"ERROR: Failed to generate event displays: {e!s}")
        return None


def find_events_with_tracks(root_file_path: str, tree_name: str = DEFAULT_TREE_NAME, min_tracks: int = 1) -> list[tuple[int, int]]:
    """Find events that contain tracks meeting the minimum requirement.

    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file.
    tree_name : str
        Name of the ROOT tree.
    min_tracks : int
        Minimum number of tracks required per event.

    Returns:
    -------
    list
        List of (event_idx, num_tracks) tuples for events with sufficient tracks.
    """
    print(f"Searching for events with at least {min_tracks} tracks...")

    try:
        analyzer = RootAnalyzer(root_file_path, tree_name)
        tracks_data = analyzer.analyze_tracks_and_lengths()

        if not tracks_data or "track_counts" not in tracks_data:
            print("ERROR: No track data found in the ROOT file")
            return []

        track_counts = tracks_data["track_counts"]
        events_with_tracks = [(event_idx, num_tracks) for event_idx, num_tracks in enumerate(track_counts) if num_tracks >= min_tracks]

        print(f"Found {len(events_with_tracks)} events with at least {min_tracks} tracks")
        return events_with_tracks

    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"ERROR: Failed to find events with tracks: {e!s}")
        return []


def select_diverse_events(events_with_tracks: list[tuple[int, int]], num_events: int) -> list[tuple[int, int]]:
    """Select events with diverse track counts for visualization variety.

    Parameters
    ----------
    events_with_tracks : list
        List of (event_idx, num_tracks) tuples.
    num_events : int
        Number of events to select.

    Returns:
    -------
    list
        Selected events with diverse track counts.
    """
    if len(events_with_tracks) <= num_events:
        print(f"WARNING: Only {len(events_with_tracks)} events with tracks available")
        return events_with_tracks

    # Group events by track count
    track_count_groups: dict[int, list[tuple[int, int]]] = {}
    for event_idx, num_tracks in events_with_tracks:
        if num_tracks not in track_count_groups:
            track_count_groups[num_tracks] = []
        track_count_groups[num_tracks].append((event_idx, num_tracks))

    print(f"Track count distribution: {[(count, len(events)) for count, events in track_count_groups.items()]}")

    selected_events: list[tuple[int, int]] = []
    rng = np.random.default_rng(42)

    # Sort track counts to prioritize higher track counts
    sorted_track_counts = sorted(track_count_groups.keys(), reverse=True)

    # Get at least one event from each track count
    for track_count in sorted_track_counts:
        if len(selected_events) >= num_events:
            break

        events_for_count = track_count_groups[track_count]
        selected_event = events_for_count[rng.integers(len(events_for_count))]
        selected_events.append(selected_event)

    # Fill remaining slots
    remaining_slots = num_events - len(selected_events)
    if remaining_slots > 0:
        remaining_events = [event for event in events_with_tracks if event not in selected_events]

        if remaining_events:
            additional_indices = rng.choice(
                len(remaining_events),
                size=min(remaining_slots, len(remaining_events)),
                replace=False,
            )
            additional_events = [remaining_events[i] for i in additional_indices]
            selected_events.extend(additional_events)

    return selected_events[:num_events]


def create_html_index(
    index_path: Path,
    results: dict[str, bool],
    output_dir: Path,
    num_events: int,
    file_paths: dict[str, str] | None = None,
) -> None:
    """Create HTML index file for the analysis results.

    Parameters
    ----------
    index_path : Path
        Path to save the HTML index file.
    results : dict
        Dictionary of config_key -> success status.
    output_dir : Path
        Output directory containing plots.
    num_events : int
        Number of events processed.
    file_paths : dict, optional
        Dictionary of config_key -> file path for display.
    """
    import glob

    if file_paths is None:
        file_paths = {}

    with index_path.open("w") as f:
        f.write(
            f"""<!DOCTYPE html>
<html>
<head>
    <title>ROOT Analysis Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .file-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; }}
        .success {{ background-color: #d4edda; }}
        .error {{ background-color: #f8d7da; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>ROOT File Analysis Results</h1>
    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

"""
        )

        for config_key, success in results.items():
            section_class = "success" if success else "error"
            f.write(f'    <div class="file-section {section_class}">\n')
            f.write(f"        <h2>{config_key}</h2>\n")
            f.write(f"        <p><strong>File:</strong> {file_paths.get(config_key, 'Unknown')}</p>\n")

            if success:
                f.write("        <p><strong>Status:</strong> ✓ Successfully processed</p>\n")

                file_subdir = sanitize_filename(config_key)
                hits_plot = f"{file_subdir}/{config_key}_hits_distribution.png"
                tracks_plot = f"{file_subdir}/{config_key}_tracks_analysis.png"
                summary_file = f"{file_subdir}/{config_key}_summary.txt"

                f.write("        <h3>Analysis Plots:</h3>\n")
                f.write("        <h4>Hits Distribution</h4>\n")
                f.write(f'        <img src="{hits_plot}" alt="Hits Distribution">\n')
                f.write("        <h4>Tracks Analysis</h4>\n")
                f.write(f'        <img src="{tracks_plot}" alt="Tracks Analysis">\n')

                # Event displays
                f.write("        <h3>Event Displays:</h3>\n")
                f.write(f"        <p>Random selection of up to {num_events} events showing track reconstructions:</p>\n")

                event_display_pattern = f"{output_dir}/{file_subdir}/events/*tracks.png"
                event_display_files = sorted(glob.glob(event_display_pattern))

                if event_display_files:
                    f.write('        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 15px;">\n')
                    for event_file in event_display_files:
                        rel_path = str(Path(event_file).relative_to(output_dir))
                        filename = Path(event_file).stem
                        parts = filename.split("_")
                        event_num = parts[-2] if len(parts) >= 2 else "unknown"
                        track_info = parts[-1] if len(parts) >= 1 else "unknown"
                        event_title = f"Event {event_num} ({track_info})"

                        f.write('            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">\n')
                        f.write(f'                <h5 style="margin-top: 0;">{event_title}</h5>\n')
                        f.write(
                            f'                <img src="{rel_path}" alt="{event_title}" style="width: 100%; height: auto; border: 1px solid #ccc;">\n'
                        )
                        f.write("            </div>\n")
                    f.write("        </div>\n")
                else:
                    f.write("        <p><em>No event displays generated</em></p>\n")

                # Branch histograms
                f.write("        <h3>Branch Histograms:</h3>\n")
                histogram_pattern = f"{output_dir}/{file_subdir}/histograms/*_histogram.png"
                histogram_files = sorted(glob.glob(histogram_pattern))

                if histogram_files:
                    f.write('        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 15px;">\n')
                    for hist_file in histogram_files:
                        rel_path = str(Path(hist_file).relative_to(output_dir))
                        filename = Path(hist_file).stem
                        branch_name = filename.replace("_histogram", "")

                        f.write('            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">\n')
                        f.write(f'                <h5 style="margin-top: 0;">{branch_name}</h5>\n')
                        f.write(
                            f'                <img src="{rel_path}" '
                            f'alt="{branch_name} Histogram" '
                            'style="width: 100%; height: auto; '
                            'border: 1px solid #ccc;">\n'
                        )
                        f.write("            </div>\n")
                    f.write("        </div>\n")
                else:
                    f.write("        <p><em>No histograms generated</em></p>\n")

                f.write(f'        <p><a href="{summary_file}">View detailed summary</a></p>\n')
            else:
                f.write("        <p><strong>Status:</strong> ✗ Processing failed</p>\n")

            f.write("    </div>\n")

        f.write("</body>\n</html>\n")

    print(f"HTML index created: {index_path}")


def main() -> None:
    """Main function to process ROOT files."""
    parser = argparse.ArgumentParser(description="Generate ROOT analysis plots for ROOT files")
    parser.add_argument(
        "--root-file",
        type=str,
        required=True,
        help="Path to the ROOT file to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./root_analysis_output",
        help="Base directory for output plots (default: ./root_analysis_output)",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default=DEFAULT_TREE_NAME,
        help=f"ROOT tree name to analyze (default: {DEFAULT_TREE_NAME})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for the analysis (default: derived from filename)",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=10,
        help="Number of random events to display (default: 10)",
    )
    parser.add_argument(
        "--min-tracks",
        type=int,
        default=1,
        help="Minimum number of tracks required per event (default: 1)",
    )
    parser.add_argument(
        "--skip-histograms",
        action="store_true",
        help="Skip generation of branch histograms",
    )

    args = parser.parse_args()

    # Validate input file exists
    root_file_path = Path(args.root_file)
    if not root_file_path.exists():
        print(f"ERROR: ROOT file not found: {root_file_path}")
        return

    output_dir = create_output_directory(args.output_dir)
    print(f"Output directory: {output_dir}")
    print(f"ROOT file: {root_file_path}")

    # Use provided name or derive from filename
    config_key = args.name or root_file_path.stem

    print(f"Processing ROOT file: {config_key}")

    file_info = generate_plots_for_file(
        str(root_file_path),
        config_key,
        output_dir,
        args.tree_name,
        args.num_events,
        args.min_tracks,
        not args.skip_histograms,
    )
    success = file_info is not None

    # Print summary
    print(f"\n{'=' * 80}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 80}")

    if success:
        print(f"✓ Successfully processed: {config_key}")
    else:
        print(f"✗ Failed to process: {config_key}")

    print(f"\nAll plots saved to: {output_dir}")

    # Create HTML index with single result
    results = {config_key: success}
    file_paths = {config_key: str(root_file_path)}
    index_path = output_dir / "index.html"
    create_html_index(index_path, results, output_dir, args.num_events, file_paths)


if __name__ == "__main__":
    main()
