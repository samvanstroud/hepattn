# ruff: noqa: TRY300,RUF059,SIM118,DOC501

"""ROOT file analysis utilities for muon tracking data.

This module provides the RootAnalyzer class for analyzing ROOT files
containing ATLAS muon tracking data, including hit distributions,
track analysis, and branch histogram generation.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

from .config import HISTOGRAM_SETTINGS


class RootAnalyzer:
    """Analyzer for ROOT files containing ATLAS muon tracking data.

    Provides methods for analyzing hit distributions, track properties,
    and generating HEP-style histograms.

    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file.
    tree_name : str
        Name of the tree in the ROOT file (default: "MuonHitDump").
    """

    def __init__(self, root_file_path: str, tree_name: str = "MuonHitDump") -> None:
        self.root_file_path = root_file_path
        self.tree_name = tree_name
        self._tree = None
        self._file_info: dict[str, list[str] | int] | None = None

    def _load_tree(self) -> None:
        """Load the ROOT tree if not already loaded."""
        if self._tree is None:
            with uproot.open(self.root_file_path) as file:
                if self.tree_name not in file:
                    available_trees = [key for key in file.keys() if ";" in key]
                    raise ValueError(f"Tree '{self.tree_name}' not found. Available: {available_trees}")
                self._tree = file[self.tree_name]

    def get_file_info(self) -> dict[str, list[str] | int]:
        """Get information about the ROOT file structure.

        Returns:
        -------
        dict
            Dictionary with 'keys', 'num_entries', and 'branches'.
        """
        if self._file_info is None:
            with uproot.open(self.root_file_path) as file:
                self._file_info = {
                    "keys": list(file.keys()),
                    "num_entries": file[self.tree_name].num_entries if self.tree_name in file else 0,
                    "branches": list(file[self.tree_name].keys()) if self.tree_name in file else [],
                }
        return self._file_info

    def analyze_hits_per_event(self, output_plot_path: str | None = None) -> dict[int, int] | None:
        """Analyze the distribution of hits per event.

        Parameters
        ----------
        output_plot_path : str, optional
            Path to save the plot.

        Returns:
        -------
        dict or None
            Dictionary with event numbers as keys and hit counts as values.
        """
        try:
            with uproot.open(self.root_file_path) as file:
                tree = file[self.tree_name]

                event_numbers = tree["eventNumber"].array(library="np")
                space_points_x = tree["spacePoint_PositionX"].array(library="np")

                print(f"Found {len(event_numbers)} entries")
                print(f"Event number range: {np.min(event_numbers)} to {np.max(event_numbers)}")

                hits_per_event = {}

                if hasattr(space_points_x, "__len__") and len(space_points_x) > 0:
                    if hasattr(space_points_x[0], "__len__"):
                        # Jagged array case
                        for event_num, x_positions in zip(event_numbers, space_points_x, strict=False):
                            if event_num not in hits_per_event:
                                hits_per_event[event_num] = 0
                            hits_per_event[event_num] += len(x_positions)
                    else:
                        # Flat array case
                        hit_counter = Counter(event_numbers)
                        hits_per_event = dict(hit_counter)

                self._plot_hits_distribution(hits_per_event, output_plot_path)

                return hits_per_event

        except (OSError, ValueError, KeyError) as e:
            print(f"Error analyzing hits per event: {e}")
            return None

    def _plot_hits_distribution(self, hits_per_event: dict[int, int], output_plot_path: str | None = None) -> None:
        """Create histogram of hits per event distribution."""
        hit_counts = list(hits_per_event.values())

        plt.figure(figsize=(10, 6))
        plt.hist(hit_counts, bins=100, alpha=0.7, edgecolor="black")
        plt.xlabel("Number of Hits per Event")
        plt.ylabel("Number of Events")
        plt.title("Distribution of Hits per Event")
        plt.grid(True, alpha=0.3)

        mean_hits = np.mean(hit_counts)
        std_hits = np.std(hit_counts)
        plt.axvline(
            mean_hits,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_hits:.1f}",
        )
        plt.axvline(
            mean_hits + std_hits,
            color="orange",
            linestyle="--",
            linewidth=1,
            label=f"Mean + sigma: {mean_hits + std_hits:.1f}",
        )
        plt.axvline(
            mean_hits - std_hits,
            color="orange",
            linestyle="--",
            linewidth=1,
            label=f"Mean - sigma: {mean_hits - std_hits:.1f}",
        )
        plt.legend()

        print("\nHits per Event Statistics:")
        print(f"Total events: {len(hits_per_event)}")
        print(f"Mean hits per event: {mean_hits:.2f}")
        print(f"Standard deviation: {std_hits:.2f}")
        print(f"Min hits per event: {np.min(hit_counts)}")
        print(f"Max hits per event: {np.max(hit_counts)}")

        if output_plot_path:
            plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {output_plot_path}")

        plt.show()

    def analyze_tracks_and_lengths(self, output_plot_path: str | None = None) -> dict[str, dict[int, int] | list[int]] | None:
        """Analyze tracks per event and track lengths.

        Parameters
        ----------
        output_plot_path : str, optional
            Path to save the plot.

        Returns:
        -------
        dict or None
            Dictionary containing analysis results.
        """
        try:
            with uproot.open(self.root_file_path) as file:
                tree = file[self.tree_name]

                event_numbers = tree["eventNumber"].array(library="np")
                truth_links = tree["spacePoint_truthLink"].array(library="np")

                print(f"Found {len(event_numbers)} entries")

                # Analyze tracks per event
                tracks_per_event: dict[int, set] = {}
                for event_num, links in zip(event_numbers, truth_links, strict=False):
                    if event_num not in tracks_per_event:
                        tracks_per_event[event_num] = set()

                    if hasattr(links, "__len__") and len(links) > 0:
                        valid_links = links[(links >= 0) & (links < 1e6)]
                        unique_track_ids = np.unique(valid_links)
                        for track_id in unique_track_ids:
                            tracks_per_event[event_num].add(track_id)

                tracks_per_event_counts = {event: len(tracks) for event, tracks in tracks_per_event.items()}

                # Analyze track lengths
                all_event_track_lengths = []
                for links in truth_links:
                    if hasattr(links, "__len__") and len(links) > 0:
                        valid_links = links[(links >= 0) & (links < 1e6)]
                        if len(valid_links) > 0:
                            unique_tracks, track_hit_counts = np.unique(valid_links, return_counts=True)
                            all_event_track_lengths.extend(track_hit_counts)

                self._plot_tracks_analysis(tracks_per_event_counts, all_event_track_lengths, output_plot_path)

                return {
                    "tracks_per_event": tracks_per_event_counts,
                    "track_lengths": all_event_track_lengths,
                    "track_counts": list(tracks_per_event_counts.values()),
                }

        except (OSError, ValueError, KeyError) as e:
            print(f"Error in tracks analysis: {e}")
            return None

    def _plot_tracks_analysis(
        self,
        tracks_per_event_counts: dict[int, int],
        all_event_track_lengths: list[int],
        output_plot_path: str | None = None,
    ) -> None:
        """Create side-by-side plots for tracks analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        track_counts = list(tracks_per_event_counts.values())

        # Create bins aligned with integer values
        if len(track_counts) > 0:
            min_tracks = min(track_counts)
            max_tracks = max(track_counts)
            bins_tracks = np.arange(min_tracks - 0.5, max_tracks + 1.5, 1)
        else:
            bins_tracks = np.arange(-0.5, 1.5, 1)

        ax1.hist(
            track_counts,
            bins=bins_tracks,
            alpha=0.7,
            edgecolor="black",
            color="lightgreen",
        )
        ax1.set_xlabel("Number of Tracks per Event")
        ax1.set_ylabel("Number of Events")
        ax1.set_title("Distribution of Tracks per Event")
        ax1.grid(True, alpha=0.3)

        if len(track_counts) > 0:
            ax1.set_xticks(range(min(track_counts), max(track_counts) + 1))

        mean_tracks = np.mean(track_counts)
        ax1.axvline(
            mean_tracks,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_tracks:.1f}",
        )
        ax1.legend()

        # Plot track lengths
        if len(all_event_track_lengths) > 0:
            min_length = min(all_event_track_lengths)
            max_length = max(all_event_track_lengths)
            bins_lengths = np.arange(min_length - 0.5, max_length + 1.5, 1)

            ax2.hist(
                all_event_track_lengths,
                bins=bins_lengths,
                alpha=0.7,
                edgecolor="black",
                color="orange",
            )
            ax2.set_xlabel("Track Length (Hits per Track)")
            ax2.set_ylabel("Number of Track Segments")
            ax2.set_title("Distribution of Track Lengths\n(Within Individual Events)")
            ax2.grid(True, alpha=0.3)

            if max_length - min_length <= 20:
                ax2.set_xticks(range(min_length, max_length + 1))
            else:
                ax2.set_xticks(
                    range(
                        min_length,
                        max_length + 1,
                        max(1, (max_length - min_length) // 10),
                    )
                )

            mean_length = np.mean(all_event_track_lengths)
            ax2.axvline(
                mean_length,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_length:.1f}",
            )
            ax2.legend()

        plt.tight_layout()

        if output_plot_path:
            plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {output_plot_path}")

        plt.show()

    def generate_branch_histograms(
        self,
        output_dir: str | Path,
        histogram_settings: dict | None = None,
    ) -> dict[str, bool]:
        """Generate HEP ROOT style histograms for all branches.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save histogram plots.
        histogram_settings : dict, optional
            Dictionary defining histogram settings. Uses HISTOGRAM_SETTINGS if None.

        Returns:
        -------
        dict
            Dictionary with branch names as keys and success status as values.
        """
        if histogram_settings is None:
            histogram_settings = HISTOGRAM_SETTINGS

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        try:
            with uproot.open(self.root_file_path) as file:
                tree = file[self.tree_name]
                available_branches = list(tree.keys())

                print(f"Available branches: {len(available_branches)}")
                print(f"Branches to plot: {len(histogram_settings)}")

                for branch_name, settings in histogram_settings.items():
                    try:
                        if branch_name not in available_branches:
                            print(f"WARNING: Branch '{branch_name}' not found in tree")
                            results[branch_name] = False
                            continue

                        print(f"Processing branch: {branch_name}")
                        branch_data = tree[branch_name].array(library="np")

                        # Flatten if jagged array
                        if hasattr(branch_data, "tolist") and any(
                            hasattr(item, "__len__") and not isinstance(item, str) for item in branch_data[:10]
                        ):
                            flattened_data = []
                            for item in branch_data:
                                if hasattr(item, "__len__") and not isinstance(item, str):
                                    flattened_data.extend(item)
                                else:
                                    flattened_data.append(item)
                            data_to_plot = np.array(flattened_data)
                        else:
                            data_to_plot = np.array(branch_data)

                        # Remove invalid values
                        data_to_plot = data_to_plot[np.isfinite(data_to_plot)]

                        if len(data_to_plot) == 0:
                            print(f"WARNING: No valid data for branch '{branch_name}'")
                            results[branch_name] = False
                            continue

                        success = self._create_hep_style_histogram(data_to_plot, branch_name, settings, output_path)
                        results[branch_name] = success

                    except (OSError, ValueError, KeyError) as e:
                        print(f"ERROR processing branch '{branch_name}': {e!s}")
                        results[branch_name] = False

        except (OSError, ValueError, KeyError) as e:
            print(f"ERROR reading ROOT file: {e!s}")
            return dict.fromkeys(histogram_settings, False)

        return results

    def _create_hep_style_histogram(
        self,
        data: np.ndarray,
        branch_name: str,
        settings: dict,
        output_dir: Path,
    ) -> bool:
        """Create a HEP ROOT style histogram for a given branch.

        Parameters
        ----------
        data : np.ndarray
            Data to plot.
        branch_name : str
            Name of the branch.
        settings : dict
            Dictionary with 'bins' and 'range' settings.
        output_dir : Path
            Directory to save the plot.

        Returns:
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            plt.style.use("default")

            fig, ax = plt.subplots(figsize=(10, 8))

            bins = settings.get("bins", 50)
            data_range = settings.get("range", (np.min(data), np.max(data)))

            # Apply range filter
            mask = (data >= data_range[0]) & (data <= data_range[1])
            filtered_data = data[mask]

            # Calculate exclusion counts
            total_entries = len(data)
            entries_in_range = len(filtered_data)
            excluded_below = len(data[data < data_range[0]])
            excluded_above = len(data[data > data_range[1]])
            excluded_total = excluded_below + excluded_above

            if len(filtered_data) == 0:
                print(f"WARNING: No data in range {data_range} for branch '{branch_name}'")
                return False

            # Determine bins
            if bins is None:
                min_val = int(np.floor(np.min(filtered_data)))
                max_val = int(np.ceil(np.max(filtered_data)))
                bins_array = np.arange(min_val - 0.5, max_val + 1.5, 1)
            else:
                bins_array = np.linspace(data_range[0], data_range[1], bins + 1)

            # Create histogram with HEP ROOT style
            n, bins_edges, patches = ax.hist(
                filtered_data,
                bins=bins_array,
                histtype="step",
                linewidth=0.5,
                color="black",
                alpha=0.8,
            )

            # Fill histogram
            ax.hist(
                filtered_data,
                bins=bins_array,
                alpha=0.3,
                color="lightblue",
                edgecolor="black",
            )

            # Formatting
            ax.set_xlabel(branch_name, fontsize=14, fontweight="bold")
            ax.set_ylabel("Entries", fontsize=14, fontweight="bold")
            ax.set_title(
                f"Distribution of {branch_name}",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )

            # Statistics box
            entries = len(filtered_data)
            mean = np.mean(filtered_data)
            std = np.std(filtered_data)

            stats_text = f"Entries: {entries}\nMean: {mean:.3g}\nStd: {std:.3g}"
            stats_text += f"\n\nExcluded: {excluded_total}"
            if excluded_below > 0:
                stats_text += f"\n  < {data_range[0]:.3g}: {excluded_below}"
            if excluded_above > 0:
                stats_text += f"\n  > {data_range[1]:.3g}: {excluded_above}"

            stats_y_pos = 0.98 if excluded_total == 0 else 0.95
            ax.text(
                0.72,
                stats_y_pos,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "pad": 0.5,
                },
                fontsize=11,
                fontfamily="monospace",
            )

            ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

            # Log scale for Y-axis
            log_y = settings.get("log_y", True)
            if log_y:
                ax.set_yscale("log")
                if len(n[n > 0]) > 0:
                    ax.set_ylim(bottom=max(0.1, np.min(n[n > 0]) * 0.5))
                else:
                    ax.set_ylim(bottom=0.1)

            ax.tick_params(labelsize=12)
            ax.tick_params(direction="in", length=6, width=1)
            ax.minorticks_on()
            ax.tick_params(which="minor", direction="in", length=3, width=0.5)

            plt.tight_layout()

            output_file = output_dir / f"{branch_name}_histogram.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            # Print summary
            if excluded_total > 0:
                exclusion_pct = (excluded_total / total_entries) * 100
                print(f"✓ Histogram saved: {output_file}")
                print(f"  Entries: {entries_in_range}/{total_entries} ({exclusion_pct:.1f}% excluded)")
            else:
                print(f"✓ Histogram saved: {output_file}")
                print(f"  Entries: {entries_in_range} (no exclusions)")

            return True

        except (OSError, ValueError, KeyError) as e:
            print(f"ERROR creating histogram for '{branch_name}': {e!s}")
            return False
