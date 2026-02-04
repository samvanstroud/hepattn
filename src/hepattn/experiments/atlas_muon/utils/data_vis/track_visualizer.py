# ruff: noqa: N803,N806,RET504
"""Track visualization utilities for ROOT-based muon tracking data.

This module provides the TrackVisualizer class for visualizing ATLAS muon
spectrometer tracks as scatter plots in three 2D projections from ROOT files.
"""

import matplotlib.pyplot as plt
import numpy as np
import uproot


class TrackVisualizer:
    """Visualizer for ATLAS muon tracks from ROOT files using scatter plots.

    This class creates three 2D projection plots (X-Y, Z-Y, Z-X) showing
    track hits as scatter points.

    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file containing muon data.
    tree_name : str
        Name of the tree in the ROOT file (default: "MuonHitDump").
    """

    def __init__(self, root_file_path: str, tree_name: str = "MuonHitDump") -> None:
        self.root_file_path = root_file_path
        self.tree_name = tree_name

        # Define saturated colors for up to 8 tracks
        self.track_colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FFFF00",  # Yellow
            "#FF8000",  # Orange
            "#8000FF",  # Purple
        ]

    def plot_muon_tracks(self, event_index: int, show_plot: bool = True) -> plt.Figure | None:
        """Plot ATLAS Muon spectrometer tracks in three 2D projections.

        Parameters
        ----------
        event_index : int
            Event index to visualize (0-based).
        show_plot : bool, optional
            Whether to display the plot (default: True).

        Returns:
        -------
        matplotlib.figure.Figure or None
            The figure object containing the plots, or None on error.
        """
        try:
            with uproot.open(self.root_file_path) as file:
                tree = file[self.tree_name]

                if event_index < 0 or event_index >= tree.num_entries:
                    print(f"Event index {event_index} is out of range. Available entries: 0 to {tree.num_entries - 1}")
                    return None

                # Read event data
                event_number = tree["eventNumber"].array(library="np")[event_index]
                truth_links = tree["spacePoint_truthLink"].array(library="np")[event_index]
                pos_x = tree["spacePoint_globPosX"].array(library="np")[event_index]
                pos_y = tree["spacePoint_globPosY"].array(library="np")[event_index]
                pos_z = tree["spacePoint_globPosZ"].array(library="np")[event_index]
                truthMuon_phi = tree["truthMuon_phi"].array(library="np")[event_index]
                truthMuon_eta = tree["truthMuon_eta"].array(library="np")[event_index]

                all_x, all_y, all_z, all_truth = self._prepare_arrays(pos_x, pos_y, pos_z, truth_links)

                print(f"Event {event_number} (index {event_index}): Found {len(all_x)} space points")
                print(f"X range: [{np.min(all_x):.1f}, {np.max(all_x):.1f}] mm")
                print(f"Y range: [{np.min(all_y):.1f}, {np.max(all_y):.1f}] mm")
                print(f"Z range: [{np.min(all_z):.1f}, {np.max(all_z):.1f}] mm")

                fig = self._create_track_plots(
                    all_x,
                    all_y,
                    all_z,
                    all_truth,
                    truthMuon_phi,
                    truthMuon_eta,
                    event_number,
                    show_plot,
                )

                return fig

        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error plotting tracks: {e}")
            return None

    def _prepare_arrays(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        truth_links: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and align arrays for plotting."""
        all_x = np.array(pos_x)
        all_y = np.array(pos_y)
        all_z = np.array(pos_z)
        all_truth = np.array(truth_links)

        # Handle array length mismatch by taking minimum length
        min_length = min(len(all_x), len(all_y), len(all_z), len(all_truth))
        all_x = all_x[:min_length]
        all_y = all_y[:min_length]
        all_z = all_z[:min_length]
        all_truth = all_truth[:min_length]

        return all_x, all_y, all_z, all_truth

    def _create_track_plots(
        self,
        all_x: np.ndarray,
        all_y: np.ndarray,
        all_z: np.ndarray,
        all_truth: np.ndarray,
        truthMuon_phi: np.ndarray,
        truthMuon_eta: np.ndarray,
        event_number: int,
        show_plot: bool = True,
    ) -> plt.Figure:
        """Create the three 2D projection plots."""
        # Separate background and track hits
        background_mask = all_truth == -1
        track_mask = all_truth >= 0

        print(f"Background hits: {np.sum(background_mask)}")
        print(f"Track hits: {np.sum(track_mask)}")
        print(f"All truth values: {np.unique(all_truth)}")

        # Create figure with pyramid layout
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            2,
            4,
            height_ratios=[1, 1],
            width_ratios=[1, 1, 1, 1],
            hspace=0.25,
            wspace=0.35,
        )

        # X-Y plot centered on top row
        ax_xy = fig.add_subplot(gs[0, 1:3])
        # Z-Y plot on bottom left
        ax_zy = fig.add_subplot(gs[1, 0:2])
        # Z-X plot on bottom right
        ax_zx = fig.add_subplot(gs[1, 2:4])

        axes = [ax_xy, ax_zy, ax_zx]

        # Plot background hits
        if np.sum(background_mask) > 0:
            self._plot_background_hits(axes, all_x, all_y, all_z, background_mask)

        # Plot track hits
        if np.sum(track_mask) > 0:
            self._plot_track_hits(
                axes,
                all_x,
                all_y,
                all_z,
                all_truth,
                truthMuon_phi,
                truthMuon_eta,
                track_mask,
            )

        self._format_subplots(axes, event_number)
        self._create_single_legend(fig, axes, all_truth, track_mask, background_mask)

        plt.tight_layout()

        if show_plot:
            plt.show()

        return fig

    def _plot_background_hits(
        self,
        axes: list[plt.Axes],
        all_x: np.ndarray,
        all_y: np.ndarray,
        all_z: np.ndarray,
        background_mask: np.ndarray,
    ) -> None:
        """Plot background hits on all three projections."""
        axes[0].scatter(all_x[background_mask], all_y[background_mask], c="gray", alpha=0.3, s=10)
        axes[1].scatter(all_z[background_mask], all_y[background_mask], c="gray", alpha=0.3, s=10)
        axes[2].scatter(all_z[background_mask], all_x[background_mask], c="gray", alpha=0.3, s=10)

    def _plot_track_hits(
        self,
        axes: list[plt.Axes],
        all_x: np.ndarray,
        all_y: np.ndarray,
        all_z: np.ndarray,
        all_truth: np.ndarray,
        truthMuon_phi: np.ndarray,
        truthMuon_eta: np.ndarray,
        track_mask: np.ndarray,
        line_length: float = 14000,
        line_length_z: float = 25000,
    ) -> None:
        """Plot track hits with different colors for each track."""
        unique_tracks = np.unique(all_truth[track_mask])
        print(f"Unique track IDs: {unique_tracks}")

        for i, track_id in enumerate(unique_tracks):
            track_color = self.track_colors[i % len(self.track_colors)]
            track_points = all_truth == track_id

            num_hits = np.sum(track_points)
            print(f"  Track {track_id}: {num_hits} hits, color: {track_color}")

            # Plot hits as scatter points
            axes[0].scatter(
                all_x[track_points],
                all_y[track_points],
                c=track_color,
                alpha=0.9,
                s=30,
                marker="x",
                linewidth=2,
            )

            # Plot truth direction line in X-Y plane
            phi = truthMuon_phi[i]
            x1 = line_length * np.cos(phi)
            y1 = line_length * np.sin(phi)
            axes[0].plot([0, x1], [0, y1], color=track_color, linewidth=1, alpha=0.5)

            # Calculate Z projections
            eta = truthMuon_eta[i]
            theta = 2 * np.arctan(np.exp(-eta))
            z1 = line_length_z * np.cos(theta)
            x1_zplane = line_length * np.sin(theta) * np.cos(phi)
            y1_zplane = line_length * np.sin(theta) * np.sin(phi)

            # Z-Y plot
            axes[1].scatter(
                all_z[track_points],
                all_y[track_points],
                c=track_color,
                alpha=0.9,
                s=30,
                marker="x",
                linewidth=2,
            )
            axes[1].plot([0, z1], [0, y1_zplane], color=track_color, linewidth=1, alpha=0.5)

            # Z-X plot
            axes[2].scatter(
                all_z[track_points],
                all_x[track_points],
                c=track_color,
                alpha=0.9,
                s=30,
                marker="x",
                linewidth=2,
            )
            axes[2].plot([0, z1], [0, x1_zplane], color=track_color, linewidth=1, alpha=0.5)

        self._print_track_stats(unique_tracks, all_truth)

    def _print_track_stats(self, unique_tracks: np.ndarray, all_truth: np.ndarray) -> None:
        """Print statistics about the tracks."""
        print("\nTrack statistics:")
        print(f"Number of unique tracks: {len(unique_tracks)}")
        for i, track_id in enumerate(unique_tracks):
            track_hits = np.sum(all_truth == track_id)
            color_used = self.track_colors[i % len(self.track_colors)]
            print(f"  Track {track_id}: {track_hits} hits (color: {color_used})")

        if len(unique_tracks) > len(self.track_colors):
            print(f"  Warning: {len(unique_tracks)} tracks found, but only {len(self.track_colors)} colors available.")

    def _format_subplots(self, axes: list[plt.Axes], event_number: int) -> None:
        """Format the three subplots with proper labels and styling."""
        # X-Y plane (top)
        axes[0].set_xlabel("X [mm]")
        axes[0].set_ylabel("Y [mm]")
        axes[0].set_title(f"X-Y Plane (Event {event_number})")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim([-14000, 14000])
        axes[0].set_ylim([-14000, 14000])

        # Z-Y plane (bottom left)
        axes[1].set_xlabel("Z [mm]")
        axes[1].set_ylabel("Y [mm]")
        axes[1].set_title(f"Z-Y Plane (Event {event_number})")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([-25000, 25000])
        axes[1].set_ylim([-14000, 14000])

        # Z-X plane (bottom right)
        axes[2].set_xlabel("Z [mm]")
        axes[2].set_ylabel("X [mm]")
        axes[2].set_title(f"Z-X Plane (Event {event_number})")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([-25000, 25000])
        axes[2].set_ylim([-14000, 14000])

    def _create_single_legend(
        self,
        fig: plt.Figure,
        axes: list[plt.Axes],
        all_truth: np.ndarray,
        track_mask: np.ndarray,
        background_mask: np.ndarray,
    ) -> None:
        """Create a single legend positioned outside the plots."""
        legend_elements = []

        if np.sum(background_mask) > 0:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    markersize=8,
                    alpha=0.6,
                    label="Background",
                )
            )

        if np.sum(track_mask) > 0:
            unique_tracks = np.unique(all_truth[track_mask])
            for i, track_id in enumerate(unique_tracks):
                track_color = self.track_colors[i % len(self.track_colors)]
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="x",
                        color="w",
                        markerfacecolor=track_color,
                        markeredgecolor=track_color,
                        markersize=10,
                        markeredgewidth=2,
                        label=f"Track {track_id}",
                    )
                )

        fig.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(0.85, 0.92),
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08)

    def plot_and_save_event(self, event_index: int, save_path: str | None = None) -> plt.Figure | None:
        """Plot tracks for a specific event and optionally save the figure.

        Parameters
        ----------
        event_index : int
            Event index to plot (0-based).
        save_path : str, optional
            Path to save the figure.

        Returns:
        -------
        matplotlib.figure.Figure or None
            The figure object.
        """
        fig = self.plot_muon_tracks(event_index)

        if fig is not None and save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        return fig

    def plot_and_save_true_hits_histogram(
        self,
        max_events: int | None = None,
        num_bins: int = 100,
        show_plot: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure | None:
        """Create a histogram showing the distribution of true hits per event.

        Parameters
        ----------
        max_events : int, optional
            Maximum number of events to process.
        num_bins : int, optional
            Number of bins for the histogram (default: 100).
        show_plot : bool, optional
            Whether to display the plot (default: True).
        save_path : str, optional
            Path to save the figure.

        Returns:
        -------
        matplotlib.figure.Figure or None
            The figure object containing the histogram.
        """
        try:
            with uproot.open(self.root_file_path) as file:
                tree = file[self.tree_name]

                total_events = tree.num_entries
                num_events = min(total_events, max_events) if max_events else total_events

                truth_links = tree["spacePoint_truthLink"].array(library="np", entry_stop=num_events)

                # Count true hits per event (non -1 values)
                true_hits_counts: list[int] = [np.sum(event_links != -1) for event_links in truth_links]
                n_0_hits: int = np.sum(np.array(true_hits_counts) == 0)
                true_hits_counts = [count for count in true_hits_counts if count > 0]

                fig, ax = plt.subplots(figsize=(10, 6))
                bins = np.arange(
                    np.min(true_hits_counts) - 0.5,
                    max(true_hits_counts) + 1.5,
                    1,
                )

                ax.hist(
                    true_hits_counts,
                    bins=bins,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                )

                ax.set_xlabel("Number of True Hits per Event")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of True Hits per Event\n(Analysis of {num_events} events)")
                ax.grid(alpha=0.3)

                mean_hits = float(np.mean(true_hits_counts))
                median_hits = float(np.median(true_hits_counts))
                max_hits = np.max(true_hits_counts)
                min_hits = np.min(true_hits_counts)

                stats_text = (
                    f"Statistics:\n"
                    f"Mean: {mean_hits:.1f}\n"
                    f"Median: {median_hits:.1f}\n"
                    f"Min: {min_hits}\n"
                    f"Max: {max_hits}\n"
                    f"Total events: {num_events}\n"
                    f"Events with 0 hits: {n_0_hits} ({n_0_hits / num_events * 100:.1f}%)"
                )

                ax.text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )

                plt.tight_layout()

                if save_path:
                    fig.savefig(save_path, dpi=300, bbox_inches="tight")
                    print(f"Histogram saved to: {save_path}")

                if show_plot:
                    plt.show()

                return fig

        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error plotting true hits histogram: {e}")
            return None
