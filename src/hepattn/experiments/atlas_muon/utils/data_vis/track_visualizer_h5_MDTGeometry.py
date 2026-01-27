# ruff: noqa: N801,N803,N806,TRY300,SIM108,RET504,ICN001,TID252,PLC2801

"""Track visualization utilities for HDF5-based muon tracking data.

This module provides the h5TrackVisualizerMDTGeometry class for visualizing
ATLAS muon spectrometer tracks in three 2D projections using MDT geometry
representation (wire endpoints).
"""

import matplotlib
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..data import AtlasMuonDataset

# Plot styling constants
BACKGROUND_LINEWIDTH = 0.3
TRACK_LINEWIDTH = 0.5


class h5TrackVisualizerMDTGeometry:
    """Visualizer for ATLAS muon tracks from HDF5 datasets using MDT geometry.

    This class creates three 2D projection plots (X-Y, Z-Y, Z-X) showing
    track hits as line segments representing MDT drift tubes.

    Parameters
    ----------
    dataset : AtlasMuonDataset
        The dataset containing muon tracking events.
    """

    def __init__(self, dataset: AtlasMuonDataset) -> None:
        self.dataset = dataset

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

    def plot_muon_tracks(self, event_index: int, show_plot: bool = True, just_background: bool = False) -> plt.Figure | None:
        """Plot ATLAS Muon spectrometer tracks in three 2D projections.

        Parameters
        ----------
        event_index : int
            Event index to visualize (0-based).
        show_plot : bool, optional
            Whether to display the plot (default: True).
        just_background : bool, optional
            If True, only plot background hits (default: False).

        Returns:
        -------
        matplotlib.figure.Figure or None
            The figure object containing the plots, or None on error.
        """
        if event_index < 0 or event_index >= len(self.dataset):
            print(f"Event index {event_index} is out of range. Available entries: 0 to {len(self.dataset) - 1}")
            return None

        # Load event data
        inputs, targets = self.dataset.__getitem__(event_index)

        num_particles = np.sum(targets["particle_valid"].numpy())

        # Extract hit positions (convert back to mm for display)
        all_high_x = inputs["hit_spacePoint_globEdgeHighX"][0].numpy() * 1000
        all_high_y = inputs["hit_spacePoint_globEdgeHighY"][0].numpy() * 1000
        all_high_z = inputs["hit_spacePoint_globEdgeHighZ"][0].numpy() * 1000
        all_low_x = inputs["hit_spacePoint_globEdgeLowX"][0].numpy() * 1000
        all_low_y = inputs["hit_spacePoint_globEdgeLowY"][0].numpy() * 1000
        all_low_z = inputs["hit_spacePoint_globEdgeLowZ"][0].numpy() * 1000

        # Build truth assignment array
        truth_links = targets["particle_hit_valid"][0][:num_particles, :].numpy()
        all_truth = np.full(len(all_high_x), -1, dtype=int)

        for track_id, truth_link in enumerate(truth_links):
            indices = np.where(truth_link)[0]
            all_truth[indices] = track_id

        # Get truth muon parameters
        truthMuon_phi = targets["particle_truthMuon_phi"][:num_particles].numpy()
        truthMuon_eta = targets["particle_truthMuon_eta"][:num_particles].numpy()

        print(f"Event {event_index}: Found {len(all_high_x)} space points")
        print(f"X range: [{np.min(all_high_x):.1f}, {np.max(all_high_x):.1f}] mm")
        print(f"Y range: [{np.min(all_high_y):.1f}, {np.max(all_high_y):.1f}] mm")
        print(f"Z range: [{np.min(all_high_z):.1f}, {np.max(all_high_z):.1f}] mm")

        fig = self._create_track_plots(
            all_high_x,
            all_high_y,
            all_high_z,
            all_low_x,
            all_low_y,
            all_low_z,
            all_truth,
            truthMuon_phi,
            truthMuon_eta,
            event_index,
            show_plot,
            just_background=just_background,
        )

        return fig

    def _create_track_plots(
        self,
        all_high_x: np.ndarray,
        all_high_y: np.ndarray,
        all_high_z: np.ndarray,
        all_low_x: np.ndarray,
        all_low_y: np.ndarray,
        all_low_z: np.ndarray,
        all_truth: np.ndarray,
        truthMuon_phi: np.ndarray,
        truthMuon_eta: np.ndarray,
        event_number: int,
        show_plot: bool = True,
        just_background: bool = False,
    ) -> plt.Figure:
        """Create the three 2D projection plots."""
        # Separate background and track hits
        if just_background:
            background_mask = np.ones(len(all_truth), dtype=bool)
        else:
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
            self._plot_background_hits(
                axes,
                all_high_x,
                all_high_y,
                all_high_z,
                all_low_x,
                all_low_y,
                all_low_z,
                background_mask,
            )

        # Plot track hits
        if np.sum(track_mask) > 0 and not just_background:
            self._plot_track_hits(
                axes,
                all_high_x,
                all_high_y,
                all_high_z,
                all_low_x,
                all_low_y,
                all_low_z,
                all_truth,
                truthMuon_phi,
                truthMuon_eta,
                track_mask,
            )

        self._format_subplots(axes, event_number)

        if not just_background:
            self._create_single_legend(fig, axes, all_truth, track_mask, background_mask)

        plt.tight_layout()
        return fig

    def _plot_background_hits(
        self,
        axes: list[plt.Axes],
        all_high_x: np.ndarray,
        all_high_y: np.ndarray,
        all_high_z: np.ndarray,
        all_low_x: np.ndarray,
        all_low_y: np.ndarray,
        all_low_z: np.ndarray,
        background_mask: np.ndarray,
    ) -> None:
        """Plot background hits on all three projections."""
        for low_x, low_y, low_z, high_x, high_y, high_z in zip(
            all_low_x[background_mask],
            all_low_y[background_mask],
            all_low_z[background_mask],
            all_high_x[background_mask],
            all_high_y[background_mask],
            all_high_z[background_mask],
            strict=False,
        ):
            axes[0].plot(
                [low_x, high_x],
                [low_y, high_y],
                color="gray",
                alpha=0.3,
                linewidth=BACKGROUND_LINEWIDTH,
            )
            axes[1].plot(
                [low_z, high_z],
                [low_y, high_y],
                color="gray",
                alpha=0.3,
                linewidth=BACKGROUND_LINEWIDTH,
            )
            axes[2].plot(
                [low_z, high_z],
                [low_x, high_x],
                color="gray",
                alpha=0.3,
                linewidth=BACKGROUND_LINEWIDTH,
            )

    def _plot_track_hits(
        self,
        axes: list[plt.Axes],
        all_high_x: np.ndarray,
        all_high_y: np.ndarray,
        all_high_z: np.ndarray,
        all_low_x: np.ndarray,
        all_low_y: np.ndarray,
        all_low_z: np.ndarray,
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

            # Plot hits as line segments
            for x_low, y_low, z_low, x_high, y_high, z_high in zip(
                all_low_x[track_points],
                all_low_y[track_points],
                all_low_z[track_points],
                all_high_x[track_points],
                all_high_y[track_points],
                all_high_z[track_points],
                strict=False,
            ):
                axes[0].plot(
                    [x_low, x_high],
                    [y_low, y_high],
                    color=track_color,
                    alpha=0.9,
                    linewidth=TRACK_LINEWIDTH,
                )
                axes[1].plot(
                    [z_low, z_high],
                    [y_low, y_high],
                    color=track_color,
                    alpha=0.9,
                    linewidth=TRACK_LINEWIDTH,
                )
                axes[2].plot(
                    [z_low, z_high],
                    [x_low, x_high],
                    color=track_color,
                    alpha=0.9,
                    linewidth=TRACK_LINEWIDTH,
                )

            # Plot truth direction lines
            if i < len(truthMuon_phi):
                phi = truthMuon_phi[i]
                eta = truthMuon_eta[i]
                theta = 2 * np.arctan(np.exp(-eta))

                x1 = line_length * np.cos(phi)
                y1 = line_length * np.sin(phi)
                z1 = line_length_z * np.cos(theta)
                x1_zplane = line_length * np.sin(theta) * np.cos(phi)
                y1_zplane = line_length * np.sin(theta) * np.sin(phi)

                axes[0].plot([0, x1], [0, y1], color=track_color, linewidth=1, alpha=0.5)
                axes[1].plot([0, z1], [0, y1_zplane], color=track_color, linewidth=1, alpha=0.5)
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

    def plot_and_save_event(
        self,
        event_index: int,
        save_path: str | None = None,
        just_background: bool = False,
    ) -> plt.Figure | None:
        """Plot tracks for a specific event and optionally save the figure.

        Parameters
        ----------
        event_index : int
            Event index to plot (0-based).
        save_path : str, optional
            Path to save the figure.
        just_background : bool, optional
            If True, only plot background hits.

        Returns:
        -------
        matplotlib.figure.Figure or None
            The figure object.
        """
        try:
            fig = self.plot_muon_tracks(event_index)

            if fig is not None and save_path is not None:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Figure saved to: {save_path}")
                plt.close(fig)

            # Generate background-only plot
            fig_background = self.plot_muon_tracks(event_index, just_background=True)

            if fig_background is not None and save_path is not None:
                save_path_bg = str(save_path).replace(".png", "_background.png")
                fig_background.savefig(save_path_bg, dpi=300, bbox_inches="tight")
                print(f"Background figure saved to: {save_path_bg}")
                plt.close(fig_background)

            return fig

        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error in plot_and_save_event: {e}")
            return None

    def calculate_detector_technology_statistics(self, dataloader: DataLoader, num_events: int = 1000) -> dict:
        """Calculate detector technology statistics from the dataset.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing batches of events.
        num_events : int, optional
            Maximum number of events to process (default: 1000).

        Returns:
        -------
        dict
            Statistics for each detector technology.
        """
        technology_mapping = {"MDT": 0, "RPC": 2, "TGC": 3, "STGC": 4, "MM": 5}

        tech_true_hits = dict.fromkeys(technology_mapping, 0)
        tech_total_hits = dict.fromkeys(technology_mapping, 0)
        total_true_hits = 0
        total_hits = 0

        try:
            for i, batch in tqdm(
                enumerate(dataloader),
                desc="Processing technology statistics",
                total=num_events,
            ):
                if i >= num_events:
                    break

                inputs, targets = batch

                hit_technologies = inputs["hit_spacePoint_technology"][0].numpy()
                hit_valid = targets["hit_valid"][0].numpy()
                hit_on_valid_particle = targets["hit_on_valid_particle"][0].numpy()

                valid_hit_mask = hit_valid.astype(bool)
                hit_technologies = hit_technologies[valid_hit_mask]
                hit_on_valid_particle = hit_on_valid_particle[valid_hit_mask]

                total_hits += len(hit_technologies)
                total_true_hits += np.sum(hit_on_valid_particle)

                for tech_name, tech_value in technology_mapping.items():
                    tech_mask = hit_technologies == tech_value
                    tech_total_hits[tech_name] += np.sum(tech_mask)
                    tech_true_hits[tech_name] += np.sum(hit_on_valid_particle & tech_mask)

            stats = {}
            for tech_name in technology_mapping:
                true_count = tech_true_hits[tech_name]
                total_count = tech_total_hits[tech_name]

                true_percentage = (true_count / total_true_hits * 100) if total_true_hits > 0 else 0.0
                total_percentage = (total_count / total_hits * 100) if total_hits > 0 else 0.0

                stats[tech_name] = {
                    "true_hits": int(true_count),
                    "total_hits": int(total_count),
                    "true_hits_percentage": true_percentage,
                    "total_hits_percentage": total_percentage,
                }

            stats["overall"] = {
                "total_true_hits": int(total_true_hits),
                "total_hits": int(total_hits),
                "events_processed": min(num_events, i + 1),
            }

            return stats

        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error calculating detector technology statistics: {e}")
            return {}

    def plot_and_save_true_hits_histogram(
        self,
        dataloader: DataLoader,
        max_events: int | None = None,
        num_bins: int = 100,
        num_events: int = 1000,
        show_plot: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure | None:
        """Create a histogram showing the distribution of true hits per event.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing batches of events.
        max_events : int, optional
            Maximum number of events to process.
        num_bins : int, optional
            Number of bins for the histogram (default: 100).
        num_events : int, optional
            Number of events to analyze (default: 1000).
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
            true_hits_counts = []
            n_0_hits = 0

            for i, batch in tqdm(enumerate(dataloader), desc="Processing events", total=num_events):
                if i >= num_events:
                    break
                _, targets = batch
                hit_count = np.sum(targets["hit_on_valid_particle"].numpy())
                true_hits_counts.append(hit_count)
                if hit_count == 0:
                    n_0_hits += 1

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

            return fig

        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error plotting true hits histogram: {e}")
            return None
