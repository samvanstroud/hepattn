from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .performance import Performance
from .plot_helper_event import DEFAULT_PT_BINS, PlotEventHelper
from .plot_helper_particle import DEFAULT_QS_ALL, DEFAULT_QS_NEUTRALS, PlotParticleHelper

DEFAULT_FIG_KWARGS = {
    "dpi": 300,
    "bbox_inches": "tight",
}


class PlotHelper(PlotEventHelper, PlotParticleHelper):
    def __init__(self, perf: Performance, labels=None, style_dict=None, plot_path=None):
        self.perf = perf
        if labels is None:
            self.labels = {name: name for name in perf.network_names}
        else:
            self.labels = labels
        if style_dict is None:
            self.style_dict = {name: {} for name in perf.network_names}
        else:
            self.style_dict = style_dict
        self.plot_path = Path(plot_path) if plot_path is not None else None
        if self.plot_path is not None:
            self.plot_path.mkdir(parents=True, exist_ok=True)

    def save_and_close(self, fig: Figure, filename: Path, fig_kwargs: dict | None = None):
        fig.savefig(filename, **(fig_kwargs or {}))
        plt.close(fig)

    def plot_event(self, pt_bins=None, fig_kwargs=None):
        assert self.plot_path is not None, "plot_path must be set to save figures."

        fig_kwargs = DEFAULT_FIG_KWARGS.copy() if fig_kwargs is None else DEFAULT_FIG_KWARGS | fig_kwargs
        self.save_and_close(
            self.plot_evt_res(),
            filename=self.plot_path / "event_response.png",
            fig_kwargs=fig_kwargs,
        )
        self.save_and_close(
            self.plot_jet_residuals(),
            filename=self.plot_path / "jet_residuals.png",
            fig_kwargs=fig_kwargs,
        )
        if pt_bins is None:
            pt_bins = DEFAULT_PT_BINS
        self.save_and_close(
            self.plot_jet_res_boxplot(bins=pt_bins),
            filename=self.plot_path / "jet_residuals_boxplot.png",
            fig_kwargs=fig_kwargs,
        )
        self.save_and_close(
            self.plot_jet_response(pt_bins=pt_bins, use_energy=True),
            filename=self.plot_path / "jet_response.png",
            fig_kwargs=fig_kwargs,
        )

    def plot_particles(self, qs_all=None, qs_neutrals=None, fig_kwargs=None):
        assert self.plot_path is not None, "plot_path must be set to save figures."
        fig_kwargs = DEFAULT_FIG_KWARGS.copy() if fig_kwargs is None else DEFAULT_FIG_KWARGS | fig_kwargs
        if qs_all is None:
            qs_all = DEFAULT_QS_ALL
        if qs_neutrals is None:
            qs_neutrals = DEFAULT_QS_NEUTRALS

        self.save_and_close(
            self.plot_residuals(pt_relative=True, log_y=True, qs=qs_all),
            filename=self.plot_path / "residuals_all.png",
            fig_kwargs=fig_kwargs,
        )
        self.save_and_close(
            self.plot_residuals_neutrals(pt_relative=True, log_y=True, qs=qs_neutrals),
            filename=self.plot_path / "residuals_neutrals.png",
            fig_kwargs=fig_kwargs,
        )
