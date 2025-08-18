import matplotlib.pyplot as plt
import numpy as np

from .performance import Performance
from .style_sheet import FIG_DPI, FIG_H_1ROW, FIG_W
from .utils import custom_hist_v1, custom_hist_v2

DEFAULT_PT_BINS = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])


class PlotEventHelper:
    def __init__(self, perf: Performance, labels=None, style_dict=None):
        self.perf = perf
        if labels is None:
            self.labels = {name: name for name in perf.network_names}
        else:
            self.labels = labels
        if style_dict is None:
            self.style_dict = {name: {} for name in perf.network_names}
        else:
            self.style_dict = style_dict

    def plot_evt_res(self, separate_figures=False):
        evt_vars = ["met", "ht", "nconst_ch", "nconst_neut"]
        xlabel_dict = {
            "met": r"$(p_T^{\text{miss}, \text{reco}} - p_T^{\text{miss}, \text{truth}}) / p_T^{\text{miss}, \text{truth}}$",
            "ht": r"$(H_T^\text{reco} - H_T^\text{truth}) / H_T^\text{truth}$",
            "nconst_ch": r"$\Delta$ number of charged constituents",
            "nconst_neut": r"$\Delta$ number of neutral constituents",
        }

        figs = []
        for v_i, var in enumerate(evt_vars):
            if separate_figures:
                fig, ax = plt.subplots(figsize=(FIG_W / 3, FIG_H_1ROW), dpi=FIG_DPI, constrained_layout=True)
            else:
                if v_i == 0:
                    fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW * 2), dpi=FIG_DPI, constrained_layout=True)
                    gs = fig.add_gridspec(2, 2, hspace=0.0, wspace=0.0)
                ax = fig.add_subplot(gs[v_i])
            comb = np.hstack([self.perf.data[name][var + "_res"] for name in self.perf.network_names])
            match var:
                case "met":
                    min_percent, max_percent = 2, 98
                case "ht":
                    min_percent, max_percent = 2, 98
                case "nconst_ch" | "nconst_neut":
                    min_percent, max_percent = 2, 98
            min_, max_ = np.percentile(comb, min_percent), np.percentile(comb, max_percent)
            abs_max = max(abs(min_), abs(max_))
            bins = np.linspace(-abs_max, abs_max, 50)
            if var in {"nconst_ch", "nconst_neut"}:
                bins = np.linspace(-abs_max - 0.5, abs_max + 0.5, int(2 * abs_max) + 2)
            for name in self.perf.network_names:
                custom_hist_v1(
                    ax,
                    self.perf.data[name][var + "_res"],
                    bins=bins,
                    label=self.labels[name],
                    **self.style_dict[name],
                )
            ax.set_xlabel(xlabel_dict[var])
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
            ax.legend()
            ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0)
            ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(self.perf.network_names) * 0.18))
            ax.set_ylabel("Events")
            if separate_figures:
                figs.append(fig)
        return figs if separate_figures else fig

    def plot_jet_residuals(self, pt_relative=True, separate_figures=False):
        xlabel_dict = {
            "pt": r"Jet $p_T^\text{reco} - p_T^\text{truth}$ [GeV]",
            "pt_rel": r"Jet $(p_T^\text{reco} - p_T^\text{truth})/p_T^\text{truth}$",
            "e": r"Jet $E^\text{reco} - E^\text{truth}$ [GeV]",
            "e_rel": r"Jet $(E^\text{reco} - E^\text{truth})/E^\text{truth}$",
            "eta": r"Jet $\eta^\text{reco} - \eta^\text{truth}$",
            "phi": r"Jet $\phi^\text{reco} - \phi^\text{truth}$",
            "dR": r"Jet $\Delta R \left(\text{truth}, \; \text{reco} \right)$",
            "nconst": r"$\Delta$ number of jet constituents",
        }

        jet_vars = ["pt", "e_rel", "nconst", "dR"]  # 'eta', 'phi']
        if pt_relative:
            jet_vars[0] = "pt_rel"

        figs = []
        for v_i, var in enumerate(jet_vars):
            if separate_figures:
                fig, ax = plt.subplots(figsize=(FIG_W / 3, FIG_H_1ROW), dpi=FIG_DPI, constrained_layout=True)
            else:
                if v_i == 0:
                    fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW * 2), dpi=FIG_DPI, constrained_layout=True)
                    gs = fig.add_gridspec(2, 2, hspace=0.0, wspace=0.0)
                ax = fig.add_subplot(gs[v_i])

            comb = np.hstack([self.perf.data[name]["jet_residuals"][var] for name in self.perf.network_names])
            # _min, _max = np.percentile(comb, 2), np.percentile(comb, 98)  # COCOA
            min_, max_ = np.percentile(comb, 2), np.percentile(comb, 98)  # CLIC
            abs_max = max(abs(min_), abs(max_))
            bins = np.linspace(-abs_max, abs_max, 50)
            if var == "dR":
                bins = np.linspace(0, abs_max, 50)
            if var == "nconst":
                bins = np.linspace(-abs_max - 0.5, abs_max + 0.5, int(2 * abs_max) + 2)

            for name in self.perf.network_names:
                res = self.perf.data[name]["jet_residuals"]
                if var == "dR":
                    ax.hist(
                        np.clip(res[var], bins[0], bins[-1]),
                        bins=bins,
                        label=self.labels[name],
                        **self.style_dict[name],
                    )
                else:
                    custom_hist_v2(
                        ax,
                        res[var],
                        metrics="mean std iqr",
                        f=res["f_matched"],
                        bins=bins,
                        label=self.labels[name],
                        **self.style_dict[name],
                    )
            ax.set_xlabel(xlabel_dict[var])
            ax.set_ylabel("Jets")
            ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0)
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
            ax.legend()
            if var == "dR":
                ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(self.perf.network_names) * 0.1))
            else:
                ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(self.perf.network_names) * 0.25))

            if separate_figures:
                figs.append(fig)

        return figs if separate_figures else fig

    def plot_jet_res_boxplot(self, var="pt", bins=None):
        fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW), dpi=FIG_DPI, constrained_layout=True)
        ax = fig.add_subplot(111)
        if bins is None:
            bins = np.arange(0, 1000, 50)  # default
        bin_mins, bin_maxs = bins[:-1], bins[1:]
        bin_mids = (bins[:-1] + bins[1:]) / 2

        boxplot_data = []
        labels = []
        for bin_i, (bin_min, bin_max) in enumerate(zip(bin_mins, bin_maxs, strict=False)):
            bin_data = []
            for name in self.perf.network_names:
                res = self.perf.data[name]["jet_residuals"]
                mask = (res[f"ref_{var}"] > bin_min) & (res[f"ref_{var}"] < bin_max)
                bin_data.append(res["pt_rel"][mask] if np.sum(mask) != 0 else [])
                if bin_i == 0:  # Add labels only once
                    labels.append(self.labels[name])
            boxplot_data.append(bin_data)

        ax.hlines(0, -1, len(bin_mids) * (len(self.perf.network_names) + 1), color="black", linestyle="--", alpha=0.5, label="_nolegend_")

        # Plot box plots
        for i, (_bin_mid, data) in enumerate(zip(bin_mids, boxplot_data, strict=False)):
            positions = np.arange(len(data)) + i * (len(data) + 1)
            for j, (name, d) in enumerate(zip(self.perf.network_names, data, strict=False)):
                color = self.style_dict[name].get("color", "C" + str(j % 10))
                ax.boxplot(
                    d,
                    positions=[positions[j]],
                    widths=0.6,
                    patch_artist=True,
                    boxprops={"facecolor": color, "color": color, "alpha": 0.5},
                    whiskerprops={"color": color},
                    capprops={"color": color},
                    medianprops={"color": "red"},
                    showfliers=False,  # Do not show outliers
                    whis=[2.5, 97.5],
                )  # Set whiskers to 5th and 95th percentiles

        ax.set_xticks(np.arange(len(bin_mids)) * (len(self.perf.network_names) + 1) + len(self.perf.network_names) / 2)
        if var == "pt":
            xlabel = r"Jet $p_T^\text{truth}$ [GeV]"
            ax.set_xticklabels([f"{bin_min}-{bin_max}" for bin_min, bin_max in zip(bin_mins, bin_maxs, strict=False)])
        elif var == "eta":
            xlabel = r"Jet $\eta^\text{truth}$"
            ax.set_xticklabels([f"{bin_min:.1f}-{bin_max:.1f}" for bin_min, bin_max in zip(bin_mins, bin_maxs, strict=False)])
        elif var == "e":
            xlabel = r"Jet $E^\text{truth}$ [GeV]"
            ax.set_xticklabels([f"{bin_min}-{bin_max}" for bin_min, bin_max in zip(bin_mins, bin_maxs, strict=False)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Jet ($p_T^\text{reco} - p_T^\text{truth}) / p_T^\text{truth}$")
        ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
        ax.legend(labels, loc="upper right", ncol=len(self.perf.network_names))
        ax.set_xlim(-1, len(bin_mids) * (len(self.perf.network_names) + 1))
        return fig

    def plot_jet_response(self, pt_bins=None, separate_figures=False, use_energy=False):
        def error_on_median(n, iqr=None, std=None):
            if std is not None:
                std = std
            elif iqr is not None:
                # IQR ≈ 1.35 * std
                std = iqr / 1.35
            else:
                raise ValueError("Either std or iqr must be provided")

            # SE(Median) ≈ 1.253 * std / sqrt(N)
            return 1.253 * std / np.sqrt(n)

        def error_on_iqr(n, iqr):
            # SE(IQR) ≈ 1.16 * IQR / sqrt(N)
            return 1.16 * iqr / np.sqrt(n)

        pt_or_e = "pt"
        if use_energy:
            pt_or_e = "e"

        figs = []
        if separate_figures:
            fig1, ax1 = plt.subplots(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5 / 2), dpi=FIG_DPI, constrained_layout=True)
            fig2, ax2 = plt.subplots(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5 / 2), dpi=FIG_DPI, constrained_layout=True)

        else:
            fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW * 1.1), dpi=FIG_DPI, constrained_layout=True)
            gs = fig.add_gridspec(1, 2, hspace=0.0)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

        if pt_bins is None:
            pt_bins = np.arange(0, 1000, 50)  # default

        pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
        pt_mids = (pt_bins[:-1] + pt_bins[1:]) / 2
        for i, name in enumerate(self.perf.network_names):
            res = self.perf.data[name]["jet_residuals"]
            color = self.style_dict[name].get("color", "C" + str(i % 10))
            linestyle = self.style_dict[name].get("linestyle", "-")
            y_medians = np.full(len(pt_mids), np.nan)
            y_iqrs = np.full(len(pt_mids), np.nan)
            ns = np.full(len(pt_mids), np.nan)

            for pt_i, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs, strict=False)):
                mask = (res[f"ref_{pt_or_e}"] > pt_min) & (res[f"ref_{pt_or_e}"] < pt_max)
                if np.sum(mask) == 0:
                    continue
                res_vals = res[f"{pt_or_e}_rel"][mask]

                y_medians[pt_i] = np.percentile(res_vals, 50)
                y_iqrs[pt_i] = np.percentile(res_vals, 75) - np.percentile(res_vals, 25)
                ns[pt_i] = len(res_vals)

            ax1.fill_between(
                pt_mids,
                y_medians - error_on_median(ns, iqr=y_iqrs),
                y_medians + error_on_median(ns, iqr=y_iqrs),
                color=color,
                alpha=0.3,
                zorder=10,
            )
            ax1.plot(
                pt_mids,
                y_medians,
                label=self.labels[name],
                color=color,
                linestyle=linestyle,
                zorder=10,
                marker="o",
                markersize=2,
            )
            ax1_x_min = ax1.get_xlim()[0]
            ax1_x_max = ax1.get_xlim()[1]
            ax1.plot(np.linspace(ax1_x_min, ax1_x_max, 2), [0, 0], ls="--", color="k", alpha=0.5)
            ax1.set_xlim((ax1_x_min, ax1_x_max))

            ax2.fill_between(pt_mids, y_iqrs - error_on_iqr(ns, y_iqrs), y_iqrs + error_on_iqr(ns, y_iqrs), color=color, alpha=0.3)
            ax2.plot(pt_mids, y_iqrs, label=self.labels[name], color=color, linestyle=linestyle, marker="o", markersize=2)

        v_name = "p_T"
        if use_energy:
            v_name = "E"

        ax1.set_ylabel(
            r"Jet $\text{median} \left(\left("
            + v_name
            + r"^{\text{reco}} - "
            + v_name
            + r"^{\text{truth}}\right) / "
            + v_name
            + r"^{\text{truth}}\right)$"
        )
        ax2.set_ylabel(
            r"Jet $\text{IQR} \left(\left("
            + v_name
            + r"^{\text{reco}} - "
            + v_name
            + r"^{\text{truth}}\right) / "
            + v_name
            + r"^{\text{truth}}\right)$"
        )

        for ax in [ax1, ax2]:
            ax.set_xlabel(r"Jet $" + v_name + r"^{\text{truth}}$ [GeV]")
            ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.2, zorder=0)
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
            ax.legend(loc="upper right", ncol=len(self.perf.network_names), fontsize=8)
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.set_ylim(ax.get_ylim()[0] - 0.2 * y_range, ax.get_ylim()[1] + 0.2 * y_range)

        ax1.set_ylim(-0.05, 0.05)

        if separate_figures:
            figs.extend((fig1, fig2))
            return figs

        return fig
