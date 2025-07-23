import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from .style_sheet import FIG_DPI, FIG_H_1ROW, FIG_W
from .utils import custom_hist_v1, custom_hist_v2, update_stylesheet


def compute_jet_residual_dict(matched_jets_dict, dr_cut=0.1, leading_n_jets=999, pt_min=10, eta_max=2.5, n_events=None, entry_start=0):
    """Args:
    matched_jets: {name: (truth, reco), ...].
    """
    residual_dict = {}
    for name in matched_jets_dict:
        residual_dict[name] = {
            "pt": [],
            "pt_rel": [],
            "eta": [],
            "phi": [],
            "dR": [],
            "ref_pt": [],
            "ref_eta": [],
            "nconst": [],
            "e": [],
            "e_rel": [],
            "ref_e": [],
        }

    for name, (ref_jets, reco_jets) in matched_jets_dict.items():
        n_events_ = len(ref_jets) if n_events is None else min(len(ref_jets), n_events)
        ref_count = 0
        matched_count = 0
        for ev_i in range(entry_start, n_events_):
            ref_jets_ev, reco_jets_ev = ref_jets[ev_i], reco_jets[ev_i]
            for j_i, (ref_j, reco_j) in enumerate(zip(ref_jets_ev, reco_jets_ev, strict=False)):
                dr = ref_j.delta_r(reco_j)
                if dr < dr_cut and ref_j.pt > pt_min and abs(ref_j.eta) < eta_max:
                    residual_dict[name]["pt"].append(reco_j.pt - ref_j.pt)
                    residual_dict[name]["pt_rel"].append(residual_dict[name]["pt"][-1] / ref_j.pt)
                    residual_dict[name]["e"].append(reco_j.e - ref_j.e)
                    residual_dict[name]["e_rel"].append(residual_dict[name]["e"][-1] / ref_j.e)
                    residual_dict[name]["eta"].append(reco_j.eta - ref_j.eta)
                    residual_dict[name]["phi"].append(reco_j.phi - ref_j.phi)
                    residual_dict[name]["dR"].append(dr)
                    residual_dict[name]["ref_pt"].append(ref_j.pt)
                    residual_dict[name]["ref_eta"].append(ref_j.eta)
                    residual_dict[name]["ref_e"].append(ref_j.e)
                    residual_dict[name]["nconst"].append(reco_j.n_constituents - ref_j.n_constituents)

                    matched_count += 1
                ref_count += 1

                if j_i == leading_n_jets - 1:
                    break

        f_matched = matched_count / ref_count
        residual_dict[name]["f_matched"] = f_matched

    for key in residual_dict:
        for var in residual_dict[key]:
            residual_dict[key][var] = np.array(residual_dict[key][var])

    return residual_dict


def plot_jet_residuals(residual_dict, pt_relative=True, stylesheet=None, separate_figures=False):
    colors, labels, histtypes, alphas, line_styles, label_len = update_stylesheet(stylesheet)

    xlabel_dict = {
        "pt": "Jet $p_T^{reco} - p_T^{truth}$ [GeV]",
        "pt_rel": "Jet $(p_T^{reco} - p_T^{truth})/p_T^{truth}$",
        "e": "Jet $E^{reco} - E^{truth}$ [GeV]",
        "e_rel": "Jet $(E^{reco} - E^{truth})/E^{truth}$",
        "eta": r"Jet $\eta^{reco} - \eta^{truth}$",
        "phi": r"Jet $\phi^{reco} - \phi^{truth}$",
        "dR": "Jet $\\Delta R \\left( truth, \\; reco \\right)$",
        "nconst": r"$\Delta$ number of jet constituents",
    }

    jet_vars = ["pt", "dR", "nconst", "e_rel"]  # 'eta', 'phi']
    if pt_relative:
        jet_vars[0] = "pt_rel"

    figs = []
    for v_i, var in enumerate(jet_vars):
        if separate_figures:
            fig, ax = plt.subplots(figsize=(FIG_W / 3, FIG_H_1ROW), dpi=FIG_DPI)
        else:
            if v_i == 0:
                fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW * 2), dpi=FIG_DPI)
                gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax = fig.add_subplot(gs[v_i])

        comb = np.hstack([residual_dict[key][var] for key in residual_dict])
        # _min, _max = np.percentile(comb, 2), np.percentile(comb, 98)  # COCOA
        min_, max_ = np.percentile(comb, 5), np.percentile(comb, 95)  # CLIC
        abs_max = max(abs(min_), abs(max_))
        bins = np.linspace(-abs_max, abs_max, 50)
        if var == "dR":
            bins = np.linspace(0, abs_max, 50)
        if var == "nconst":
            bins = np.linspace(-abs_max - 0.5, abs_max + 0.5, int(2 * abs_max) + 2)

        for name, res in residual_dict.items():
            if var == "dR":
                ax.hist(
                    res[var],
                    bins=bins,
                    histtype=histtypes[name],
                    label=labels[name],
                    color=colors[name],
                    linestyle=line_styles[name],
                    alpha=alphas[name],
                )
            else:
                custom_hist_v2(
                    ax,
                    res[var],
                    label_length=label_len[name],
                    metrics="mean std iqr",
                    f=res["f_matched"],
                    bins=bins,
                    histtype=histtypes[name],
                    label=labels[name],
                    color=colors[name],
                    linestyle=line_styles[name],
                    alpha=alphas[name],
                )
        ax.set_xlabel(xlabel_dict[var])
        ax.set_ylabel("Jets")
        ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
        ax.legend()
        if var == "dR":
            ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(residual_dict) * 0.1))
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(residual_dict) * 0.2))

        if separate_figures:
            figs.append(fig)

    return figs if separate_figures else fig


def plot_jet_res_boxplot(residual_dict, var="pt", bins=None, stylesheet=None):
    colors, labels_, _histtypes, _alphas, _line_styles, _label_len = update_stylesheet(stylesheet)

    fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    if bins is None:
        bins = np.arange(0, 1000, 50)  # default
    bin_mins, bin_maxs = bins[:-1], bins[1:]
    bin_mids = (bins[:-1] + bins[1:]) / 2

    boxplot_data = []
    labels = []
    for bin_i, (bin_min, bin_max) in enumerate(zip(bin_mins, bin_maxs, strict=False)):
        bin_data = []
        for name, res in residual_dict.items():
            mask = (res[f"ref_{var}"] > bin_min) & (res[f"ref_{var}"] < bin_max)
            bin_data.append(res["pt_rel"][mask] if np.sum(mask) != 0 else [])
            if bin_i == 0:  # Add labels only once
                labels.append(labels_[name])
        boxplot_data.append(bin_data)

    ax.hlines(0, -1, len(bin_mids) * (len(residual_dict) + 1), color="black", linestyle="--", alpha=0.5, label="_nolegend_")

    # Plot box plots
    for i, (_bin_mid, data) in enumerate(zip(bin_mids, boxplot_data, strict=False)):
        positions = np.arange(len(data)) + i * (len(data) + 1)
        for j, (name, d) in enumerate(zip(residual_dict.keys(), data, strict=False)):
            ax.boxplot(
                d,
                positions=[positions[j]],
                widths=0.6,
                patch_artist=True,
                boxprops={"facecolor": colors[name], "color": colors[name], "alpha": 0.5},
                whiskerprops={"color": colors[name]},
                capprops={"color": colors[name]},
                medianprops={"color": "red"},
                showfliers=False,  # Do not show outliers
                whis=[2.5, 97.5],
            )  # Set whiskers to 5th and 95th percentiles

    ax.set_xticks(np.arange(len(bin_mids)) * (len(residual_dict) + 1) + len(residual_dict) / 2)
    if var == "pt":
        xlabel = "Jet $p_T^{truth}$ [GeV]"
        ax.set_xticklabels([f"{bin_min}-{bin_max}" for bin_min, bin_max in zip(bin_mins, bin_maxs, strict=False)])
    elif var == "eta":
        xlabel = r"Jet $\eta^{truth}$"
        ax.set_xticklabels([f"{bin_min:.1f}-{bin_max:.1f}" for bin_min, bin_max in zip(bin_mins, bin_maxs, strict=False)])
    elif var == "e":
        xlabel = "Jet $E^{truth}$ [GeV]"
        ax.set_xticklabels([f"{bin_min}-{bin_max}" for bin_min, bin_max in zip(bin_mins, bin_maxs, strict=False)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Jet ($p_T^{reco} - p_T^{truth}) / p_T^{truth}$")
    ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
    ax.legend(labels, loc="upper right", ncol=len(residual_dict))
    ax.set_xlim(-1, len(bin_mids) * (len(residual_dict) + 1))
    return fig


def plot_jet_response_old(residual_dict, pt_bins=None, stylesheet=None, separate_figures=False, use_energy=False):
    colors, labels, _histtypes, _alphas, line_styles, _label_len = update_stylesheet(stylesheet)
    pt_or_e = "pt"
    if use_energy:
        pt_or_e = "e"

    figs = []
    if separate_figures:
        fig1, ax1 = plt.subplots(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5 / 2), dpi=FIG_DPI)
        fig2, ax2 = plt.subplots(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5 / 2), dpi=FIG_DPI)

    else:
        fig = plt.figure(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5), dpi=FIG_DPI)
        gs = fig.add_gridspec(2, 1, hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

    if pt_bins is None:
        pt_bins = np.arange(0, 1000, 50)  # default

    pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
    pt_mids = (pt_bins[:-1] + pt_bins[1:]) / 2
    for name, res in residual_dict.items():
        y_vals1 = np.full(len(pt_mids), np.nan)
        y_vals2 = np.full(len(pt_mids), np.nan)

        for pt_i, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs, strict=False)):
            mask = (res[f"ref_{pt_or_e}"] > pt_min) & (res[f"ref_{pt_or_e}"] < pt_max)
            if np.sum(mask) == 0:
                continue
            res_vals = res[f"{pt_or_e}_rel"][mask]
            res_q_mask = (res_vals > np.percentile(res_vals, 2.5)) & (res_vals < np.percentile(res_vals, 97.5))
            y_vals1[pt_i] = np.std(res_vals[res_q_mask]) / (1 + np.mean(res_vals[res_q_mask]))

            q_25 = np.percentile(res[f"{pt_or_e}_rel"][mask], 25)
            q_50 = np.percentile(res[f"{pt_or_e}_rel"][mask], 50)
            q_75 = np.percentile(res[f"{pt_or_e}_rel"][mask], 75)
            y_vals2[pt_i] = (q_75 - q_25) / (q_50 + 1 + 1e-6)

        ax1.plot(pt_mids, y_vals1, label=labels[name], color=colors[name], linestyle=line_styles[name], marker="o")
        ax2.plot(pt_mids, y_vals2, label=labels[name], color=colors[name], linestyle=line_styles[name], marker="o")

    ax1.set_ylabel("Jet $\\sigma \\left( p_T^{reco} / p_T^{truth} \\right) / \\mu \\left( p_T^{reco} / p_T^{truth} \\right)$")
    ax2.set_ylabel("Jet $IQR \\left( p_T^{reco} / p_T^{truth} \\right) / median \\left( p_T^{reco} / p_T^{truth} \\right)$")
    if use_energy:
        ax1.set_ylabel("Jet $\\sigma \\left( E^{reco} / E^{truth} \\right) / \\mu \\left( E^{reco} / E^{truth} \\right)$")
        ax2.set_ylabel("Jet $IQR \\left( E^{reco} / E^{truth} \\right) / median \\left( E^{reco} / E^{truth} \\right)$")

    for ax in [ax1, ax2]:
        ax.set_xlabel("Jet $p_T^{truth}$ [GeV]")
        if use_energy:
            ax.set_xlabel("Jet $E^{truth}$ [GeV]")
        ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
        ax.legend()
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.set_ylim(ax.get_ylim()[0] - 0.2 * y_range, ax.get_ylim()[1] + 0.2 * y_range)

    if separate_figures:
        figs.extend((fig1, fig2))
        return figs

    return fig


def plot_jet_response(residual_dict, pt_bins=None, stylesheet=None, separate_figures=False, use_energy=False):
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

    colors, labels, _histtypes, _alphas, line_styles, _label_len = update_stylesheet(stylesheet)
    pt_or_e = "pt"
    if use_energy:
        pt_or_e = "e"

    figs = []
    if separate_figures:
        fig1, ax1 = plt.subplots(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5 / 2), dpi=FIG_DPI)
        fig2, ax2 = plt.subplots(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5 / 2), dpi=FIG_DPI)
        fig3, ax3 = plt.subplots(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5 / 2), dpi=FIG_DPI)

    else:
        fig = plt.figure(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5 / 2 * 3), dpi=FIG_DPI)
        gs = fig.add_gridspec(3, 1, hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

    if pt_bins is None:
        pt_bins = np.arange(0, 1000, 50)  # default

    # ax[1].fill_between(r_dict[algo]['bin_mid'],
    #                       r_dict[algo]['median'] - error_on_median(r_dict[algo]['n'], iqr=r_dict[algo]['iqr']),
    #                       r_dict[algo]['median'] + error_on_median(r_dict[algo]['n'], iqr=r_dict[algo]['iqr']),
    #                       color=cols[algo], alpha=0.3)
    # ax[1].plot(r_dict[algo]['bin_mid'], r_dict[algo]['median'], color=cols[algo], marker='o', markersize=3)
    # ax[2].fill_between(r_dict[algo]['bin_mid'],
    #                    r_dict[algo]['iqr'] - error_on_iqr(r_dict[algo]['n'], r_dict[algo]['iqr']),
    #                    r_dict[algo]['iqr'] + error_on_iqr(r_dict[algo]['n'], r_dict[algo]['iqr']),
    #                    color=cols[algo], alpha=0.3)
    # ax[2].plot(r_dict[algo]['bin_mid'], r_dict[algo]['iqr'], color=cols[algo], marker='o', markersize=3)
    # ax[3].plot(r_dict[algo]['bin_mid'], r_dict[algo]['iqr']/r_dict[algo]['median'], color=cols[algo], marker='o', markersize=5)

    pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
    pt_mids = (pt_bins[:-1] + pt_bins[1:]) / 2
    for name, res in residual_dict.items():
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
            color=colors[name],
            alpha=0.3,
            zorder=10,
        )
        ax1.plot(pt_mids, y_medians, label=labels[name], color=colors[name], linestyle=line_styles[name], zorder=10, marker="o", markersize=2)
        ax1_x_min = ax1.get_xlim()[0]
        ax1_x_max = ax1.get_xlim()[1]
        ax1.plot(np.linspace(ax1_x_min, ax1_x_max, 2), [0, 0], ls="--", color="k", alpha=0.5)
        ax1.set_xlim((ax1_x_min, ax1_x_max))

        ax2.fill_between(pt_mids, y_iqrs - error_on_iqr(ns, y_iqrs), y_iqrs + error_on_iqr(ns, y_iqrs), color=colors[name], alpha=0.3)
        ax2.plot(pt_mids, y_iqrs, label=labels[name], color=colors[name], linestyle=line_styles[name], marker="o", markersize=2)
        ax3.plot(pt_mids, y_iqrs / (y_medians + 1), label=labels[name], color=colors[name], linestyle=line_styles[name], marker="o")

    v_name = "p_T"
    if use_energy:
        v_name = "E"

    ax1.set_ylabel(f"Jet $median \\left( \\left( {v_name}^{{reco}} - {v_name}^{{truth}} \\right) / {v_name}^{{truth}} \\right)$")
    ax2.set_ylabel(f"Jet $IQR \\left( \\left( {v_name}^{{reco}} - {v_name}^{{truth}} \\right) / {v_name}^{{truth}} \\right)$")
    ax3.set_ylabel(
        f"Jet $IQR \\left( {v_name}^{{reco}} / {v_name}^{{truth}} \\right) / median \\left( {v_name}^{{reco}} / {v_name}^{{truth}} \\right)$"
    )

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(f"Jet ${v_name}^{{truth}}$ [GeV]")
        ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
        ax.legend()
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.set_ylim(ax.get_ylim()[0] - 0.2 * y_range, ax.get_ylim()[1] + 0.2 * y_range)

    ax1.set_ylim(-0.05, 0.05)

    if separate_figures:
        figs.extend((fig1, fig2, fig3))
        return figs

    return fig


def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    gauss1 = a1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    gauss2 = a2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    return gauss1 + gauss2


def plot_jet_response_gaussian_fit(residual_dict, pt_bins=None, stylesheet=None, return_fit_plots=False, use_energy=False):
    colors, labels, histtypes, alphas, line_styles, _label_len = update_stylesheet(stylesheet)

    pt_or_e = "pt"
    if use_energy:
        pt_or_e = "e"

    if pt_bins is None:
        pt_bins = np.arange(0, 1000, 50)  # default
    pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
    pt_mids = (pt_bins[:-1] + pt_bins[1:]) / 2

    if return_fit_plots:
        nrow = int(np.ceil((len(pt_bins) - 1) / 3))
        fig_fit = plt.figure(figsize=(24, nrow * 6), dpi=100)
        gs_fit = fig_fit.add_gridspec(nrow, 3, hspace=0.3, wspace=0.3)
        axs_fit = [fig_fit.add_subplot(gs_fit[i]) for i in range(len(pt_bins) - 1)]

    response_dict = {k: np.full(len(pt_mids), np.nan) for k in residual_dict}

    for pt_i, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs, strict=False)):
        res_dict_pt_i = {}
        for name, res in residual_dict.items():
            mask = (res[f"ref_{pt_or_e}"] > pt_min) & (res[f"ref_{pt_or_e}"] < pt_max)
            if np.sum(mask) == 0:
                continue
            res_vals = res[f"{pt_or_e}_rel"][mask] + 1
            res_dict_pt_i[name] = res_vals

        min_ = min(np.percentile(x, 2.5) for x in res_dict_pt_i.values())
        max_ = max(np.percentile(x, 97.5) for x in res_dict_pt_i.values())
        bins = np.linspace(min_, max_, 20)
        bin_mids = 0.5 * (bins[:-1] + bins[1:])

        for name, res_vals in res_dict_pt_i.items():
            hist, _ = np.histogram(res_vals, bins=bins)

            try:
                popt, _pcov = curve_fit(double_gaussian, bin_mids, hist)  # , p0=initial_guesses)
                a1, mu1, sigma1, a2, mu2, sigma2 = popt
                axs_fit[pt_i].plot(bin_mids, double_gaussian(bin_mids, *popt), color=colors[name])
                print(name, a1, mu1, sigma1, a2, mu2, sigma2)
            except:  # noqa: E722
                a1 = np.nan
                mu1 = np.nan
                sigma1 = np.nan
                a2 = np.nan
                mu2 = np.nan
                sigma2 = np.nan

            label = f"{labels[name]} {a1:.3f} * N({mu1:.3f}, {sigma1:.3f}) + {a2:.3f} * N({mu2:.3f}, {sigma2:.3f})"
            hist, _, _ = axs_fit[pt_i].hist(
                res_vals, bins=bins, density=False, label=label, color=colors[name], alpha=alphas[name], histtype=histtypes[name]
            )

            if np.abs(sigma1) < np.abs(sigma2):
                response_dict[name][pt_i] = np.abs(sigma1)  # / mu1
            else:
                response_dict[name][pt_i] = np.abs(sigma2)  # / mu2

            axs_fit[pt_i].legend()
            axs_fit[pt_i].grid(True)
            ymax = axs_fit[pt_i].get_ylim()[1] * 1.3
            axs_fit[pt_i].set_ylim((0, ymax))
            axs_fit[pt_i].set_title(f"{pt_min} < {pt_or_e} < {pt_max}")

    fig, ax = plt.subplots(figsize=(FIG_W // 2, FIG_H_1ROW * 1.5 / 2), dpi=FIG_DPI)
    for name, response in response_dict.items():
        ax.plot(pt_mids, response, label=labels[name], color=colors[name], linestyle=line_styles[name], marker="o")
    ax.set_ylabel("Jet $\\sigma \\left( p_T^{reco} / p_T^{truth} \\right) / \\mu \\left( p_T^{reco} / p_T^{truth} \\right)$")
    ax.set_xlabel("Jet $p_T^{truth}$ [GeV]")
    if use_energy:
        ax.set_ylabel("Jet $\\sigma \\left( E^{reco} / E^{truth} \\right) / \\mu \\left( E^{reco} /E^{truth} \\right)$")
        ax.set_xlabel("Jet $E^{truth}$ [GeV]")
    ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
    ax.legend()
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.set_ylim(ax.get_ylim()[0] - 0.2 * y_range, ax.get_ylim()[1] + 0.2 * y_range)

    if return_fit_plots:
        return fig, fig_fit

    return fig


def plot_jet_marginals(jet_dict, nleading=1, stylesheet=None):
    colors, labels, histtypes, alphas, _line_styles, _label_len = update_stylesheet(stylesheet)

    fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW * 2), dpi=FIG_DPI)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    varname_dict = {"pt": "Jet $p_T$ [GeV]", "eta": r"Jet $\eta$", "phi": r"Jet $\phi$"}
    jet_dict_mod = {}

    for _i, (name, jets) in enumerate(jet_dict.items()):
        var_dict = {"pt": [], "eta": [], "phi": []}
        for ev_jets in jets:
            for j_i, jet in enumerate(ev_jets):
                if j_i >= nleading:
                    break
                var_dict["pt"].append(jet.pt)
                var_dict["eta"].append(jet.eta)
                var_dict["phi"].append(jet.phi)

        jet_dict_mod[name] = var_dict

    bins = {"pt": None, "eta": None, "phi": None}
    for _var_i, var in enumerate(["pt", "eta", "phi"]):
        tmp_var = [var_dict[var] for var_dict in jet_dict_mod.values()]
        comb = np.hstack(tmp_var)
        bins[var] = np.linspace(comb.min(), comb.max(), 50)
        if var == "pt":
            bins[var] = np.linspace(comb.min(), np.percentile(comb, 99), 50)

    for var_i, var in enumerate(["pt", "pt", "eta", "phi"]):
        ax = fig.add_subplot(gs[var_i])
        for _name_i, (name, var_dict) in enumerate(jet_dict_mod.items()):
            ax.hist(var_dict[var], bins=bins[var], histtype=histtypes[name], label=labels[name], color=colors[name], alpha=alphas[name])
            ax.set_xlabel(varname_dict[var])
            ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
            ax.legend()

        if var_i == 1:
            ax.set_yscale("log")

    return fig


def compute_event_energy_residuals(ref, comp_dict):
    ev_eres_dict = {}
    for name in comp_dict:
        ev_eres_dict[name] = {"charged": [], "neutral": []}
    ref_dict = {"charged": [], "neutral": []}

    n_ev = len(ref["e"])
    for ev_i in tqdm(range(n_ev)):
        ref_ch_mask_ev = ref["charge"][ev_i] != 0
        ref_ch_energy_ev = ref["e"][ev_i][ref_ch_mask_ev].sum()
        ref_neut_energy_ev = ref["e"][ev_i][~ref_ch_mask_ev].sum()

        ref_dict["charged"].append(ref_ch_energy_ev)
        ref_dict["neutral"].append(ref_neut_energy_ev)

        for name in comp_dict:
            ch_mask_ev = comp_dict[name]["charge"][ev_i] != 0

            ev_eres_dict[name]["charged"].append(comp_dict[name]["e"][ev_i][ch_mask_ev].sum() - ref_ch_energy_ev)
            ev_eres_dict[name]["neutral"].append(comp_dict[name]["e"][ev_i][~ch_mask_ev].sum() - ref_neut_energy_ev)

    for name in ev_eres_dict:
        ev_eres_dict[name]["charged"] = np.array(ev_eres_dict[name]["charged"])
        ev_eres_dict[name]["neutral"] = np.array(ev_eres_dict[name]["neutral"])

    ref_dict["charged"] = np.array(ref_dict["charged"])
    ref_dict["neutral"] = np.array(ref_dict["neutral"])
    ref_dict["total"] = ref_dict["charged"] + ref_dict["neutral"]

    return ref_dict, ev_eres_dict


def plot_event_neut_energy_res(ev_eres_dict, stylesheet=None):
    colors, labels, histtypes, alphas, line_styles, _label_len = update_stylesheet(stylesheet)

    n_row = int(np.ceil(len(ev_eres_dict) / 3))
    fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW / 1.5 * n_row), dpi=FIG_DPI)
    gs = fig.add_gridspec(n_row, 3, hspace=0.5, wspace=0.3)

    for k_i, (pt_key, (ref, eres_dict)) in enumerate(ev_eres_dict.items()):
        ax = fig.add_subplot(gs[k_i])

        for name, eres in eres_dict.items():
            e_neut_frac = eres["neutral"] / ref["total"]
            ax.hist(
                e_neut_frac,
                bins=np.linspace(0, 0.5, 26),
                histtype=histtypes[name],
                label=labels[name],
                color=colors[name],
                alpha=alphas[name],
                ls=line_styles[name],
                density=True,
            )

        ax.set_title(f"$p_T = ${pt_key} GeV")
        ax.set_xlabel(r"$E_\mathrm{neutral} / E_{\pi^+}$")
        ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
        ax.legend()

        # change the y-axis to fraction
        vals = ax.get_yticks()
        ax.set_yticks(vals)
        ax.set_yticklabels([str(2 * x / 100) for x in vals])
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, ax.get_ylim()[1] * 0.9)

    return fig


def plot_met_res_and_ht_res(truth_pt, truth_phi, comp_dict, stylesheet=None, separate_figures=False):
    colors, labels, histtypes, alphas, _line_styles, label_len = update_stylesheet(stylesheet)

    met_res_dict = {}
    ht_res_dict = {}
    for name in comp_dict:
        met_res_dict[name] = []
        ht_res_dict[name] = []

    n_events = len(truth_pt)

    truth_met = []
    truth_ht = []
    for ev_i in range(n_events):
        truth_metx = -np.sum(truth_pt[ev_i] * np.cos(truth_phi[ev_i]))
        truth_mety = -np.sum(truth_pt[ev_i] * np.sin(truth_phi[ev_i]))
        truth_met.append(np.sqrt(truth_metx**2 + truth_mety**2))
        truth_ht.append(np.sum(truth_pt[ev_i]))

    for k, (pt, phi) in comp_dict.items():
        for ev_i in range(n_events):
            reco_metx = -np.sum(pt[ev_i] * np.cos(phi[ev_i]))
            reco_mety = -np.sum(pt[ev_i] * np.sin(phi[ev_i]))
            reco_met = np.sqrt(reco_metx**2 + reco_mety**2)
            reco_ht = np.sum(pt[ev_i])
            met_res_dict[k].append((reco_met - truth_met[ev_i]) / (truth_met[ev_i] + 1e-8))
            ht_res_dict[k].append((reco_ht - truth_ht[ev_i]) / (truth_ht[ev_i] + 1e-8))

    comb_met = []
    comb_ht = []
    for name in met_res_dict:
        comb_met.append(met_res_dict[name])
        comb_ht.append(ht_res_dict[name])
    comb_met = np.hstack(comb_met)
    comb_ht = np.hstack(comb_ht)
    abs_max_met = max(abs(np.percentile(comb_met, 10)), abs(np.percentile(comb_met, 90)))
    abs_max_ht = max(abs(np.percentile(comb_ht, 1)), abs(np.percentile(comb_ht, 99)))
    bins_met = np.linspace(-abs_max_met, abs_max_met, 50)
    bins_ht = np.linspace(-abs_max_ht, abs_max_ht, 50)

    if separate_figures:
        fig1, ax1 = plt.subplots(figsize=(FIG_W / 3, FIG_H_1ROW), dpi=FIG_DPI)
        fig2, ax2 = plt.subplots(figsize=(FIG_W / 3, FIG_H_1ROW), dpi=300)
    else:
        fig1 = plt.figure(figsize=(FIG_W / 3, FIG_H_1ROW * 2), dpi=FIG_DPI)
        gs = fig1.add_gridspec(2, 1, hspace=0.3)
        ax1 = fig1.add_subplot(gs[0])
        ax2 = fig1.add_subplot(gs[1])
    for name in comp_dict:
        custom_hist_v1(
            ax1,
            met_res_dict[name],
            label_length=label_len[name],
            bins=bins_met,
            histtype=histtypes[name],
            label=labels[name],
            alpha=alphas[name],
            color=colors[name],
        )
    ax1.set_xlabel(r"$(p_T^{miss} reco - p_T^{miss} truth) / p_T^{miss} truth$")

    for name in comp_dict:
        custom_hist_v1(
            ax2,
            ht_res_dict[name],
            label_length=label_len[name],
            bins=bins_ht,
            histtype=histtypes[name],
            label=labels[name],
            alpha=alphas[name],
            color=colors[name],
        )
    ax2.set_xlabel(r"$(H_T^{reco} - H_T^{truth}) / H_T^{truth}$")

    for ax in [ax1, ax2]:
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
        ax.legend()
        ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(met_res_dict) * 0.1))
        ax.set_ylabel("Events")
    if separate_figures:
        return fig1, fig2
    return fig1


def get_invariant_mass(jets, option="two-jet"):
    m = np.nan
    if option == "two-jet":
        if len(jets) >= 2:
            m = (jets[0] + jets[1]).mass
    elif option == "one-jet" and len(jets) >= 1:
        m = jets[0].mass
    return m


def plot_jet_inv_mass(jet_dict, leading_jet_option="two-jet", xlabel_flag="jj", stylesheet=None):
    colors, labels, histtypes, alphas, _line_styles, label_len = update_stylesheet(stylesheet)

    mass_dict = {}
    for name in jet_dict:
        mass_dict[name] = []

    n_events = len(jet_dict["truth"])
    for name, jets in jet_dict.items():
        for ev_i in range(n_events):
            mass_dict[name].append(get_invariant_mass(jets[ev_i], option=leading_jet_option))

    fig, ax = plt.subplots(figsize=(FIG_W / 3, FIG_H_1ROW), dpi=FIG_DPI)

    comb = [np.array(mass_dict[name]) for name in jet_dict]
    comb = np.hstack(comb)
    bins = np.linspace(np.nanpercentile(comb, 3), np.nanpercentile(comb, 98), 50)

    for name in jet_dict:
        custom_hist_v1(
            ax,
            mass_dict[name],
            label_length=label_len[name],
            bins=bins,
            histtype=histtypes[name],
            label=labels[name],
            alpha=alphas[name],
            color=colors[name],
        )
    xlabel = rf"$m_{{{xlabel_flag}}}$ [GeV]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
    ax.legend()
    ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(jet_dict) * 0.1))

    return fig


def plot_jet_inv_mass_residual(ref_jets, comp_jet_dict, leading_jet_option="two-jet", xlabel_flag="jj", stylesheet=None):
    colors, labels, histtypes, alphas, _line_styles, label_len = update_stylesheet(stylesheet)

    res_dict = {}
    for name in comp_jet_dict:
        res_dict[name] = []

    n_events = len(ref_jets)

    ref_masses = [get_invariant_mass(ref_jets[ev_i], option=leading_jet_option) for ev_i in range(n_events)]
    for name, jets in comp_jet_dict.items():
        for ev_i in range(n_events):
            res_dict[name].append(get_invariant_mass(jets[ev_i], option=leading_jet_option) - ref_masses[ev_i])

    fig, ax = plt.subplots(figsize=(FIG_W / 3, FIG_H_1ROW), dpi=FIG_DPI)

    comb = [np.array(res_dict[name]) for name in comp_jet_dict]
    comb = np.hstack(comb)
    abs_max = max(abs(np.nanpercentile(comb, 5)), abs(np.nanpercentile(comb, 95)))
    bins = np.linspace(-abs_max, abs_max, 50)

    for name in comp_jet_dict:
        custom_hist_v1(
            ax,
            res_dict[name],
            label_length=label_len[name],
            bins=bins,
            histtype=histtypes[name],
            label=labels[name],
            alpha=alphas[name],
            color=colors[name],
        )
    xlabel = rf"$m_{{{xlabel_flag}}}^{{reco}} - m_{{{xlabel_flag}}}^{{truth}} [GeV]$"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
    ax.legend()
    ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(comp_jet_dict) * 0.1))

    return fig


def plot_leading_jet_substructure(substructure_dict, stylesheet=None, separate_figures=False):
    colors, labels, histtypes, alphas, line_styles, _label_len = update_stylesheet(stylesheet)

    figs = []
    for i, var_name in enumerate(["$ln (D_2)$", "$C_2$", "$C_3$"]):
        if separate_figures:
            fig = plt.figure(figsize=(FIG_W / 3, FIG_H_1ROW), dpi=FIG_DPI)
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.4], hspace=0)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

        else:
            if i == 0:
                fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW), dpi=FIG_DPI)
                gs = fig.add_gridspec(2, 3, hspace=0, wspace=0.3, height_ratios=[1, 0.4])
            ax1 = fig.add_subplot(gs[0, i])
            ax2 = fig.add_subplot(gs[1, i])

        # marginals
        comb = [substructure[i] for substructure in substructure_dict.values()]
        comb = np.hstack(comb)
        bins = np.linspace(np.nanpercentile(comb, 1), np.nanpercentile(comb, 99), 10)

        stored_hists = {}
        for name, substructure in substructure_dict.items():
            h, _, _ = ax1.hist(
                substructure[i],
                bins=bins,
                histtype=histtypes[name],
                label=labels[name],
                color=colors[name],
                alpha=alphas[name],
                ls=line_styles[name],
            )
            stored_hists[name] = h

        # ratio to truth
        # bin_mids = (bins[:-1] + bins[1:]) * 0.5
        for name, h in stored_hists.items():
            ratio = h / stored_hists["truth"]
            for ri, r in enumerate(ratio):
                ax2.hlines(r, bins[ri], bins[ri + 1], color=colors[name], linestyle=line_styles[name])

        for ax in [ax1, ax2]:
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
            ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
        ax2.set_xlabel(var_name)
        ax1.set_ylabel("Jets")
        ax2.set_ylabel("Ratio")
        ax1.legend()

        if separate_figures:
            figs.append(fig)

    return figs if separate_figures else fig
