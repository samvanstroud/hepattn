import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from . import style_sheet
from .utils import custom_hist_v2, get_cmap, update_stylesheet

FIG_W = style_sheet.FIG_W
FIG_H_1ROW = style_sheet.FIG_H_1ROW
FIG_DPI = 300


def plot_scatter(ref_dict, comp_dict, comp_name, ref_name="truth"):
    """Args:
    ref_dict: {'pt': [], 'eta': [], 'phi': [], 'class': []} # flat_arrays.
    """
    cmap = get_cmap("lin_seg")

    n_row, n_col = 2, 3

    fig = plt.figure(figsize=(FIG_W, 0.8 * FIG_H_1ROW * n_row), dpi=FIG_DPI)
    gs = fig.add_gridspec(n_row, n_col, hspace=0.4, wspace=0.4)

    cl_masks = {
        "Charged": {"ref": ref_dict["class"] <= 2, "comp": comp_dict["class"] <= 2},
        "Neutral": {"ref": ref_dict["class"] >= 3, "comp": comp_dict["class"] >= 3},
    }

    for v_i, var in enumerate(["pt", "eta", "phi"]):
        for cl_i, (cl_name, mask_dict) in enumerate(cl_masks.items()):
            min_ = min(ref_dict[var][mask_dict["ref"]].min(), comp_dict[var][mask_dict["comp"]].min())
            max_ = max(ref_dict[var][mask_dict["ref"]].max(), comp_dict[var][mask_dict["comp"]].max())

            ax = fig.add_subplot(gs[cl_i, v_i])
            ax.hist2d(
                ref_dict[var][mask_dict["ref"]],
                comp_dict[var][mask_dict["comp"]],
                bins=50,
                range=((min_, max_), (min_, max_)),
                cmap=cmap,
                norm=LogNorm(),
            )
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
            ax.set_xlabel(f"{ref_name} {var}")
            ax.set_ylabel(f"{style_sheet.LABELS[comp_name]} {var}")
            ax.set_title(f"({cl_name})")

    return fig


def plot_residuals(data_dicts, ref_name="truth", pt_relative=False, log_y=False, qs=None, stylesheet=None):  # noqa: ARG001
    """Args:
    data_dicts: {name: (ref_dict, comp_dict), ...}
        ref_dict: {'pt': [], 'eta': [], 'phi': [], 'class': []} # flat_arrays.
    """
    colors, labels, histtypes, alphas, line_styles, _label_len = update_stylesheet(stylesheet)

    residual_dict = {}
    for name in data_dicts:
        residual_dict[name] = {"pt": [], "eta": [], "phi": [], "ref_class": []}
    abs_max_dict = {"Charged": {"pt": 0, "eta": 0, "phi": 0}, "Neutral": {"pt": 0, "eta": 0, "phi": 0}}
    if qs is None:
        qs = {"Charged": {"pt": 97, "eta": 97, "phi": 97}, "Neutral": {"pt": 97, "eta": 97, "phi": 97}}

    for name, (ref_dict, comp_dict, _, _) in data_dicts.items():
        for var in ["pt", "eta", "phi"]:
            residual_dict[name][var] = comp_dict[var] - ref_dict[var]
            if var == "pt" and pt_relative:
                residual_dict[name][var] /= ref_dict["pt"]

            mask = ref_dict["class"] <= 2
            abs_max_dict["Charged"][var] = max(
                abs_max_dict["Charged"][var], np.percentile(np.abs(residual_dict[name][var][mask]), qs["Charged"][var])
            )
            abs_max_dict["Neutral"][var] = max(
                abs_max_dict["Neutral"][var], np.percentile(np.abs(residual_dict[name][var][~mask]), qs["Neutral"][var])
            )

        residual_dict[name]["ref_class"] = ref_dict["class"]

    # figure
    n_row, n_col = 2, 3

    fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW * n_row), dpi=FIG_DPI)
    gs = fig.add_gridspec(n_row, n_col, hspace=0.3, wspace=0.3)
    axs = [[fig.add_subplot(gs[i, j]) for j in range(n_col)] for i in range(n_row)]

    xlabel_dict = {
        "pt": "$p_T^{reco} - p_T^{truth}$ [GeV]" if not pt_relative else "$(p_T^{reco} - p_T^{truth})/p_T^{truth}$",
        "eta": r"$\eta^{reco} - \eta^{truth}$",
        "phi": r"$\phi^{reco} - \phi^{truth}$",
    }

    for name, res_dict in residual_dict.items():
        # cl_masks = {
        #     'Charged': ref_dict['class'] <= 2, 'Neutral': ref_dict['class'] >= 3}

        cl_masks = {"Charged": res_dict["ref_class"] <= 2, "Neutral": res_dict["ref_class"] >= 3}

        for cl_i, (cl_name, cl_mask) in enumerate(cl_masks.items()):
            for v_i, var in enumerate(["pt", "eta", "phi"]):
                ax = axs[cl_i][v_i]
                bins = np.linspace(-abs_max_dict[cl_name][var], abs_max_dict[cl_name][var], 50)
                custom_hist_v2(
                    ax,
                    res_dict[var][cl_mask],
                    label_length=-1,
                    metrics="mean std iqr",
                    bins=bins,
                    histtype=histtypes[name],
                    label=labels[name],
                    color=colors[name],
                    linestyle=line_styles[name],
                    alpha=alphas[name],
                )

                ax.minorticks_on()
                ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
                ax.set_xlabel(xlabel_dict[var])
                ax.set_ylabel("Particles")
                ax.set_title(f"({cl_name})", y=1.1)
                ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
                if log_y:
                    ax.set_yscale("log")

    for axs_row in axs:
        for ax in axs_row:
            ax.legend()  # loc='lower left', bbox_to_anchor=(-0.27, 1.01))
            ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(residual_dict) * 0.23))

    return fig


def plot_residuals_neutrals(data_dicts, ref_name="truth", pt_relative=False, log_y=False, qs=None, stylesheet=None, separate_figures=False):  # noqa: ARG001
    """Args:
    data_dicts: {name: (ref_dict, comp_dict), ...}
        ref_dict: {'pt': [], 'eta': [], 'phi': [], 'class': []} # flat_arrays.
    """
    colors, labels, histtypes, alphas, line_styles, _label_len = update_stylesheet(stylesheet)

    residual_dict = {}
    for name in data_dicts:
        residual_dict[name] = {"pt": [], "eta": [], "phi": [], "ref_class": [], "f": {}}
    abs_max_dict = {"Neutral hadron": {"pt": 0, "eta": 0, "phi": 0}, "Photon": {"pt": 0, "eta": 0, "phi": 0}}
    if qs is None:
        qs = {"Neutral hadron": {"pt": 97, "eta": 97, "phi": 97}, "Photon": {"pt": 97, "eta": 97, "phi": 97}}

    for name, (ref_dict, comp_dict, ref_dict_unmatched, _) in data_dicts.items():
        for var in ["pt", "eta", "phi"]:
            residual_dict[name][var] = comp_dict[var] - ref_dict[var]
            if var == "pt" and pt_relative:
                residual_dict[name][var] /= ref_dict["pt"]

            mask = ref_dict["class"] == 3
            abs_max_dict["Neutral hadron"][var] = max(
                abs_max_dict["Neutral hadron"][var], np.percentile(np.abs(residual_dict[name][var][mask]), qs["Neutral hadron"][var])
            )

            mask = ref_dict["class"] == 4
            abs_max_dict["Photon"][var] = max(abs_max_dict["Photon"][var], np.percentile(np.abs(residual_dict[name][var][~mask]), qs["Photon"][var]))

        residual_dict[name]["ref_class"] = ref_dict["class"]

        # f computation
        for cl_name, cl in zip(["Neutral hadron", "Photon"], [3, 4], strict=False):
            f = np.sum(ref_dict["class"] == cl) / (np.sum(ref_dict["class"] == cl) + np.sum(ref_dict_unmatched["class"] == cl))
            residual_dict[name]["f"][cl_name] = f

    # figure
    n_row, n_col = 2, 3

    if separate_figures:
        figs = []
        axs = []
        for _i in range(n_row):
            tmp_axs = []
            for _j in range(n_col):
                fig, ax = plt.subplots(figsize=(FIG_W / 3, FIG_H_1ROW), dpi=FIG_DPI)
                figs.append(fig)
                tmp_axs.append(ax)
            axs.append(tmp_axs)
    else:
        fig = plt.figure(figsize=(FIG_W, FIG_H_1ROW * n_row), dpi=FIG_DPI * 3)
        gs = fig.add_gridspec(n_row, n_col, hspace=0.3, wspace=0.3)
        axs = [[fig.add_subplot(gs[i, j]) for j in range(n_col)] for i in range(n_row)]

    xlabel_dict = {
        "pt": "$p_T^{reco} - p_T^{truth}$ [GeV]" if not pt_relative else "$(p_T^{reco} - p_T^{truth})/p_T^{truth}$",
        "eta": r"$\eta^{reco} - \eta^{truth}$",
        "phi": r"$\phi^{reco} - \phi^{truth}$",
    }

    for name, res_dict in residual_dict.items():
        cl_masks = {"Neutral hadron": res_dict["ref_class"] == 3, "Photon": res_dict["ref_class"] == 4}

        for cl_i, (cl_name, cl_mask) in enumerate(cl_masks.items()):
            for v_i, var in enumerate(["pt", "eta", "phi"]):
                ax = axs[cl_i][v_i]
                bins = np.linspace(-abs_max_dict[cl_name][var], abs_max_dict[cl_name][var], 50)
                custom_hist_v2(
                    ax,
                    res_dict[var][cl_mask],
                    label_length=-1,
                    metrics="mean std iqr",
                    f=res_dict["f"][cl_name],
                    bins=bins,
                    histtype=histtypes[name],
                    label=labels[name],
                    color=colors[name],
                    linestyle=line_styles[name],
                    alpha=alphas[name],
                )

                ax.minorticks_on()
                ax.tick_params(which="both", direction="in", top=True, left=True, right=True)
                ax.set_xlabel(xlabel_dict[var])
                ax.set_ylabel("Particles")
                ax.set_title(cl_name, y=1.0)
                ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
                if log_y:
                    ax.set_yscale("log")

    for axs_row in axs:
        for ax in axs_row:
            ax.legend()  # loc='lower left', bbox_to_anchor=(-0.27, 1.01))
            ax.set_ylim(0, ax.get_ylim()[1] * (1 + len(residual_dict) * 0.23))

    return figs if separate_figures else fig


def plot_eff_fr_purity(eff_fr_purity_input_dict, stylesheet=None):
    """Compute efficiency, fake rate, and purity in pt bins separated by class
    efficiency = N(truth particles of this class matched to pflow particles) / N(total truth particles of this class)
    fake rate  = N(pflow particles of this class not matched to truth particles) / N(total pflow particles of this class)
    purity     = N(truth particles of this class that are matched to pflow particles of this class)
                 / N(total truth particles of this class that are matched to pflow particles).
    """
    colors, labels, _histtypes, _alphas, line_styles, _label_len = update_stylesheet(stylesheet)

    # subplots of eff, fr, and purity versus pt stacked vertically
    # All classes are drawn on same axis
    fig = plt.figure(figsize=(FIG_W / 2, 2 * FIG_H_1ROW), dpi=FIG_DPI)
    gs = fig.add_gridspec(3, 1, hspace=0.4, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for name, hung_part_dict in eff_fr_purity_input_dict.items():
        truth_pt_matched, truth_class_matched = hung_part_dict["ref_matched"]["pt"], hung_part_dict["ref_matched"]["class"]
        pflow_pt_matched, pflow_class_matched = hung_part_dict["comp_matched"]["pt"], hung_part_dict["comp_matched"]["class"]
        truth_pt_unmatched, truth_class_unmatched = hung_part_dict["ref_unmatched"]["pt"], hung_part_dict["ref_unmatched"]["class"]
        pflow_pt_unmatched, pflow_class_unmatched = hung_part_dict["comp_unmatched"]["pt"], hung_part_dict["comp_unmatched"]["class"]

        n_bins = 10
        pt_bins = np.logspace(np.log10(1), np.log10(100), n_bins + 1)
        pt_bin_mids = (pt_bins[1:] + pt_bins[:-1]) / 2

        eff_fr_dict = {"neut had": {"class": 3, "color": "cornflowerblue"}, "photon": {"class": 4, "color": "orange"}}

        # initialize binned arrays to zero for each class
        binned_arrays = ["n_truth_matched", "n_truth_tot", "eff", "n_pflow_unmatched", "n_pflow_tot", "fr", "n_class_matched", "purity"]
        for cl_dict in eff_fr_dict.values():
            for arr in binned_arrays:
                cl_dict[arr] = np.zeros(n_bins)

        # compute efficiency, fake rate, and purity in one go
        for cl_dict in eff_fr_dict.values():
            for bin_idx in range(len(pt_bins) - 1):
                truth_mask_matched = (
                    (truth_pt_matched > pt_bins[bin_idx]) & (truth_pt_matched <= pt_bins[bin_idx + 1]) & (truth_class_matched == cl_dict["class"])
                )
                truth_mask_unmatched = (
                    (truth_pt_unmatched > pt_bins[bin_idx])
                    & (truth_pt_unmatched <= pt_bins[bin_idx + 1])
                    & (truth_class_unmatched == cl_dict["class"])
                )
                pflow_mask_matched = (
                    (pflow_pt_matched > pt_bins[bin_idx]) & (pflow_pt_matched <= pt_bins[bin_idx + 1]) & (pflow_class_matched == cl_dict["class"])
                )
                pflow_mask_unmatched = (
                    (pflow_pt_unmatched > pt_bins[bin_idx])
                    & (pflow_pt_unmatched <= pt_bins[bin_idx + 1])
                    & (pflow_class_unmatched == cl_dict["class"])
                )

                cl_dict["n_truth_matched"][bin_idx] += np.sum(truth_mask_matched)
                cl_dict["n_truth_tot"][bin_idx] += np.sum(truth_mask_matched) + np.sum(truth_mask_unmatched)
                cl_dict["n_pflow_unmatched"][bin_idx] += np.sum(pflow_mask_unmatched)
                cl_dict["n_pflow_tot"][bin_idx] += np.sum(pflow_mask_matched) + np.sum(pflow_mask_unmatched)
                cl_dict["n_class_matched"][bin_idx] += np.sum(truth_mask_matched & (pflow_class_matched == cl_dict["class"]))

            # ratios
            cl_dict["eff"] = cl_dict["n_truth_matched"] / (cl_dict["n_truth_tot"] + 1e-8)
            cl_dict["fr"] = cl_dict["n_pflow_unmatched"] / (cl_dict["n_pflow_tot"] + 1e-8)
            cl_dict["purity"] = cl_dict["n_class_matched"] / (cl_dict["n_truth_matched"] + 1e-8)

        for cl_name, cl_dict in eff_fr_dict.items():
            ax1.hist(
                pt_bin_mids,
                bins=pt_bins,
                weights=cl_dict["eff"],
                ls=line_styles[name],
                label=f"{cl_name} ({labels[name]})",
                histtype="step",
                linewidth=1,
                color=colors[name][cl_name],
            )

        for cl_name, cl_dict in eff_fr_dict.items():
            ax2.hist(
                pt_bin_mids,
                bins=pt_bins,
                weights=cl_dict["fr"],
                ls=line_styles[name],
                label=f"{cl_name} ({labels[name]})",
                histtype="step",
                linewidth=1,
                color=colors[name][cl_name],
            )

        for cl_name, cl_dict in eff_fr_dict.items():
            ax3.hist(
                pt_bin_mids,
                bins=pt_bins,
                weights=cl_dict["purity"],
                ls=line_styles[name],
                label=f"{cl_name} ({labels[name]})",
                histtype="step",
                linewidth=1,
                color=colors[name][cl_name],
            )

    ax1.set_xlabel("Truth $p_\\mathrm{T}$ [GeV]")
    ax1.set_ylabel("Efficiency")
    # ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylim(0.75, 1.05)

    ax2.set_xlabel("Pred $p_\\mathrm{T}$ [GeV]")
    ax2.set_ylabel("Fake rate")
    # ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 0.35)

    ax3.set_xlabel("Truth $p_\\mathrm{T}$ [GeV]")
    ax3.set_ylabel("$p$(class match | match)")
    # ax3.set_ylim(-0.05, 1.05)
    ax3.set_ylim(0.65, 1.05)

    for ax in [ax1, ax2, ax3]:
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_xlim(pt_bins[0], pt_bins[-1])
        ax.legend(ncol=2, loc="lower right")
    ax2.legend(ncol=2, loc="upper right")

    return fig
