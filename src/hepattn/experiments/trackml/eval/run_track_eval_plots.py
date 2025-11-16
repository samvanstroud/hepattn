import math
import pathlib

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from plot_utils import binned, hist_plot, profile_plot
from track_evaluate import load_events

# ----------------------------------------------------
# Plotting setup
# ----------------------------------------------------

plt.rcParams["figure.dpi"] = 400
# plt.rcParams["text.usetex"] = True
plt.rcParams["text.usetex"] = False
# disabled due to missing font in texlive on the Nikhef clusters
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True

training_colours = {
    "Paper": "tab:green",
    "TRK-v0 0.9 GeV": "tab:orange",  # |eta| < 4.0
    "trackml 1 GeV": "tab:blue",  # |eta| < 2.5
    "trackml 0.6 GeV": "mediumvioletred",
    "600 MeV": "mediumvioletred",
    "750 MeV": "cornflowerblue",
    "1 GeV": "mediumseagreen",
}

qty_bins = {
    "pt": np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10]),
    # "eta": np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]),
    "eta": np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
    "phi": np.array([-math.pi, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, math.pi]),
    "vz": np.array([-100, -50, -20, -10, 0, 10, 20, 50, 100]),
}

qty_symbols = {"pt": "p_\\mathrm{T}", "eta": "\\eta", "phi": "\\phi", "vz": "v_z"}
qty_units = {"pt": "[GeV]", "eta": "", "phi": "", "vz": "[mm]"}
out_dir = "test/"

# ----------------------------------------------------
# Read configuration file information
# ----------------------------------------------------

tracking_fnames = {
    "Paper": "/data/atlas/users/slin/hepattn/src/hepattn/experiments/trackml/logs/epoch=028-val_loss=1.29786__test_test.h5",
    "TRK-v0 0.9 GeV": "/data/atlas/users/slin/myHepattn/hepattn/src/hepattn/experiments/trackml/logs/TRK-v0-full_20250906-T205842/\
        ckpts/epoch=029-val_loss=50.09092_test_eval.h5",
    "trackml 1 GeV": "/data/atlas/users/slin/hepattn/src/hepattn/experiments/trackml/logs/trackml_tracking_20250711-T162137/\
        ckpts/epoch=029-val_loss=12.68400_test_eval.h5",
    "trackml 0.6 GeV": "/data/atlas/users/slin/hepattn/src/hepattn/experiments/trackml/logs/trackml_tracking_20251103-T102312/\
        ckpts/epoch=029-val_loss=3.11541_test_eval.h5",
}
has_regression = {
    "Paper": True,
    "TRK-v0 0.9 GeV": True,
    "trackml 1 GeV": False,
    "trackml 0.6 GeV": False,
}
tracking_config_fname = {
    "Paper": "/data/atlas/users/slin/hepattn/src/hepattn/experiments/trackml/logs/trackml_tracking_20251103-T102312/config.yaml",
    "TRK-v0 0.9 GeV": "/data/atlas/users/slin/myHepattn/hepattn/src/hepattn/experiments/trackml/logs/TRK-v0-full_20250906-T205842/config.yaml",
    "trackml 1 GeV": "/data/atlas/users/slin/hepattn/src/hepattn/experiments/trackml/logs/trackml_tracking_20250711-T162137/config.yaml",
    "trackml 0.6 GeV": "/data/atlas/users/slin/hepattn/src/hepattn/experiments/trackml/logs/trackml_tracking_20251103-T102312/config.yaml",
}
tracking_params = ["particle_min_pt", "particle_max_abs_eta"]
tracking_configs = {}
for name in tracking_config_fname:
    with pathlib.Path(tracking_config_fname[name]).open() as f:
        fconfig = yaml.safe_load(f)
        print("name: " + fconfig["name"])
        for i in tracking_params:
            print("> " + i + "\t: ", fconfig["data"][i])
    tracking_configs[name] = fconfig

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

tracking_results = {}
num_events = 100
index_list = None  # ["event_0", ..., "event_99"] or ["29800", ..., "29899"]
for name, fname in tracking_fnames.items():
    eta_cut = tracking_configs[name]["data"]["particle_max_abs_eta"]
    pt_cut = tracking_configs[name]["data"]["particle_min_pt"]
    key_mode = "old" if name == "Paper" else None  # None or "old"
    particle_targets = ["pt", "eta", "phi", "vz"]
    print(f"Loading {name} model with PT > {pt_cut} and |eta| < {eta_cut}", f", Regression = {has_regression[name]}")
    tracking_results[name] = load_events(
        fname=fname,
        eta_cut=eta_cut,
        index_list=index_list,
        pt_cut=pt_cut,
        randomize=num_events,
        particle_targets=particle_targets,
        regression=has_regression[name],
        key_mode=key_mode,
    )

# ----------------------------------------------------
# Efficiency and fake rate plots
# ----------------------------------------------------

plot_fr = False
plot_trainings = {"Paper", "TRK-v0 0.9 GeV", "trackml 1 GeV", "trackml 0.6 GeV"}  # trainings to be included in plot
for qty in particle_targets:
    if qty not in {"pt", "eta", "phi", "vz"}:
        continue

    axlist = []
    if plot_fr:
        fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(12, 4), constrained_layout=True)
        axlist.extend([ax, ax1])
        ax1.set_xlabel(rf"Track ${qty_symbols[qty]}^\mathrm{{Reco}}$ {qty_units[qty]}")
        ax1.set_ylabel("Fake Rate")

    else:
        fig, ax = plt.subplots(ncols=1, figsize=(6, 4), constrained_layout=True)
        axlist.append(ax)

    names = []
    for name, (tracks, parts) in tracking_results.items():
        if name in plot_trainings:
            names.append(name)
            """Efficiency plots"""
            reconstructable = parts["reconstructable"]
            # double majority
            bin_count, bin_error = binned(
                parts["eff_dm"][reconstructable],
                parts["particle_" + qty][reconstructable],
                qty_bins[qty],
                underflow=True,
                overflow=True,
                binomial=False,
            )
            profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="solid")

            # perfect
            bin_count, bin_error = binned(
                parts["eff_perfect"][reconstructable],
                parts["particle_" + qty][reconstructable],
                qty_bins[qty],
                underflow=True,
                overflow=True,
                binomial=False,
            )
            profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="dotted")

            """Fake rate plots"""
            if "track_" + qty in tracks.columns and plot_fr:
                reconstructable = tracks["reconstructable"]
                # fake rate
                fakes = (~tracks["eff_dm"]) & (~tracks["duplicate"])
                bin_count, bin_error = binned(
                    fakes[reconstructable],
                    tracks["track_" + qty][reconstructable],
                    qty_bins[qty],
                    underflow=True,
                    overflow=True,
                    binomial=False,
                )
                profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax1, colour=training_colours[name], ls="solid")
                # duplicate
                bin_count, bin_error = binned(
                    tracks["duplicate"][reconstructable],
                    tracks["track_" + qty][reconstructable],
                    qty_bins[qty],
                    underflow=True,
                    overflow=True,
                    binomial=False,
                )
                profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax1, colour=training_colours[name], ls="dotted")

    # axis ranges
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("Efficiency")
    ax.set_xlabel(rf"Particle ${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")

    for i in axlist:
        i.grid(zorder=0, alpha=0.25, linestyle="--")
        if qty == "pt":
            i.set_xlim([0, 10.5])
            i.set_xticks(np.arange(start=2, stop=11, step=2))
        if qty == "eta":
            i.set_xlim([-4.5, 4.5])
            i.set_xticks(np.arange(start=-4, stop=4.5, step=1))
        if qty == "phi":
            i.set_xlim([-3.5, 3.5])
            i.set_xticks(np.arange(start=-3, stop=3.5, step=1))
        if qty == "vz":
            i.set_xlim([-112, 112])
            i.set_xticks(np.arange(start=-100, stop=110, step=25))

    # custom legends
    legend_elements_0 = [Line2D([0], [0], color=training_colours[training], label=training) for training in names]
    leg1_0 = ax.legend(handles=legend_elements_0, frameon=False, loc="upper left")
    ax.add_artist(leg1_0)

    legend_elements_eff = [Line2D([0], [0], color="black", label="DM"), Line2D([0], [0], color="black", ls="dotted", label="Perfect")]
    leg2_0 = ax.legend(handles=legend_elements_eff, frameon=False, loc="upper right")
    ax.add_artist(leg2_0)
    if plot_fr:
        leg1_1 = ax1.legend(handles=legend_elements_0, frameon=False, loc="upper left")
        ax1.add_artist(leg1_1)
        legend_elements_fake = [Line2D([0], [0], color="black", label="Fake"), Line2D([0], [0], color="black", ls="dotted", label="Duplicate")]
        leg2_1 = ax1.legend(handles=legend_elements_fake, frameon=False, loc="upper right")
        ax1.add_artist(leg2_1)
        if qty == "pt":
            axlist[1].set_ylim(0.0, 0.03)
        if qty == "phi":
            axlist[1].set_ylim(0.0, 0.02)
        if qty == "vz":
            axlist[1].set_ylim(0.0, 0.02)

    fig.savefig(out_dir + f"{qty}_eff.pdf")

# ----------------------------------------------------
# Regression residuals
# ----------------------------------------------------
plot_regression = True
plot_trainings = {"Paper", "TRK-v0 0.9 GeV"}  # trainings to be included in plot

if plot_regression:
    nbins = 55
    qty_res_bins = {
        "pt": np.linspace(-1, 1, nbins),
        "eta": np.linspace(-0.1, 0.1, nbins),
        "phi": np.linspace(-0.1, 0.1, nbins),
        "vz": np.linspace(-15, 15, nbins),
    }
    fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(10, 4)
    ax = ax.flatten()

    for i, qty in enumerate(["pt", "eta", "phi", "vz"]):
        labels = []
        colours = []
        for name, (tracks, _parts) in tracking_results.items():
            if name in plot_trainings:
                bins = qty_res_bins[qty]
                colour = training_colours[name]
                # track physicsal quantity regression predicted value
                tracks_qty = tracks["track_" + qty][tracks["eff_dm"] & tracks["reconstructable_parts"]]
                # particle physical quantity true value
                parts_qty = tracks["matched_" + qty][tracks["eff_dm"] & tracks["reconstructable_parts"]]
                res = tracks_qty - parts_qty

                label = hist_plot(xs=res, bins=bins, xrange=(bins[0], bins[-1]), name=name, axes=ax[i], colour=colour)
                labels.append(label)
                colours.append(colour)

        ax[i].grid(zorder=0, alpha=0.25, linestyle="--")
        ax[i].set_xlabel(rf"${qty_symbols[qty]}^\mathrm{{Reco}} - {qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")
        ax[i].set_ylabel("Density")

        ticks = None
        if qty in {"eta", "phi"}:
            ticks = np.arange(-0.1, 0.11, 0.05).round(2)
        if qty == "pt":
            ticks = np.arange(-1, 1.1, 0.5).round(2)
        if qty == "vz":
            ticks = np.arange(-15, 16, 5).round(2)

        ax[i].set_xticks(ticks)
        ax[i].set_xticklabels(ticks)
        legend_elements = [Line2D([0], [0], color=colours[j], label=labels[j]) for j in range(len(labels))]
        ax[i].legend(handles=legend_elements, frameon=False, loc="upper left", fontsize=8)

    fig.savefig(out_dir + "regr_residuals.pdf")

# ----------------------------------------------------
# Plots for the pt response vs the pt of the particle, and also vs the number of hits on the track
# ----------------------------------------------------

if plot_regression:
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    fig.set_size_inches(10, 3)
    qty = "pt"
    for name, (tracks, _parts) in tracking_results.items():
        if name in plot_trainings:
            parts_eff_qty = tracks["matched_" + qty][tracks["eff_dm"]]
            tracks_eff_qty = tracks["track_" + qty][tracks["eff_dm"]]
            n_assigned = tracks["n_pred_hits"][tracks["eff_dm"]]
            response = tracks_eff_qty / parts_eff_qty
            # response = np.clip(response, 0.5, 1.5)

            # make hist
            bins = qty_bins[qty]
            ys, ys_err = binned(response, parts_eff_qty, bins, underflow=True, overflow=True, binomial=False)
            profile_plot(ys, ys_err, bins, axes=ax[0], colour=training_colours[name], ls="solid", label=name)

            bins = np.linspace(3, 10, 8)
            ys, ys_err = binned(response, n_assigned, bins, binomial=False)
            profile_plot(ys, ys_err, bins, axes=ax[1], colour=training_colours[name], ls="solid", label=name)

    ax[0].set_xlabel(rf"${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")
    ax[0].set_ylabel(rf"${qty_symbols[qty]}^\mathrm{{Reco}}/{qty_symbols[qty]}^\mathrm{{True}}$")
    ax[0].grid(zorder=0, alpha=0.25, linestyle="--")
    ax[0].legend(frameon=False)
    ax[1].set_xlabel(r"Number of assigned hits")
    ax[1].set_ylabel(rf"${qty_symbols[qty]}^\mathrm{{Reco}}/{qty_symbols[qty]}^\mathrm{{True}}$")
    # ax[1].set_ylabel(rf"${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")
    ax[1].grid(zorder=0, alpha=0.25, linestyle="--")
    ax[1].legend(frameon=False)
    fig.savefig(out_dir + "pt_regr-nhits-response_paper.pdf")

# ----------------------------------------------------
# Efficiency and fake rate numbers
# ----------------------------------------------------

for name, (tracks, parts) in tracking_results.items():
    print(name)
    tgts = parts[parts.reconstructable]
    trks = tracks[tracks.reconstructable]
    # compute high pt integrated metrics
    high_pt_parts = tgts[tgts.particle_pt > 1.0]
    high_pt_parts_900 = tgts[tgts.particle_pt > 0.9]
    high_pt_eff = high_pt_parts.eff_dm.mean()
    high_pt_eff_900 = high_pt_parts_900.eff_dm.mean()
    high_pt_tracks = trks[trks.matched_pt > 1.0]
    high_pt_tracks_900 = trks[trks.matched_pt > 0.9]
    high_pt_fr = (~high_pt_tracks.eff_dm & ~trks.duplicate).mean()
    high_pt_fr_900 = (~high_pt_tracks_900.eff_dm & ~trks.duplicate).mean()

    # compute the overall fake rate
    integrated_fr = (~trks.eff_dm & ~trks.duplicate).mean()

    # print summary
    print(f"N events: {100 if num_events is None else num_events}, N particles: {len(parts)}, N tracks: {len(tracks)}")
    print(f"DM Integrated efficiency: {tgts.eff_dm.mean():.1%}")
    print(f"DM Efficiency for pT > 1.0 GeV: {high_pt_eff:.1%}")
    print(f"DM Efficiency for pT > 0.9 GeV: {high_pt_eff_900:.1%}")
    print()
    print(f"DM Integrated fake rate: {integrated_fr:.1%}")
    print(f"DM Fake rate for pT > 1.0 GeV: {high_pt_fr:.1%}")
    print(f"DM Fake rate for pT > 0.9 GeV: {high_pt_fr_900:.1%}")
    print()
    print(f"Perfect integrated Efficiency: {tgts.eff_perfect.mean():.1%}")
    print(f"Perfect Efficiency for pT > 1.0 GeV: {high_pt_parts.eff_perfect.mean():.1%}")
    print(f"Perfect Efficiency for pT > 0.9 GeV: {high_pt_parts_900.eff_perfect.mean():.1%}")
    print()
    print(f"Duplicate rate: {tracks.duplicate.mean():.1%}")
    print("\n")
