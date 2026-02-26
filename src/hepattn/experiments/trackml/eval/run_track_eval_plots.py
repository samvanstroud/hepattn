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
    "Tracking DQ": "tab:orange",  # |eta| < 4.0
    "LCA": "tab:blue",
}

qty_bins = {
    "pt": np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10]),
    # "eta": np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]),
    "eta": np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
    "phi": np.array([-math.pi, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, math.pi]),
}

qty_symbols = {"pt": "p_\\mathrm{T}", "eta": "\\eta", "phi": "\\phi"}
qty_units = {"pt": "[GeV]", "eta": "", "phi": ""}
out_dir = "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/test/"

# ----------------------------------------------------
# Read configuration file information
# ----------------------------------------------------

tracking_fnames = {
    "Paper": "/share/rcif2/pduckett/hepattn-trahxam-old/src/hepattn/experiments/trackml/eval/paper_eval/epoch=028-val_loss=1.29786__test_test.h5",
    "Tracking DQ": "/share/rcif2/pduckett/hepattn-trahxam-old/logs/TRK-v8-3l-fast-fix-vec-threeshold-0p01_20260208-T135534/ckpts/epoch=009-val_loss=0.27284_test_eval.h5",
    "LCA": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-original-dq-lca_20260222-T012055/ckpts/epoch=009-val_loss=0.28793_test_eval.h5",
}

tracking_config_fname = {
    "Paper": "/share/rcif2/pduckett/hepattn-trahxam-old/src/hepattn/experiments/trackml/configs/tracking-threshold.yaml",
    "Tracking DQ": "/share/rcif2/pduckett/hepattn-trahxam-old/src/hepattn/experiments/trackml/configs/tracking-threshold.yaml",
    "LCA": "/share/rcif2/pduckett/hepattn-trahxam-old/src/hepattn/experiments/trackml/configs/tracking-threshold.yaml",
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
num_events = 10
track_valid_threshold = 0.5
iou_threshold = 0.5

# Per-file hit-order handling for track evaluation:
# - auto: infer best alignment (recommended default)
# - as_saved: do not reorder
# - unsort_preds: unsort predicted masks back to input order
# - sort_targets: sort targets to prediction order
hit_order_by_name = {
    "Paper": "as_saved",
    "Tracking DQ": "as_saved",
    "LCA": "auto",
}

index_list = None  # ["event_0", ..., "event_99"] or ["29800", ..., "29899"]
for name, fname in tracking_fnames.items():
    eta_cut = tracking_configs[name]["data"]["particle_max_abs_eta"]
    pt_cut = tracking_configs[name]["data"]["particle_min_pt"]
    key_mode = "old" if name == "Paper" else None  # None or "old"
    particle_targets = ["pt", "eta", "phi"]
    print(f"Loading {name} model with PT > {pt_cut} and |eta| < {eta_cut}", f"")
    tracking_results[name] = load_events(
        fname=fname,
        eta_cut=eta_cut,
        index_list=index_list,
        pt_cut=pt_cut,
        randomize=num_events,
        particle_targets=particle_targets,
        regression=False,
        key_mode=key_mode,
        track_valid_threshold=track_valid_threshold,
        iou_threshold=iou_threshold,
        hit_order_mode=hit_order_by_name.get(name, "auto"),
    )

print("loaded events")

# ----------------------------------------------------
# Efficiency and fake rate plots
# ----------------------------------------------------

for qty in particle_targets:
    if qty not in {"pt", "eta", "phi"}:
        continue

    axlist = []
    fig, ax = plt.subplots(ncols=1, figsize=(6, 4), constrained_layout=True)
    axlist.append(ax)

    names = []
    for name, (tracks, parts) in tracking_results.items():
        if name in tracking_fnames:
            names.append(name)
            """Efficiency plots"""
            reconstructable = parts["reconstructable"]
            # double majority
            bin_count, bin_error = binned(
                parts["eff_dm"][reconstructable],
                parts["particle_" + qty][reconstructable],
                qty_bins[qty],
                underflow=True,
                overflow=False,
                binomial=False,
            )
            profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="solid")

            # perfect
            bin_count, bin_error = binned(
                parts["eff_perfect"][reconstructable],
                parts["particle_" + qty][reconstructable],
                qty_bins[qty],
                underflow=True,
                overflow=False,
                binomial=False,
            )
            profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="dotted")

    print("got this far")
    # axis ranges
    ax.set_ylim(0.8, 1.04)
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

    # custom legends
    legend_elements_0 = [Line2D([0], [0], color=training_colours[training], label=training) for training in names]
    leg1_0 = ax.legend(handles=legend_elements_0, frameon=False, loc="upper left")
    ax.add_artist(leg1_0)

    legend_elements_eff = [Line2D([0], [0], color="black", label="DM"), Line2D([0], [0], color="black", ls="dotted", label="Perfect")]
    leg2_0 = ax.legend(handles=legend_elements_eff, frameon=False, loc="upper right")
    ax.add_artist(leg2_0)

    fig.savefig(out_dir + f"{qty}_eff.png")
    plt.close(fig)


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
