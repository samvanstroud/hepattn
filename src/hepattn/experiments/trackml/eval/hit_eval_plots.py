import h5py
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml
from hit_evaluate import load_events
from plot_utils import binned, profile_plot

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
    "600 MeV": "mediumvioletred",
    "750 MeV": "cornflowerblue",
    # "1 GeV": "mediumseagreen", # |eta| < 2.5
    "0.9 GeV": "mediumseagreen",  # |eta| < 4.0
}

qty_bins = {
    "pt": np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10]),
    # "eta": np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]),
    "eta": np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
    "phi": np.array([-3.14, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, 3.14]),
    "vz": np.array([-100, -50, -20, -10, 0, 10, 20, 50, 100]),
}

qty_symbols = {"pt": "p_\\mathrm{T}", "eta": "\\eta", "phi": "\\phi", "vz": "v_z"}
qty_units = {"pt": "[GeV]", "eta": "", "phi": "", "vz": "[mm]"}
out_dir = "plots"

# ----------------------------------------------------
# Read configuration file information
# ----------------------------------------------------

with open("/data/atlas/users/slin/myHepattn/hepattn/src/hepattn/experiments/trackml/configs/filtering.yaml", "r") as f:
    fconfig= yaml.safe_load(f)

filtering_configs = {
    "0.9 GeV": fconfig
}

filtering_fnames = {
    "0.9 GeV": "/data/atlas/users/slin/myHepattn/hepattn/src/hepattn/experiments/trackml/logs/HC-v3_20250818-T211321/ckpts/epoch=029-val_loss=0.34545_test_eval.h5"
}

# filter_inputs = ["hits_"+filtering_configs["0.9 GeV"]["data"]["inputs"]["hit"][i] for i in range(len(filtering_configs["0.9 GeV"]["data"]["inputs"]["hit"]))]

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

filtering_results = {}
num_events=None
for name, fname in filtering_fnames.items():
    filter_threshold = filtering_configs[name]["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"][0]["init_args"]["threshold"]
    filtering_results[name] = load_events(fname=fname, randomize=num_events, write_inputs=None, write_parts=True, threshold=filter_threshold)

# ----------------------------------------------------
# Efficiency-Purity curve and PT binned efficiency plot
# ----------------------------------------------------

fig, ax = plt.subplots(ncols=2, figsize=(10,3), constrained_layout=True)
for name, (hits, targets, parts, metrics) in filtering_results.items():
    ax[0].plot(metrics["roc_eff"], metrics["roc_pur"],
            color=training_colours[name],
            label="%s %s\nAUC: %.4f"%(filtering_configs[name]["name"], name, metrics["roc_eff_pur_auc"])
           )
    thid = np.argmin(np.abs(metrics["roc_eff_pur_thr"] - 0.1))
    ax[0].scatter(metrics["roc_eff"][thid], metrics["roc_pur"][thid],
                  color=training_colours[name], s=100)

    # reconstructable particles must have >=3 hits
    reconstructable = np.where(parts["pred_hits"] >= 3, True, False)
    # apply valid_particle selection
    reconstructable = reconstructable & parts["valid"]
    # remove excess entries (particles in event less than n_max_particles)
    valid = ~np.isnan(parts["particle_pt"])
    bin_count, bin_error = binned(reconstructable[valid], parts["particle_pt"][valid], qty_bins["pt"])
    profile_plot(bin_count, bin_error, qty_bins["pt"], axes=ax[1], color=training_colours["0.9 GeV"], label="%s %s"%(filtering_configs[name]["name"], name))

ax[0].set_xlabel("Hit Efficiency")
ax[0].set_ylabel("Hit Purity")
ax[0].set_xlim(0.9,1.01)
ax[0].set_ylim(0.3, 1.01)
ax[0].grid(which="both")
ax[0].grid(zorder=0, alpha=0.25, linestyle="--")
ax[0].legend(loc=3)

ax[1].set_xlabel(r'Particle $%s$ %s'%(qty_symbols["pt"], qty_units["pt"]))
ax[1].set_ylabel('Reconstructable Particles')
ax[1].set_ylim(0.97,1)
ax[1].set_xticks(np.arange(start=2, stop=11, step=2))
ax[1].grid(which='both')
ax[1].grid(zorder=0, alpha=0.25, linestyle="--")
ax[1].legend(loc=3)

fig.savefig(out_dir + "/filter_response.pdf")
