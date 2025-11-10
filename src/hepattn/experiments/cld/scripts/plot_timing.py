from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Specify which eval you want to plot here
times_path = Path("/share/rcifdata/maxhart/hepattn/logs/CLD_8_320_10MeV_neutrals_muon_20250809-T183715/times/CLD_8_320_10MeV_neutrals_muon_times.npy")
dims_path = Path("/share/rcifdata/maxhart/hepattn/logs/CLD_8_320_10MeV_neutrals_muon_20250809-T183715/times/CLD_8_320_10MeV_neutrals_muon_dims.npy")
plot_save_path = Path("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/eval_plots")

# Load in the data
dims = np.load(dims_path, allow_pickle=True)[()]
times = np.load(times_path)

# We need to count up the hits from each different subsystem
hit_counts = []
for i in range(len(dims["sihit_valid"])):
    hit_count = 0
    for k in ["sihit", "ecal", "hcal", "muon"]:
        hit_count += dims[f"{k}_valid"][i, 1]
    hit_counts.append(hit_count)

# Allow some time for warmup
warmup_time = 100
hit_counts = np.array(hit_counts)[warmup_time:]
times = times[warmup_time:]

# First plot a scatterplot of time against number of hits
fig, ax = plt.subplots()
fig.set_size_inches(6, 4)

ax.scatter(hit_counts, times, color="black", marker="+", alpha=0.25)
ax.set_ylabel("Inference Time [ms]")
ax.set_xlabel("Total Number of Hits in Event")
ax.grid(zorder=0, alpha=0.25, linestyle="--")

fig.tight_layout()
fig.savefig(plot_save_path / Path("timing.png"))

# Also plot the times against their number in the batch
# This will allow us to check if we are warmed up
fig, ax = plt.subplots()
fig.set_size_inches(6, 4)

ax.plot(times, linestyle="none", color="black", marker="+")

fig.savefig(plot_save_path / Path("timing_dep.png"))
