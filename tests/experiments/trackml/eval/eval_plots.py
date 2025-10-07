from pathlib import Path

import matplotlib.pyplot as plt
from eval_tracking_metrics_old import eval_events
from plots import profile_plot

plt.rcParams["figure.dpi"] = 150


fnames = {
    "Paper": "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn-scale-up/epoch=028-val_loss=1.29786__test_test.h5",
    "600 MeV": "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn-scale-up/logs/TRK-ISAMBARD-180925-epochs30-LCA-eta2p5-pt600_20250918-T114424/ckpts/epoch=029-val_loss=0.41821_test_eval.h5",
}
EVAL_SET = "test"
REGRESSION = False
# if EVAL_SET == "test":
#     event_idx_start = 0

### ----------------------------------------------------
PT_CUT = 1.0  # don't change this, just for the pT > 1 GeV number
ETA_CUT = 2.5
NUM_EVENTS = 10
HIST_N_BINS = 40
### ----------------------------------------------------
results = {}

for name, fname in fnames.items():
    if name == "Paper":
        key_mode = "old"
        event_idx_start = 0
    else:
        key_mode = "new"
        event_idx_start = 29800
    print(f"Running {name}")
    parts, tracks = eval_events(fname, num_events=NUM_EVENTS, eta_cut=ETA_CUT, key_mode=key_mode, event_idx_start=event_idx_start)
    results[name] = (parts, tracks)
out_dir = Path(__file__).parent / "paperplots"
out_dir.mkdir(exist_ok=True)
(out_dir / "reg").mkdir(exist_ok=True)
print(f"Saving plots to {out_dir}")
### ----------------------------------------------------
COLOURS = {
    "Paper": "tab:orange",
    "600 MeV": "tab:blue",
    "750 MeV": "tab:orange",
    "1 GeV": "tab:green",
}


# ----------------------------------------------------
# Efficiency and fake rate plots
# ----------------------------------------------------
ptbins = [0.6, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10]
ptbins = [0.6, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
plt.figure(figsize=(10, 4), constrained_layout=True)
plt.subplot(121)
for name, (parts, _) in results.items():
    c = COLOURS[name]
    tgts = parts[parts.reconstructable]
    profile_plot(tgts.pt, tgts.eff_dm.astype(int), ptbins, underflow=False, label=name, binomial=True, colour=c)
    profile_plot(tgts.pt, tgts.eff_perfect.astype(int), ptbins, colour=c, underflow=False, binomial=True, ls="dashed")
plt.legend(frameon=False)
plt.xlabel("Matched Particle $p_T$ [GeV]")
plt.ylabel("Efficiency")

if REGRESSION:
    plt.subplot(122)
    for name, (parts, tracks) in results.items():
        c = COLOURS[name]
        tgts = parts[parts.reconstructable]
        profile_plot(tracks.pt, ~tracks.eff_dm, ptbins, underflow=False, label=name, binomial=True, colour=c)
    plt.legend(frameon=False)
    plt.xlabel("Reconstructed Particle $p_T$ [GeV]")
    plt.ylabel("Fake Rate")
plt.savefig(out_dir / "pt_eff_fr.png")
plt.close()
