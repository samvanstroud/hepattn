import matplotlib.pyplot as plt
import numpy as np

from hepattn.experiments.trackml.data import TrackMLDataset

plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True


def plot_trackml_kinematics(dataset: TrackMLDataset):
    # Define histogram binnings
    qtys_bins = {
        "pt": np.linspace(0, 10, 64),
        "eta": np.linspace(-4, 4, 64),
        "num_hits": np.linspace(40000, 75000, 24),
    }

    pt_cuts = [0.0, 0.6, 0.75, 1.0]
    for pt_cut in pt_cuts:
        qtys_bins[f"num_particles_{pt_cut}gev"] = np.logspace(2, 4.2, 32)

    # Create the empty histograms that will be filled
    hists = {qty: np.zeros(len(bins) - 1) for qty, bins in qtys_bins.items()}

    # Fill te histogram from each event
    for event_idx in range(len(dataset)):
        inputs, targets = dataset[event_idx]

        # Pull the info we want from the event
        event_qtys = {
            "pt": targets["particle_pt"][0],
            "eta": targets["particle_eta"][0],
            "num_hits": inputs["hit_valid"][0].float().sum(-1),
        }

        for pt_cut in pt_cuts:
            event_qtys[f"num_particles_{pt_cut}gev"] = ((targets["particle_pt"][0] >= pt_cut).float().sum(-1),)

        # Fill the event info into the histograms
        for qty_name, qty_value in event_qtys.items():
            hist, _ = np.histogram(qty_value, bins=qtys_bins[qty_name])
            hists[qty_name] += hist

        print(f"Done event {event_idx}")

    # Specify which histograms we want to plot on which axes
    axes_spec = [
        (0, "pt", "Particle $p_T$", "black", None),
        (1, "eta", r"Particle $\eta$", "black", None),
        (2, "num_hits", "Number of Hits in Event", "black", None),
        (3, "num_particles_0.0gev", "Number of Particles in Event", "black", "All"),
        (3, "num_particles_1.0gev", "Number of Particles in Event", "mediumseagreen", "1 GeV"),
        (3, "num_particles_0.75gev", "Number of Particles in Event", "cornflowerblue", "750 MeV"),
        (3, "num_particles_0.6gev", "Number of Particles in Event", "mediumvioletred", "600 MeV"),
    ]

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(8, 3)
    ax = ax.flatten()

    for ax_idx, qty, alias, color, label in axes_spec:
        bin_centres = (qtys_bins[qty][1:] + qtys_bins[qty][:-1]) / 2
        ax[ax_idx].step(bin_centres, hists[qty], where="mid", color=color, label=label)
        ax[ax_idx].set_xlabel(alias)
        ax[ax_idx].set_ylabel("Count")

    for ax_idx in range(len(ax)):
        ax[ax_idx].grid(zorder=0, alpha=0.25, linestyle="--")

    ax[0].set_yscale("log")
    ax[3].set_xscale("log")

    fig.tight_layout()

    return fig
