import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import torch

plt.rcParams["figure.dpi"] = 300


def plot_odd_event(data, axes_spec, object_name, batch_idx=0, valid=True, mark_transparent=None, label_objects=False, gridspec_kw=None):
    # Setup the axes
    num_axes = len(axes_spec)

    fig, ax = plt.subplots(1, num_axes, gridspec_kw=gridspec_kw)
    fig.set_size_inches(8 * num_axes, 8)

    ax = [ax] if num_axes == 1 else ax.flatten()

    # Setup the color cycler that will be used
    colormap = plt.cm.tab20
    cycler = [colormap(i) for i in range(colormap.N)]

    sihit_names = ["sihit"]
    calohit_names = ["calohit"]

    for ax_idx, ax_spec in enumerate(axes_spec):
        # Plot only the hits / subsystems specified for these axes
        for input_name in ax_spec["input_names"]:
            x = data[f"{input_name}_{ax_spec['x']}"][batch_idx]
            y = data[f"{input_name}_{ax_spec['y']}"][batch_idx]

            ax[ax_idx].scatter(x, y, alpha=0.25, s=1.0, color="black")

            num_object_slots = data[f"{object_name}_{input_name}_valid"][batch_idx].shape[-2]

            for object_idx in range(num_object_slots):
                # Plots invalid particle if valid set to be False
                if data[f"{object_name}_valid"][batch_idx][object_idx].item() == valid:
                    color = cycler[object_idx % len(cycler)]
                    mask = data[f"{object_name}_{input_name}_valid"][batch_idx][object_idx]

                    alpha = 1.0
                    linestyle = "-"

                    if mark_transparent is not None:  # noqa: SIM102
                        if not data[f"{object_name}_{mark_transparent}"][batch_idx][object_idx].item():
                            alpha = 0.5
                            linestyle = ":"

                    # Tracker hits
                    if input_name in sihit_names:
                        # Used for sorting the hits in time when we want to plot them in order in the tracker
                        idx = torch.argsort(data[f"{input_name}_time"][batch_idx][mask], dim=-1)

                        ax[ax_idx].plot(x[mask][idx], y[mask][idx], color=color, marker="o", alpha=alpha, linewidth=1.0, ms=2.0, linestyle=linestyle)

                    # Calo hits
                    elif input_name in calohit_names:
                        energy = torch.sqrt(1e6 * data[f"{input_name}_total_energy"][batch_idx])
                        ax[ax_idx].scatter(x[mask], y[mask], color=color, marker=".", alpha=0.6, s=energy[mask])
                


            ax[ax_idx].set_xlabel(ax_spec.get("xlabel", ax_spec["x"]))
            ax[ax_idx].set_ylabel(ax_spec.get("ylabel", ax_spec["y"]))
            # ax[ax_idx].set_aspect("equal", "box")

    return fig
