import matplotlib.pyplot as plt


def plot_itk_event_reconstruction(inputs, reconstruction, x_coord="x", y_coord="y"):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)

    batch_idx = 0

    colormap = plt.cm.tab10
    cycler = [colormap(i) for i in range(colormap.N)]

    for hit in ["pixel", "strip"]:
        hit_particle_valid = reconstruction[f"{hit}_on_valid_particle"][batch_idx]

        ax.scatter(
            inputs[f"{hit}_{x_coord}"][batch_idx][hit_particle_valid],
            inputs[f"{hit}_{y_coord}"][batch_idx][hit_particle_valid],
            color="black",
            s=1.0,
            alpha=0.5,
        )

        for particle_idx in range(reconstruction["particle_valid"][batch_idx].shape[-1]):
            if not reconstruction["particle_valid"][batch_idx][particle_idx]:
                continue

            color = cycler[particle_idx % len(cycler)]
            particle_mask = reconstruction[f"particle_{hit}_valid"][batch_idx][particle_idx]

            ax.plot(
                inputs[f"{hit}_{x_coord}"][batch_idx][particle_mask], inputs[f"{hit}_{y_coord}"][batch_idx][particle_mask], alpha=0.5, color=color
            )

    return fig
