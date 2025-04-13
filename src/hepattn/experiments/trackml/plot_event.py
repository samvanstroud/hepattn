import matplotlib.pyplot as plt


def plot_trackml_event_reconstruction(inputs, reconstruction, x_coord="x", y_coord="y"):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)

    batch_idx = 0

    hit_particle_valid = reconstruction["hit_on_valid_particle"][batch_idx]

    ax.scatter(
        inputs[f"hit_{x_coord}"][batch_idx][hit_particle_valid],
        inputs[f"hit_{y_coord}"][batch_idx][hit_particle_valid],
        color="black",
        s=1.0,
        alpha=0.5,
    )

    for i in range(reconstruction["particle_valid"][batch_idx].shape[-1]):
        if not reconstruction["particle_valid"][batch_idx][i]:
            continue

        if not reconstruction["particle_pt"][batch_idx][i] > 1.0:
            continue

        particle_mask = reconstruction["particle_hit_valid"][batch_idx][i]
        ax.plot(inputs[f"hit_{x_coord}"][batch_idx][particle_mask], inputs[f"hit_{y_coord}"][batch_idx][particle_mask], alpha=0.5)

    return fig
