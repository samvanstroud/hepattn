import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import torch

plt.rcParams["figure.dpi"] = 300


def plot_cld_event_reconstruction(inputs, reconstruction, axes_spec, object_name="particle", batch_idx=None, valid=True):
    num_axes = len(axes_spec)

    fig, ax = plt.subplots(1, num_axes)
    fig.set_size_inches(8 * num_axes, 8)

    ax = [ax] if num_axes == 1 else ax.flatten()

    colormap = plt.cm.tab20
    cycler = [colormap(i) for i in range(colormap.N)]

    if batch_idx is None:
        batch_idx = torch.argmax(reconstruction["particle_valid"].sum(-1))

    sihit_names = ["vtb", "vte", "itb", "ite", "otb", "ote", "sihit", "vtxd", "trkr"]
    ecal_names = ["ecb", "ece", "ecal"]
    hcal_names = ["hcb", "hce", "hcal"]

    for ax_idx, ax_spec in enumerate(axes_spec):
        for input_name in ax_spec["input_names"]:
            x = inputs[f"{input_name}_{ax_spec['x']}"][batch_idx]
            y = inputs[f"{input_name}_{ax_spec['y']}"][batch_idx]

            ax[ax_idx].scatter(x, y, alpha=0.25, s=1.0, color="black")

            num_particles = reconstruction[f"{object_name}_{input_name}_valid"][batch_idx].shape[-2]

            for mcparticle_idx in range(num_particles):
                # Plots invalid particle if valid set to be False
                if reconstruction[f"{object_name}_valid"][batch_idx][mcparticle_idx].item() == valid:
                    color = cycler[mcparticle_idx % len(cycler)]
                    mask = reconstruction[f"{object_name}_{input_name}_valid"][batch_idx][mcparticle_idx]

                    # Tracker hit
                    if input_name in sihit_names:
                        # Used for sorting the hits in time when we want to plot them in order in the tracker
                        idx = torch.argsort(inputs[f"{input_name}_time"][batch_idx][mask], dim=-1)

                        ax[ax_idx].plot(x[mask][idx], y[mask][idx], color=color, marker="o", alpha=0.75, linewidth=1.0, ms=2.0)

                        # Uncomment to leave a box denoting particle index for trkr hit
                        # if input_name == "trkr" and mask.any():
                        #     end_x = x[mask][idx][-1].item()
                        #     end_y = y[mask][idx][-1].item()
                        #     ax[ax_idx].text(
                        #         end_x,
                        #         end_y,
                        #         str(mcparticle_idx),
                        #         fontsize=5,
                        #         color="black",
                        #         ha="center",
                        #         va="center",
                        #         bbox={
                        #             "boxstyle": "round,pad=0.2",
                        #             "facecolor": "white",
                        #             "edgecolor": "black",
                        #             "linewidth": 0.5,
                        #             "alpha": 0.5,
                        #         },
                        #     )

                    # ECAL hit
                    elif input_name in ecal_names:
                        ax[ax_idx].scatter(x[mask], y[mask], color=color, marker=".", alpha=0.5, s=1.0)

                    # HCAL hit
                    elif input_name in hcal_names:
                        ax[ax_idx].scatter(x[mask], y[mask], color=color, marker="s", alpha=0.5, s=4.0)

                    # Muon hit
                    elif input_name == "muon":
                        ax[ax_idx].scatter(x[mask], y[mask], color=color, marker="x", alpha=0.75, s=4.0)

            ax[ax_idx].set_xlabel(ax_spec["x"])
            ax[ax_idx].set_ylabel(ax_spec["y"])
            ax[ax_idx].set_aspect("equal", "box")

    return fig


def _plot_matched_particle(
    ax, input_name, mc_idx, batch_idx, truth, inputs, base_color, sihit_names, ecal_names, hcal_names, spec, mode="preds", object_name="particle"
):
    mask = truth[f"{object_name}_{input_name}_valid"][batch_idx][mc_idx]
    x_hits = inputs[f"{input_name}_{spec['x']}"][batch_idx][mask]
    y_hits = inputs[f"{input_name}_{spec['y']}"][batch_idx][mask]

    if mode == "preds":
        linestyle = "-"
        si_marker = "o"
        ecal_marker = "."
        hcal_marker = "s"
        alpha = 0.9
    else:
        linestyle = "--"
        si_marker = "x"
        ecal_marker = "x"
        hcal_marker = "x"
        alpha = 0.7

    arrow_scale = {"vtxd": 100, "trkr": 4}

    if input_name in sihit_names and mask.any():
        px_hits = truth[f"{object_name}_{input_name}_{spec['px']}"][batch_idx][mc_idx][mask]
        py_hits = truth[f"{object_name}_{input_name}_{spec['py']}"][batch_idx][mc_idx][mask]

        times = inputs[f"{input_name}_time"][batch_idx][mask]
        idx = torch.argsort(times, dim=-1)
        ax.plot(
            x_hits[idx],
            y_hits[idx],
            color=base_color,
            linestyle=linestyle,
            marker=si_marker,
            markersize=2.5,
            linewidth=1.0,
            alpha=alpha,
            label=f"mc_idx:{mc_idx}",
        )

        m = torch.sqrt(px_hits[idx] ** 2 + py_hits[idx] ** 2).clamp(min=1e-6)
        dx = px_hits[idx] / m
        dy = py_hits[idx] / m
        ax.quiver(
            x_hits[idx], y_hits[idx], dx, dy, angles="xy", scale_units="xy", scale=arrow_scale[input_name], color=base_color, width=0.002, alpha=alpha
        )

        end_x = x_hits[idx][-1].item()
        end_y = y_hits[idx][-1].item()
        ax.text(
            end_x,
            end_y,
            str(mc_idx),
            fontsize=5,
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.5,
                "alpha": 0.5,
            },
        )

    elif input_name in ecal_names:
        ax.scatter(x_hits, y_hits, color=base_color, marker=ecal_marker, alpha=alpha, s=3.0)

    elif input_name in hcal_names:
        ax.scatter(x_hits, y_hits, color=base_color, marker=hcal_marker, alpha=alpha, s=6.0)

    elif input_name == "muon":
        ax.scatter(x_hits, y_hits, color=base_color, marker="h", alpha=alpha, s=6.0)


def _plot_mismatched_particle(
    ax, input_name, mc_idx, batch_idx, truth, preds, inputs, base_color, sihit_names, ecal_names, hcal_names, spec, object_name="particle"
):
    truth_mask = truth[f"{object_name}_{input_name}_valid"][batch_idx][mc_idx]
    pred_mask = preds[f"{object_name}_{input_name}_valid"][batch_idx][mc_idx]

    if truth_mask.any():
        t_x = inputs[f"{input_name}_{spec['x']}"][batch_idx][truth_mask]
        t_y = inputs[f"{input_name}_{spec['y']}"][batch_idx][truth_mask]

        linestyle = "-"
        alpha = 0.5
        if input_name in sihit_names:
            t_times = inputs[f"{input_name}_time"][batch_idx][truth_mask]
            tidx = torch.argsort(t_times, dim=-1)
            ax.plot(
                t_x[tidx],
                t_y[tidx],
                color=base_color,
                linestyle=linestyle,
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                alpha=alpha,
                label=f"mc_idx:{mc_idx}",
            )

        elif input_name in ecal_names:
            ax.scatter(t_x, t_y, color=base_color, marker=".", alpha=alpha, s=4.0)

        elif input_name in hcal_names:
            ax.scatter(t_x, t_y, color=base_color, marker="s", alpha=alpha, s=8.0)

        elif input_name == "muon":
            ax.scatter(t_x, t_y, color=base_color, marker="h", alpha=alpha, s=8.0)

    if pred_mask.any():
        p_x = inputs[f"{input_name}_{spec['x']}"][batch_idx][pred_mask]
        p_y = inputs[f"{input_name}_{spec['y']}"][batch_idx][pred_mask]

        linestyle = "--"
        marker = "x"
        alpha = 0.9
        if input_name in sihit_names:
            p_times = inputs[f"{input_name}_time"][batch_idx][pred_mask]
            pidx = torch.argsort(p_times, dim=-1)
            ax.plot(p_x[pidx], p_y[pidx], color=base_color, linestyle=linestyle, marker=marker, markersize=3.5, linewidth=1.0, alpha=alpha)

        elif input_name in ecal_names:
            ax.scatter(p_x, p_y, color=base_color, marker=marker, alpha=alpha, s=3.0)

        elif input_name in hcal_names:
            ax.scatter(p_x, p_y, color=base_color, marker=marker, alpha=alpha, s=6.0)

        elif input_name == "muon":
            ax.scatter(p_x, p_y, color=base_color, marker="H", alpha=alpha, s=6.0)

    if input_name == "trkr" and truth_mask.any():
        t_x = inputs[f"{input_name}_{spec['x']}"][batch_idx][truth_mask]
        t_y = inputs[f"{input_name}_{spec['y']}"][batch_idx][truth_mask]
        t_times = inputs[f"{input_name}_time"][batch_idx][truth_mask]
        tidx = torch.argsort(t_times, dim=-1)
        end_x = t_x[tidx][-1].item()
        end_y = t_y[tidx][-1].item()
        ax.text(
            end_x,
            end_y,
            str(mc_idx),
            fontsize=5,
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.5,
                "alpha": 0.5,
            },
        )


def plot_cld_event_match_vs_mismatch(inputs, truth, preds, axes_spec, object_name="particle", batch_idx=0):
    num_axes = len(axes_spec)
    fig, axes = plt.subplots(2, num_axes, squeeze=False)
    fig.set_size_inches(8 * num_axes, 16)

    colormap = plt.cm.tab20
    cycler = [colormap(i) for i in range(colormap.N)]

    sihit_names = ["vtb", "vte", "itb", "ite", "otb", "ote", "sihit", "vtxd", "trkr"]
    ecal_names = ["ecb", "ece", "ecal"]
    hcal_names = ["hcb", "hce", "hcal"]

    for col_idx, spec in enumerate(axes_spec):
        ax_matched = axes[0, col_idx]
        ax_mismatch = axes[1, col_idx]

        for name in spec["input_names"]:
            x_all = inputs[f"{name}_{spec['x']}"][batch_idx]
            y_all = inputs[f"{name}_{spec['y']}"][batch_idx]
            ax_matched.scatter(x_all, y_all, color="black", alpha=0.5, s=1.0)
            ax_mismatch.scatter(x_all, y_all, color="black", alpha=0.5, s=1.0)

        input_names = spec["input_names"][0]
        truth_mask_all = truth[f"{object_name}_{input_names}_valid"][batch_idx]
        num_particles = truth_mask_all.shape[0]

        for mc_idx in range(num_particles):
            base_color = cycler[mc_idx % len(cycler)]

            for name in spec["input_names"]:
                truth_mask = truth[f"{object_name}_{name}_valid"][batch_idx][mc_idx]
                pred_mask = preds[f"{object_name}_{name}_valid"][batch_idx][mc_idx]
                if not (truth_mask.any() or pred_mask.any()):
                    continue

                if torch.equal(truth_mask, pred_mask):
                    _plot_matched_particle(
                        ax_matched, name, mc_idx, batch_idx, truth, inputs, base_color, sihit_names, ecal_names, hcal_names, spec, object_name
                    )
                else:
                    _plot_mismatched_particle(
                        ax_mismatch, name, mc_idx, batch_idx, truth, preds, inputs, base_color, sihit_names, ecal_names, hcal_names, spec, object_name
                    )

        ax_matched.set_xlabel(spec["x"])
        ax_matched.set_ylabel(spec["y"])
        ax_matched.set_aspect("equal", "box")
        ax_matched.set_title("Matched Tracks")
        ax_matched.legend(fontsize=5, framealpha=0.5)

        ax_mismatch.set_xlabel(spec["x"])
        ax_mismatch.set_ylabel(spec["y"])
        ax_mismatch.set_aspect("equal", "box")
        ax_mismatch.set_title("Mismatched Tracks")
        ax_mismatch.legend(fontsize=5, framealpha=0.5)

    line_truth = mlines.Line2D([], [], color="gray", linestyle="-", marker="o", label="Truth")
    line_pred = mlines.Line2D([], [], color="gray", linestyle="--", marker="x", label="Prediction")
    fig.legend(handles=[line_truth, line_pred], loc="upper right")

    return fig


def _plot_matched_processed_particle(
    ax,
    input_name,
    mc_idx,
    batch_idx,
    orig_targets,
    post_targets,
    inputs_orig,
    inputs_post,
    base_color,
    sihit_names,
    ecal_names,
    hcal_names,
    post_idx,
    spec,
    object_name="particle",
):
    mask_pre = orig_targets[f"{object_name}_{input_name}_valid"][batch_idx][mc_idx]
    x_pre = inputs_orig[f"{input_name}_{spec['x']}"][batch_idx][mask_pre]
    y_pre = inputs_orig[f"{input_name}_{spec['y']}"][batch_idx][mask_pre]

    mask_post = post_targets[f"{object_name}_{input_name}_valid"][batch_idx][post_idx]
    x_post = inputs_post[f"{input_name}_{spec['x']}"][batch_idx][mask_post]
    y_post = inputs_post[f"{input_name}_{spec['y']}"][batch_idx][mask_post]

    linestyle_pre = "--"
    linestyle_post = "-"
    alpha_pre = 0.5
    alpha_post = 0.9
    marker_pre = "x"

    arrow_scale = {"vtxd": 100, "trkr": 4}

    if input_name in sihit_names and mask_pre.any():
        px_pre = orig_targets[f"{object_name}_{input_name}_{spec['px']}"][batch_idx][mc_idx][mask_pre]
        py_pre = orig_targets[f"{object_name}_{input_name}_{spec['py']}"][batch_idx][mc_idx][mask_pre]

        px_post = post_targets[f"{object_name}_{input_name}_{spec['px']}"][batch_idx][post_idx][mask_post]
        py_post = post_targets[f"{object_name}_{input_name}_{spec['py']}"][batch_idx][post_idx][mask_post]

        t_pre = inputs_orig[f"{input_name}_time"][batch_idx][mask_pre]
        idx_pre = torch.argsort(t_pre, dim=-1)
        ax.plot(
            x_pre[idx_pre],
            y_pre[idx_pre],
            color=base_color,
            linestyle=linestyle_pre,
            linewidth=0.5,
            marker=marker_pre,
            markersize=3.5,
            alpha=alpha_pre,
            label=f"mc_idx:{mc_idx} ({post_idx})",
        )
        end_x = x_pre[idx_pre][-1].item()
        end_y = y_pre[idx_pre][-1].item()
        ax.text(
            end_x,
            end_y,
            str(mc_idx),
            fontsize=5,
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "linestyle": "--",
                "linewidth": 0.5,
                "alpha": 0.5,
            },
        )
        m_pre = torch.sqrt(px_pre[idx_pre] ** 2 + py_pre[idx_pre] ** 2).clamp(min=1e-6)
        ax.quiver(
            x_pre[idx_pre],
            y_pre[idx_pre],
            (px_pre[idx_pre] / m_pre),
            (py_pre[idx_pre] / m_pre),
            angles="xy",
            scale_units="xy",
            scale=arrow_scale[input_name],
            color=base_color,
            width=0.002,
            alpha=alpha_pre,
        )

        t_post = inputs_post[f"{input_name}_time"][batch_idx][mask_post]
        idx_post = torch.argsort(t_post, dim=-1)
        ax.plot(
            x_post[idx_post],
            y_post[idx_post],
            color=base_color,
            linestyle=linestyle_post,
            linewidth=1.0,
            marker="o",
            markersize=2.5,
            alpha=alpha_post,
        )
        end_x = x_post[idx_post][-1].item()
        end_y = y_post[idx_post][-1].item()
        ax.text(
            end_x,
            end_y,
            str(mc_idx),
            fontsize=5,
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.5,
                "alpha": 0.5,
            },
        )
        m_post = torch.sqrt(px_post[idx_post] ** 2 + py_post[idx_post] ** 2).clamp(min=1e-6)
        ax.quiver(
            x_post[idx_post],
            y_post[idx_post],
            (px_post[idx_post] / m_post),
            (py_post[idx_post] / m_post),
            angles="xy",
            scale_units="xy",
            scale=arrow_scale[input_name],
            color=base_color,
            width=0.002,
            alpha=alpha_post,
        )

    elif input_name in ecal_names:
        ax.scatter(x_pre, y_pre, color=base_color, marker=marker_pre, alpha=alpha_pre, s=4.0)
        ax.scatter(x_post, y_post, color=base_color, marker=".", alpha=alpha_post, s=3.0)

    elif input_name in hcal_names:
        ax.scatter(x_pre, y_pre, color=base_color, marker=marker_pre, alpha=alpha_pre, s=8.0)
        ax.scatter(x_post, y_post, color=base_color, marker="s", alpha=alpha_post, s=6.0)

    elif input_name == "muon":
        ax.scatter(x_pre, y_pre, color=base_color, marker="H", alpha=alpha_pre, s=8.0)
        ax.scatter(x_post, y_post, color=base_color, marker="h", alpha=alpha_post, s=6.0)


def plot_cld_event_pre_vs_post(inputs_orig, inputs_post, orig_targets, post_targets, axes_spec, object_name="particle", batch_idx=0):
    num_axes = len(axes_spec)
    fig, axes = plt.subplots(2, num_axes, squeeze=False)
    fig.set_size_inches(8 * num_axes, 16)

    colormap = plt.cm.tab20
    cycler = [colormap(i) for i in range(colormap.N)]

    sihit_names = ["vtb", "vte", "itb", "ite", "otb", "ote", "sihit", "vtxd", "trkr"]
    ecal_names = ["ecb", "ece", "ecal"]
    hcal_names = ["hcb", "hce", "hcal"]

    fields = ["mom.x", "mom.y", "mom.z", "vtx.x", "vtx.y", "vtx.z"]
    orig_list = [orig_targets[f"{object_name}_{x}"][batch_idx] for x in fields]
    post_list = [post_targets[f"{object_name}_{x}"][batch_idx] for x in fields]
    orig_arr = torch.stack(orig_list, dim=1)
    post_arr = torch.stack(post_list, dim=1)

    eq_matrix = orig_arr.unsqueeze(1) == post_arr.unsqueeze(0)
    row_matched = eq_matrix.all(dim=2)
    matched_mask = row_matched.any(dim=1)

    n_orig = orig_arr.shape[0]
    match_idx = torch.full((n_orig,), -1)
    for i in range(n_orig):
        if matched_mask[i]:
            match_idx[i] = torch.nonzero(row_matched[i])[0]

    num_particles = orig_targets[f"{object_name}_valid"].shape[-1]

    for col_idx, spec in enumerate(axes_spec):
        ax_matched = axes[0, col_idx]
        ax_mismatch = axes[1, col_idx]

        for name in spec["input_names"]:
            x_all = inputs_orig[f"{name}_{spec['x']}"][batch_idx]
            y_all = inputs_orig[f"{name}_{spec['y']}"][batch_idx]
            ax_matched.scatter(x_all, y_all, color="black", alpha=0.5, s=1.0)
            ax_mismatch.scatter(x_all, y_all, color="black", alpha=0.5, s=1.0)

        for mc_i in range(num_particles):
            # if mc_i not in [18,41,63,120,143]:
            #     continue

            base_color = cycler[mc_i % len(cycler)]
            post_i = match_idx[mc_i].item()

            if post_i >= 0:
                for name in spec["input_names"]:
                    # Uncomment to plot only modified truth tracks for matched particles
                    # mask_pre = orig_targets[f"{object_name}_{name}_valid"][batch_idx][mc_i]
                    # x_pre = inputs_orig[f"{name}_{spec['x']}"][batch_idx][mask_pre]
                    # mask_post = post_targets[f"{object_name}_{name}_valid"][batch_idx][post_i]
                    # x_post = inputs_post[f"{name}_{spec['x']}"][batch_idx][mask_post]
                    # if x_pre.shape == x_post.shape:
                    #     continue

                    _plot_matched_processed_particle(
                        ax_matched,
                        name,
                        mc_i,
                        batch_idx,
                        orig_targets,
                        post_targets,
                        inputs_orig,
                        inputs_post,
                        base_color,
                        sihit_names,
                        ecal_names,
                        hcal_names,
                        post_i,
                        spec,
                        object_name,
                    )
                ax_matched.set_title("Matched Particles")
                ax_matched.set_xlabel(spec["x"])
                ax_matched.set_ylabel(spec["y"])
                ax_matched.set_aspect("equal", "box")
            else:
                for name in spec["input_names"]:
                    _plot_matched_particle(
                        ax_mismatch,
                        name,
                        mc_i,
                        batch_idx,
                        orig_targets,
                        inputs_orig,
                        base_color,
                        sihit_names,
                        ecal_names,
                        hcal_names,
                        spec,
                        mode="preprocess",
                        object_name=object_name,
                    )
                ax_mismatch.set_title("Dropped Particles")
                ax_mismatch.set_xlabel(spec["x"])
                ax_mismatch.set_ylabel(spec["y"])
                ax_mismatch.set_aspect("equal", "box")

        ax_matched.legend(fontsize=5, framealpha=0.5)
        ax_mismatch.legend(fontsize=5, framealpha=0.5)

    line_orig = mlines.Line2D([], [], color="gray", linestyle="--", marker="x", label="Original")
    line_pre = mlines.Line2D([], [], color="gray", linestyle="-", marker="o", label="After cuts")
    fig.legend(handles=[line_orig, line_pre], loc="upper right")

    return fig
