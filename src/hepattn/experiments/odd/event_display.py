import matplotlib.pyplot as plt
import torch

plt.rcParams["figure.dpi"] = 300


def _csr_row_indices(indptr: torch.Tensor, indices: torch.Tensor, row: int) -> torch.Tensor:
    start = int(indptr[row].item())
    end = int(indptr[row + 1].item())
    return indices[start:end].to(torch.long)


def _plot_object_sihits(ax, data, object_name: str, x, y, batch_idx: int, cycler) -> None:
    indptr = data[f"{object_name}_sihit_indptr"][batch_idx]
    indices = data[f"{object_name}_sihit_indices"][batch_idx]
    object_valid = data[f"{object_name}_valid"][batch_idx]
    sihit_time = data["sihit_time"][batch_idx]
    num_object_slots = min(max(int(indptr.numel()) - 1, 0), int(object_valid.numel()))

    for object_idx_tensor in torch.nonzero(object_valid, as_tuple=False).flatten():
        object_idx = int(object_idx_tensor.item())
        if object_idx >= num_object_slots:
            continue

        hit_indices = _csr_row_indices(indptr, indices, object_idx)
        if hit_indices.numel() == 0:
            continue

        order = torch.argsort(sihit_time[hit_indices], dim=-1)
        ordered_hits = hit_indices[order]
        color = cycler[object_idx % len(cycler)]
        ax.plot(
            x[ordered_hits],
            y[ordered_hits],
            color=color,
            marker="o",
            alpha=1.0,
            linewidth=1.0,
            ms=2.0,
        )


def _plot_particle_calohits(ax, data, x, y, batch_idx: int, cycler) -> None:
    indptr = data["particle_calohit_indptr"][batch_idx]
    indices = data["particle_calohit_indices"][batch_idx]
    particle_valid = data["particle_valid"][batch_idx]
    calohit_energy = torch.sqrt(1e6 * data["calohit_total_energy"][batch_idx])
    num_particles = min(max(int(indptr.numel()) - 1, 0), int(particle_valid.numel()))

    for particle_idx_tensor in torch.nonzero(particle_valid, as_tuple=False).flatten():
        particle_idx = int(particle_idx_tensor.item())
        if particle_idx >= num_particles:
            continue

        hit_indices = _csr_row_indices(indptr, indices, particle_idx)
        if hit_indices.numel() == 0:
            continue

        color = cycler[particle_idx % len(cycler)]
        ax.scatter(
            x[hit_indices],
            y[hit_indices],
            color=color,
            marker=".",
            alpha=0.6,
            s=calohit_energy[hit_indices],
        )


def _build_helix_path(
    phi: torch.Tensor,
    eta: torch.Tensor,
    pt: torch.Tensor,
    charge_sign: torch.Tensor,
    d0: torch.Tensor,
    z0: torch.Tensor,
    magnetic_field_t: float,
    helix_radius_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    omega = charge_sign * ((0.3 * magnetic_field_t) / pt.clamp_min(1e-6))
    curvature_radius = pt / (0.3 * magnetic_field_t)
    max_transverse_len = min(2.0 * helix_radius_m, float((4.0 * torch.pi * curvature_radius).item()))
    max_transverse_len = max(max_transverse_len, 0.05)
    path_s = torch.linspace(0.0, max_transverse_len, 256, dtype=torch.float32, device=phi.device)

    x0 = -d0 * torch.sin(phi)
    y0 = d0 * torch.cos(phi)
    y_linear = y0 + path_s * torch.sin(phi)
    if abs(float(omega.item())) < 1e-6:
        x = x0 + path_s * torch.cos(phi)
        y = y0 + path_s * torch.sin(phi)
    else:
        x = x0 + (torch.sin(phi + omega * path_s) - torch.sin(phi)) / omega
        y = y0 - (torch.cos(phi + omega * path_s) - torch.cos(phi)) / omega
    z = z0 + path_s * torch.sinh(eta)

    # Plot radius is cylindrical (transverse), so only x/y determines clipping.
    radius_xy = torch.sqrt(x**2 + y**2)
    outside = torch.nonzero(radius_xy >= helix_radius_m, as_tuple=False)
    if outside.numel() > 0:
        end_idx = max(int(outside[0].item()), 1)
        x = x[: end_idx + 1]
        y = y[: end_idx + 1]
        z = z[: end_idx + 1]
        y_linear = y_linear[: end_idx + 1]

    return x, y, z, y_linear


def _build_object_helices(data, object_name: str, batch_idx: int, magnetic_field_t: float, helix_radius_m: float, cycler):
    phi = data[f"{object_name}_phi"][batch_idx].to(torch.float32)
    if object_name == "track":
        theta = data["track_theta"][batch_idx].to(torch.float32)
        eta = -torch.log(torch.tan(0.5 * theta).clamp_min(1e-6))

        if "track_pt" in data:
            pt = data["track_pt"][batch_idx].to(torch.float32)
        else:
            qop = data["track_qop"][batch_idx].to(torch.float32)
            p = 1.0 / qop.abs().clamp_min(1e-6)
            pt = p * torch.sin(theta)

        if "track_qop" in data:
            charge_sign = torch.sign(data["track_qop"][batch_idx].to(torch.float32))
            charge_sign = torch.where(charge_sign == 0, torch.ones_like(charge_sign), charge_sign)
        else:
            charge_sign = torch.ones_like(pt)
    else:
        eta = data["particle_eta"][batch_idx].to(torch.float32)
        pt = data["particle_pt"][batch_idx].to(torch.float32)
        charge_sign = torch.sign(data["particle_charge"][batch_idx].to(torch.float32))
        charge_sign = torch.where(charge_sign == 0, torch.ones_like(charge_sign), charge_sign)

    pt = pt.abs().clamp_min(1e-6)
    # ODD impact parameters are provided in mm; convert to m for helix geometry.
    d0 = 1e-3 * data[f"{object_name}_d0"][batch_idx].to(torch.float32)
    z0 = 1e-3 * data[f"{object_name}_z0"][batch_idx].to(torch.float32)
    valid = data[f"{object_name}_valid"][batch_idx].to(torch.bool)

    helices = []
    for object_idx_tensor in torch.nonzero(valid, as_tuple=False).flatten():
        object_idx = int(object_idx_tensor.item())
        helix = _build_helix_path(
            phi[object_idx],
            eta[object_idx],
            pt[object_idx],
            charge_sign[object_idx],
            d0[object_idx],
            z0[object_idx],
            magnetic_field_t=magnetic_field_t,
            helix_radius_m=helix_radius_m,
        )
        color = cycler[object_idx % len(cycler)]
        helices.append((*helix, color))

    return helices


def _clip_helix_to_z_extent(
    x_path: torch.Tensor,
    y_path: torch.Tensor,
    z_path: torch.Tensor,
    y_linear_path: torch.Tensor,
    z_extent: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    in_range = torch.nonzero(torch.abs(z_path) <= z_extent, as_tuple=False).flatten()
    if in_range.numel() < 2:
        return x_path[:0], y_path[:0], z_path[:0], y_linear_path[:0]

    end_idx = int(in_range[-1].item())
    return (
        x_path[: end_idx + 1],
        y_path[: end_idx + 1],
        z_path[: end_idx + 1],
        y_linear_path[: end_idx + 1],
    )


def plot_odd_event(
    data,
    batch_idx: int = 0,
    plot_sihits: bool = False,
    plot_particle_sihits: bool = False,
    plot_track_sihits: bool = False,
    plot_calohits: bool = False,
    plot_particle_calohits: bool = False,
    plot_particles: bool = False,
    plot_tracker: bool = False,
    magnetic_field_t: float = 3.0,
    helix_radius_m: float = 1.0,
):
    if not any(
        [
            plot_sihits,
            plot_particle_sihits,
            plot_track_sihits,
            plot_calohits,
            plot_particle_calohits,
            plot_particles,
            plot_tracker,
        ]
    ):
        msg = "Nothing selected to plot. Enable at least one plot_* flag."
        raise ValueError(msg)

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    ax = ax.flatten()

    colormap = plt.cm.tab20
    cycler = [colormap(i) for i in range(colormap.N)]
    axis_specs = [("x", "y", r"Global $x$", r"Global $y$"), ("z", "y", r"Global $z$", r"Global $y$")]
    particle_helices = []
    track_helices = []
    if plot_particles:
        particle_helices = _build_object_helices(data, "particle", batch_idx, magnetic_field_t, helix_radius_m, cycler)
    if plot_tracker:
        track_helices = _build_object_helices(data, "track", batch_idx, magnetic_field_t, helix_radius_m, cycler)

    z_extent = float(torch.abs(data["sihit_z"][batch_idx]).max().item())
    if "calohit_z" in data:
        z_extent = max(z_extent, float(torch.abs(data["calohit_z"][batch_idx]).max().item()))

    for ax_idx, (x_field, y_field, x_label, y_label) in enumerate(axis_specs):
        si_y = data[f"sihit_{y_field}"][batch_idx]
        calo_y = data[f"calohit_{y_field}"][batch_idx]

        if plot_sihits or plot_particle_sihits or plot_track_sihits:
            si_x = data[f"sihit_{x_field}"][batch_idx]

            if plot_sihits:
                ax[ax_idx].scatter(si_x, si_y, alpha=0.25, s=1.0, color="black")
            if plot_particle_sihits:
                _plot_object_sihits(ax[ax_idx], data, "particle", si_x, si_y, batch_idx, cycler)
            if plot_track_sihits:
                _plot_object_sihits(ax[ax_idx], data, "track", si_x, si_y, batch_idx, cycler)

        if plot_calohits or plot_particle_calohits:
            calo_x = data[f"calohit_{x_field}"][batch_idx]

            if plot_calohits:
                ax[ax_idx].scatter(calo_x, calo_y, alpha=0.25, s=1.0, color="black", marker=".")
            if plot_particle_calohits:
                _plot_particle_calohits(ax[ax_idx], data, calo_x, calo_y, batch_idx, cycler)

        for x_path, y_path, z_path, y_linear_path, color in particle_helices:
            x_path, y_path, z_path, y_linear_path = _clip_helix_to_z_extent(x_path, y_path, z_path, y_linear_path, z_extent)
            if x_path.numel() < 2:
                continue
            coords = {"x": x_path, "y": y_path, "z": z_path}
            if x_field == "z" and y_field == "y":
                coords["y"] = y_linear_path
            ax[ax_idx].plot(coords[x_field], coords[y_field], color=color, linewidth=1.25, alpha=0.95)

        for x_path, y_path, z_path, y_linear_path, color in track_helices:
            x_path, y_path, z_path, y_linear_path = _clip_helix_to_z_extent(x_path, y_path, z_path, y_linear_path, z_extent)
            if x_path.numel() < 2:
                continue
            coords = {"x": x_path, "y": y_path, "z": z_path}
            if x_field == "z" and y_field == "y":
                coords["y"] = y_linear_path
            ax[ax_idx].plot(coords[x_field], coords[y_field], color=color, linewidth=1.25, alpha=0.95)

        ax[ax_idx].set_xlabel(x_label)
        ax[ax_idx].set_ylabel(y_label)

    return fig
