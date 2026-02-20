from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm
from tqdm import tqdm

from hepattn.experiments.odd.data import ODDDataset
from hepattn.utils.histogram import CountingHistogram
from hepattn.utils.plotting import plot_hist_to_ax

plt.rcParams["figure.dpi"] = 300


PARTICLE_ALIASES = {
    "pt": r"$p_T$ [GeV]",
    "energy": "Energy [GeV]",
    "energy_calo_sum": "Calo Energy Sum [GeV]",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "d0": r"$d_0$",
    "z0": r"$z_0$",
    "num_sihits": "Num. Si Hits",
    "num_ecalhits": "Num. ECAL Hits",
    "num_hcalhits": "Num. HCAL Hits",
    "num_calohits": "Num. Calo Hits",
}

PARTICLE_SCALES = {
    "pt": "log",
    "energy": "log",
    "energy_calo_sum": "log",
    "eta": "linear",
    "phi": "linear",
    "d0": "linear",
    "z0": "linear",
    "num_sihits": "linear",
    "num_ecalhits": "linear",
    "num_hcalhits": "linear",
    "num_calohits": "linear",
}

PARTICLE_BINS = {
    "pt": np.geomspace(0.5, 500.0, 40),
    "energy": np.geomspace(5e-1, 2.5e2, 40),
    "energy_calo_sum": np.geomspace(1e-4, 1e-1, 40),
    "eta": np.linspace(-4.0, 4.0, 40),
    "phi": np.linspace(-np.pi, np.pi, 40),
    "d0": np.linspace(-2.0, 2.0, 40),
    "z0": np.linspace(-100.0, 100.0, 40),
    "num_sihits": np.linspace(0.0, 32.0, 33),
    "num_ecalhits": np.linspace(0.0, 128.0, 129),
    "num_hcalhits": np.linspace(0.0, 128.0, 129),
    "num_calohits": np.linspace(0.0, 128.0, 129),
}

HIT_ALIASES = {
    "x": "Position $x$ [m]",
    "y": "Position $y$ [m]",
    "z": "Position $z$ [m]",
}

HIT_SCALES = {
    "x": "linear",
    "y": "linear",
    "z": "linear",
}

HIT_BINS = {
    "x": np.linspace(-4.0, 4.0, 40),
    "y": np.linspace(-4.0, 4.0, 40),
    "z": np.linspace(-8.0, 8.0, 40),
}

SELECTION_ALIASES = {
    "valid": "All",
    "is_charged_hadron": "Charged Hadrons",
    "is_neutral_hadron": "Neutral Hadrons",
    "is_electron": "Electrons",
    "is_photon": "Photons",
    "is_muon": "Muons",
    "is_tau": "Taus",
}

SELECTION_COLOURS = {
    "valid": "tab:blue",
    "is_charged_hadron": "tab:orange",
    "is_neutral_hadron": "tab:green",
    "is_electron": "tab:red",
    "is_photon": "tab:purple",
    "is_muon": "tab:brown",
    "is_tau": "tab:pink",
}

CALIBRATED_ENERGY_BINS = np.geomspace(1e-2, 1e1, 40)
CALIBRATED_ENERGY_ALIASES = {
    "particle_energy_ecal_calib": "ECAL Calibrated Energy [GeV]",
    "particle_energy_hcal_calib": "HCAL Calibrated Energy [GeV]",
    "particle_energy_calo_calib": "Total Calibrated Energy [GeV]",
}
CALO_SUBSYSTEM_ALIASES = {
    "ecb": "ECAL Barrel",
    "ece": "ECAL Endcap",
    "hcb": "HCAL Barrel",
    "hce": "HCAL Endcap",
}

particle_hists = {
    field: {selection: CountingHistogram(field_bins) for selection in SELECTION_ALIASES}
    for field, field_bins in PARTICLE_BINS.items()
}

hit_hists = {
    hit_name: {
        field: {selection: CountingHistogram(field_bins) for selection in SELECTION_ALIASES}
        for field, field_bins in HIT_BINS.items()
    }
    for hit_name in ("sihit", "calohit")
}


def _load_config():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    return yaml.safe_load(config_path.read_text())["data"]


def _build_dataset_kwargs(config):
    dataset_kwargs = {
        "dirpath": config.get("test_dir", config["train_dir"]),
        "num_events": 10,
        "particle_min_pt": config["particle_min_pt"],
        "particle_max_abs_eta": config["particle_max_abs_eta"],
        "particle_hit_cuts": config.get("particle_hit_cuts"),
        "particle_include_charged": config["particle_include_charged"],
        "particle_include_neutral": config["particle_include_neutral"],
        "event_type": config.get("event_type", "ttbar"),
        "debug": False,
        "return_calohits": True,
        "return_tracks": False,
        "build_calohit_associations": True,
    }

    return dataset_kwargs


def _to_numpy(values):
    return values.detach().cpu().numpy()


def _selection_masks(targets):
    particle_valid = targets["particle_valid"][0].bool()
    masks = {"valid": _to_numpy(particle_valid)}
    for selection in SELECTION_ALIASES:
        if selection == "valid":
            continue
        masks[selection] = _to_numpy(particle_valid & targets[f"particle_{selection}"][0].bool())
    return masks


def _selected_hit_indices(indptr: np.ndarray, indices: np.ndarray, particle_mask: np.ndarray) -> np.ndarray:
    particle_rows = np.nonzero(particle_mask)[0]
    if particle_rows.size == 0:
        return np.zeros(0, dtype=np.int64)

    chunks = [indices[indptr[row] : indptr[row + 1]] for row in particle_rows]
    if not chunks:
        return np.zeros(0, dtype=np.int64)

    hit_indices = np.concatenate(chunks)
    if hit_indices.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(hit_indices.astype(np.int64, copy=False))


def _fill_particle_hists(targets, selection_masks):
    num_calohits = (
        _to_numpy(targets["particle_num_calohits"][0])
        if "particle_num_calohits" in targets
        else np.diff(_to_numpy(targets["particle_calohit_indptr"][0]).astype(np.int64, copy=False))
    )
    num_ecalhits = (
        _to_numpy(targets["particle_num_ecalhits"][0])
        if "particle_num_ecalhits" in targets
        else np.zeros_like(num_calohits, dtype=np.float32)
    )
    num_hcalhits = (
        _to_numpy(targets["particle_num_hcalhits"][0])
        if "particle_num_hcalhits" in targets
        else np.zeros_like(num_calohits, dtype=np.float32)
    )

    particle_values = {
        "pt": _to_numpy(targets["particle_pt"][0]),
        "energy": _to_numpy(targets["particle_energy"][0]),
        "energy_calo_sum": _to_numpy(targets["particle_energy_calo_sum"][0]),
        "eta": _to_numpy(targets["particle_eta"][0]),
        "phi": _to_numpy(targets["particle_phi"][0]),
        "d0": _to_numpy(targets["particle_d0"][0]),
        "z0": _to_numpy(targets["particle_z0"][0]),
        "num_sihits": _to_numpy(targets["particle_num_sihits"][0]),
        "num_ecalhits": num_ecalhits,
        "num_hcalhits": num_hcalhits,
        "num_calohits": num_calohits,
    }

    for field, selection_hists in particle_hists.items():
        values = particle_values[field]
        for selection, hist in selection_hists.items():
            hist.fill(values[selection_masks[selection]])


def _plot_calibrated_energy_vs_true_2d(
    class_name: str,
    class_label: str,
    energy_true: np.ndarray,
    energy_ecal_calib: np.ndarray,
    energy_hcal_calib: np.ndarray,
    energy_total_calib: np.ndarray,
    plot_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15, 4)

    y_fields = [
        ("particle_energy_ecal_calib", energy_ecal_calib),
        ("particle_energy_hcal_calib", energy_hcal_calib),
        ("particle_energy_calo_calib", energy_total_calib),
    ]

    for ax_idx, (field_name, y_values) in enumerate(y_fields):
        valid = np.isfinite(energy_true) & np.isfinite(y_values) & (energy_true > 0.0) & (y_values > 0.0)
        if np.any(valid):
            hist = ax[ax_idx].hist2d(
                energy_true[valid],
                y_values[valid],
                bins=[PARTICLE_BINS["energy"], CALIBRATED_ENERGY_BINS],
                norm=LogNorm(),
            )
            cbar = fig.colorbar(hist[3], ax=ax[ax_idx])
            cbar.set_label("Count")
        else:
            ax[ax_idx].text(
                0.5,
                0.5,
                "No positive entries",
                ha="center",
                va="center",
                transform=ax[ax_idx].transAxes,
            )

        ax[ax_idx].set_xscale("log")
        ax[ax_idx].set_yscale("log")
        ax[ax_idx].set_xlabel("Particle Energy [GeV]")
        ax[ax_idx].set_ylabel(CALIBRATED_ENERGY_ALIASES[field_name])
        ax[ax_idx].grid(zorder=0, alpha=0.25, linestyle="--")

    fig.suptitle(f"{class_label}")
    fig.tight_layout()
    fig.savefig(plot_path / f"odd_particle_energy_calib_vs_true_2d_{class_name}.png")
    plt.close(fig)


def _fill_hit_hists(inputs, targets, selection_masks):
    hit_csr_keys = {
        "sihit": ("particle_sihit_indptr", "particle_sihit_indices"),
        "calohit": ("particle_calohit_indptr", "particle_calohit_indices"),
    }

    for hit_name, (indptr_key, indices_key) in hit_csr_keys.items():
        valid_hits = _to_numpy(inputs[f"{hit_name}_valid"][0]).astype(bool, copy=False)
        valid_hit_indices = np.nonzero(valid_hits)[0].astype(np.int64, copy=False)

        indptr = _to_numpy(targets[indptr_key][0]).astype(np.int64, copy=False)
        indices = _to_numpy(targets[indices_key][0]).astype(np.int64, copy=False)

        for field in HIT_BINS:
            values = _to_numpy(inputs[f"{hit_name}_{field}"][0])

            # All valid hits.
            hit_hists[hit_name][field]["valid"].fill(values[valid_hit_indices])

            # Hits associated to selected particle classes.
            for selection in SELECTION_ALIASES:
                if selection == "valid":
                    continue
                selected_hit_indices = _selected_hit_indices(indptr, indices, selection_masks[selection])
                if selected_hit_indices.size == 0:
                    continue
                selected_hit_indices = selected_hit_indices[valid_hits[selected_hit_indices]]
                hit_hists[hit_name][field][selection].fill(values[selected_hit_indices])


def _accumulate_calohit_energy_pairs(inputs, calohit_energy_by_subsystem):
    total_energy = _to_numpy(inputs["calohit_total_energy"][0]).astype(np.float32, copy=False)
    contrib_energy_sum = _to_numpy(inputs["calohit_contrib_energy_sum"][0]).astype(np.float32, copy=False)
    detector_id = _to_numpy(inputs["calohit_detector"][0]).astype(np.int64, copy=False)
    valid_hits = _to_numpy(inputs["calohit_valid"][0]).astype(bool, copy=False)
    valid_hits = valid_hits & np.isfinite(total_energy) & np.isfinite(contrib_energy_sum)

    for subsystem, detector_ids in ODDDataset.CALO_SUBSYSTEM_DETECTOR_IDS.items():
        subsystem_mask = valid_hits & np.isin(detector_id, detector_ids)
        if not np.any(subsystem_mask):
            continue
        calohit_energy_by_subsystem[subsystem]["total"].append(total_energy[subsystem_mask])
        calohit_energy_by_subsystem[subsystem]["contrib_sum"].append(contrib_energy_sum[subsystem_mask])


def _concat_or_empty(chunks):
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    non_empty = [np.asarray(chunk, dtype=np.float32) for chunk in chunks if np.asarray(chunk).size > 0]
    if not non_empty:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(non_empty)


def _log_bins_from_positive(values: np.ndarray, num_bins: int = 45) -> np.ndarray:
    positive_values = values[np.isfinite(values) & (values > 0.0)]
    if positive_values.size == 0:
        return np.geomspace(1e-8, 1.0, num_bins)

    lower = max(float(positive_values.min()), 1e-12)
    upper = float(positive_values.max())
    if upper <= lower:
        upper = lower * 1.001
    return np.geomspace(lower, upper, num_bins)


def _plot_calohit_total_vs_contrib_sum_2d(calohit_energy_by_subsystem, plot_path: Path) -> None:
    subsystem_order = ["ecb", "ece", "hcb", "hce"]
    subsystem_values = {}
    all_total = []
    all_contrib = []

    for subsystem in subsystem_order:
        total_energy = _concat_or_empty(calohit_energy_by_subsystem[subsystem]["total"])
        contrib_energy_sum = _concat_or_empty(calohit_energy_by_subsystem[subsystem]["contrib_sum"])
        subsystem_values[subsystem] = (total_energy, contrib_energy_sum)
        if total_energy.size > 0:
            all_total.append(total_energy)
        if contrib_energy_sum.size > 0:
            all_contrib.append(contrib_energy_sum)

    total_bins = _log_bins_from_positive(_concat_or_empty(all_total))
    contrib_bins = _log_bins_from_positive(_concat_or_empty(all_contrib))

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    axes = axes.flatten()

    for idx, subsystem in enumerate(subsystem_order):
        total_energy, contrib_energy_sum = subsystem_values[subsystem]
        valid = np.isfinite(total_energy) & np.isfinite(contrib_energy_sum) & (total_energy > 0.0) & (contrib_energy_sum > 0.0)

        if np.any(valid):
            hist = axes[idx].hist2d(
                total_energy[valid],
                contrib_energy_sum[valid],
                bins=[total_bins, contrib_bins],
                norm=LogNorm(),
            )
            cbar = fig.colorbar(hist[3], ax=axes[idx])
            cbar.set_label("Count")

            lower = max(total_bins[0], contrib_bins[0])
            upper = min(total_bins[-1], contrib_bins[-1])
            if upper > lower:
                axes[idx].plot([lower, upper], [lower, upper], color="white", linestyle="--", linewidth=1.0, alpha=0.8)
        else:
            axes[idx].text(
                0.5,
                0.5,
                "No positive entries",
                ha="center",
                va="center",
                transform=axes[idx].transAxes,
            )

        axes[idx].set_xscale("log")
        axes[idx].set_yscale("log")
        axes[idx].set_title(CALO_SUBSYSTEM_ALIASES[subsystem])
        axes[idx].set_xlabel("Calo Hit Total Energy [GeV]")
        axes[idx].set_ylabel("Sum of Contribution Energies [GeV]")
        axes[idx].grid(zorder=0, alpha=0.25, linestyle="--")

    fig.suptitle("Calo Hit Energy vs Summed Particle Contribution Energy")
    fig.tight_layout()
    fig.savefig(plot_path / "odd_calohit_total_vs_contrib_sum_2d_by_subsystem.png")
    plt.close(fig)


def _plot_hist_group(hist_group, fields, scales, aliases, x_label_prefix, plot_path):
    fig, ax = plt.subplots(1, len(fields))
    fig.set_size_inches(4 * len(fields), 3)
    ax = [ax] if len(fields) == 1 else ax

    for ax_idx, field in enumerate(fields):
        for selection, hist in hist_group[field].items():
            plot_hist_to_ax(
                ax[ax_idx],
                hist.counts,
                hist.bins,
                label=SELECTION_ALIASES[selection],
                color=SELECTION_COLOURS[selection],
                vertical_lines=True,
            )

        ax[ax_idx].set_yscale("log")
        ax[ax_idx].set_xscale(scales[field])
        ax[ax_idx].set_xlabel(f"{x_label_prefix} {aliases[field]}")
        ax[ax_idx].set_ylabel("Count")
        ax[ax_idx].grid(zorder=0, alpha=0.25, linestyle="--")

    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def main():
    config = _load_config()
    dataset = ODDDataset(**_build_dataset_kwargs(config))
    num_events = min(10, len(dataset))
    energy_by_class = {selection: [] for selection in SELECTION_ALIASES if selection != "valid"}
    ecal_calib_by_class = {selection: [] for selection in SELECTION_ALIASES if selection != "valid"}
    hcal_calib_by_class = {selection: [] for selection in SELECTION_ALIASES if selection != "valid"}
    total_calib_by_class = {selection: [] for selection in SELECTION_ALIASES if selection != "valid"}
    calohit_energy_by_subsystem = {
        subsystem: {"total": [], "contrib_sum": []}
        for subsystem in ODDDataset.CALO_SUBSYSTEM_DETECTOR_IDS
    }

    for event_idx in tqdm(range(num_events)):
        inputs, targets = dataset[event_idx]
        selection_masks = _selection_masks(targets)
        _fill_particle_hists(targets, selection_masks)
        _fill_hit_hists(inputs, targets, selection_masks)
        _accumulate_calohit_energy_pairs(inputs, calohit_energy_by_subsystem)

        energy_true = _to_numpy(targets["particle_energy"][0])
        energy_ecal_calib = _to_numpy(targets["particle_energy_ecal_calib"][0])
        energy_hcal_calib = _to_numpy(targets["particle_energy_hcal_calib"][0])
        energy_total_calib = _to_numpy(targets["particle_energy_calo_calib"][0])
        for selection in SELECTION_ALIASES:
            if selection == "valid":
                continue
            mask = selection_masks[selection]
            energy_by_class[selection].append(energy_true[mask])
            ecal_calib_by_class[selection].append(energy_ecal_calib[mask])
            hcal_calib_by_class[selection].append(energy_hcal_calib[mask])
            total_calib_by_class[selection].append(energy_total_calib[mask])

    plot_dir = Path(__file__).resolve().parents[1] / "plots" / "data"
    plot_dir.mkdir(parents=True, exist_ok=True)

    particle_plots = {
        "odd_particle_kinematics": ["pt", "eta", "phi"],
        "odd_particle_energy": ["energy", "energy_calo_sum"],
        "odd_particle_impact_params": ["d0", "z0"],
        "odd_particle_hit_multiplicity": ["num_sihits", "num_ecalhits", "num_hcalhits"],
    }

    for plot_name, fields in particle_plots.items():
        _plot_hist_group(
            hist_group=particle_hists,
            fields=fields,
            scales=PARTICLE_SCALES,
            aliases=PARTICLE_ALIASES,
            x_label_prefix="Particle",
            plot_path=plot_dir / f"{plot_name}.png",
        )

    _plot_hist_group(
        hist_group=hit_hists["sihit"],
        fields=["x", "y", "z"],
        scales=HIT_SCALES,
        aliases=HIT_ALIASES,
        x_label_prefix="Si Hit",
        plot_path=plot_dir / "odd_sihit_xyz.png",
    )

    _plot_hist_group(
        hist_group=hit_hists["calohit"],
        fields=["x", "y", "z"],
        scales=HIT_SCALES,
        aliases=HIT_ALIASES,
        x_label_prefix="Calo Hit",
        plot_path=plot_dir / "odd_calohit_xyz.png",
    )

    for selection, class_label in SELECTION_ALIASES.items():
        if selection == "valid":
            continue
        energy_true = np.concatenate(energy_by_class[selection]) if energy_by_class[selection] else np.zeros(0, dtype=np.float32)
        energy_ecal_calib = (
            np.concatenate(ecal_calib_by_class[selection]) if ecal_calib_by_class[selection] else np.zeros(0, dtype=np.float32)
        )
        energy_hcal_calib = (
            np.concatenate(hcal_calib_by_class[selection]) if hcal_calib_by_class[selection] else np.zeros(0, dtype=np.float32)
        )
        energy_total_calib = (
            np.concatenate(total_calib_by_class[selection]) if total_calib_by_class[selection] else np.zeros(0, dtype=np.float32)
        )
        _plot_calibrated_energy_vs_true_2d(
            class_name=selection,
            class_label=class_label,
            energy_true=energy_true,
            energy_ecal_calib=energy_ecal_calib,
            energy_hcal_calib=energy_hcal_calib,
            energy_total_calib=energy_total_calib,
            plot_path=plot_dir,
        )

    _plot_calohit_total_vs_contrib_sum_2d(
        calohit_energy_by_subsystem=calohit_energy_by_subsystem,
        plot_path=plot_dir,
    )


if __name__ == "__main__":
    main()
