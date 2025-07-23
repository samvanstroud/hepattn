from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.utils.array_utils import masked_angle_diff_last_axis

plt.rcParams["figure.dpi"] = 300


@pytest.fixture
def cld_datamodule():
    config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
    config = yaml.safe_load(config_path.read_text())["data"]
    config["num_workers"] = 0
    config["batch_size"] = 10

    datamodule = CLDDataModule(**config)
    datamodule.setup(stage="fit")

    return datamodule


@pytest.mark.requiresdata
def test_plot_cld_hit_coords(cld_datamodule):
    dataloader = cld_datamodule.train_dataloader()
    data_iterator = iter(dataloader)

    inputs, _ = next(data_iterator)

    fig, ax = plt.subplots(4, 1)
    fig.set_size_inches(8, 6)

    for item_name in ["vtxd", "trkr"]:
        r = inputs[f"{item_name}_pos.r"][inputs[f"{item_name}_valid"]]
        phi = inputs[f"{item_name}_pos.phi"][inputs[f"{item_name}_valid"]]
        theta = inputs[f"{item_name}_pos.theta"][inputs[f"{item_name}_valid"]]
        eta = inputs[f"{item_name}_pos.eta"][inputs[f"{item_name}_valid"]]

        eta = np.clip(eta, -4, 4)

        ax[0].hist(r, bins=np.geomspace(0.01, 10, 64), label=item_name, density=True, histtype="step")
        ax[1].hist(phi, bins=np.linspace(-np.pi, np.pi, 64), label=item_name, density=True, histtype="step")
        ax[2].hist(theta, bins=np.linspace(0, np.pi, 64), label=item_name, density=True, histtype="step")
        ax[3].hist(eta, bins=np.linspace(-4, 4, 64), label=item_name, density=True, histtype="step")

    ax[0].set_xlabel(r"$r$")
    ax[1].set_xlabel(r"$\phi$")
    ax[2].set_xlabel(r"$\theta$")
    ax[3].set_xlabel(r"$\eta$")

    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    ax[3].set_yscale("log")

    ax[0].grid(zorder=0, alpha=0.25, linestyle="--")
    ax[1].grid(zorder=0, alpha=0.25, linestyle="--")
    ax[2].grid(zorder=0, alpha=0.25, linestyle="--")
    ax[3].grid(zorder=0, alpha=0.25, linestyle="--")

    ax[0].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_hit_coords.png"))


@pytest.mark.requiresdata
def test_plot_cld_trkr_momentum(cld_datamodule):
    dataloader = cld_datamodule.train_dataloader()
    data_iterator = iter(dataloader)

    _, targets = next(data_iterator)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(8, 2)

    for item_name in ["trkr", "vtxd"]:
        mask = targets[f"particle_{item_name}_valid"]
        px = targets[f"particle_{item_name}_mom.x"]
        py = targets[f"particle_{item_name}_mom.y"]
        pz = targets[f"particle_{item_name}_mom.z"]

        p = np.sqrt(px**2 + py**2 + pz**2)
        particle_mean_trkr_p = np.ma.masked_array(p, mask=~mask).mean(-1)
        normed_p = p / particle_mean_trkr_p[..., None]

        ax[0].hist(normed_p[mask], bins=np.linspace(0, 1.6, 24), histtype="step", label=item_name.upper())
        ax[1].hist(normed_p[mask], bins=np.linspace(0.99, 1.01, 24), histtype="step", label=item_name.upper())
        ax[2].hist(normed_p[mask], bins=np.linspace(0.0, 0.01, 24), histtype="step", label=item_name.upper())

    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    ax[0].set_ylabel("Count")

    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    ax[1].set_xlabel(r"Track $p$ on Hit / Particle Average Track $p$ on Hit")

    ax[2].set_yscale("log")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_particle_trkr_mom.png"))

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 2)

    for item_name in ["trkr", "vtxd"]:
        mask = targets[f"particle_{item_name}_valid"]
        px = np.ma.masked_array(targets[f"particle_{item_name}_mom.x"], mask=~mask)
        py = np.ma.masked_array(targets[f"particle_{item_name}_mom.y"], mask=~mask)
        pz = np.ma.masked_array(targets[f"particle_{item_name}_mom.z"], mask=~mask)

        angle_diffs = [masked_angle_diff_last_axis(px[i], py[i], pz[i], ~mask[i])[mask[i]] for i in range(len(mask))]

        ax[0].hist(np.concatenate(angle_diffs), bins=np.linspace(0.0, np.pi / 2, 64), histtype="step")
        ax[1].hist(np.concatenate(angle_diffs), bins=np.linspace(0.0, 0.1, 64), histtype="step")

    ax[0].set_yscale("log")
    ax[1].set_yscale("log")

    ax[0].set_xlabel("Hit Deflection Angle in Radians")
    ax[1].set_xlabel("Hit Deflection Angle in Radians")

    ax[0].set_ylabel("Count")
    ax[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_particle_trkr_angle.png"))


@pytest.mark.requiresdata
def test_plot_cld_calo_energy(cld_datamodule):
    dataloader = cld_datamodule.train_dataloader()
    data_iterator = iter(dataloader)
    inputs, targets = next(data_iterator)

    particle_ecal_energy = targets["particle_ecal_energy"]
    particle_hcal_energy = targets["particle_hcal_energy"]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 3)

    ecal_e = particle_ecal_energy[particle_ecal_energy > 0.0] * 1000
    hcal_e = particle_hcal_energy[particle_hcal_energy > 0.0] * 1000

    ax.hist(ecal_e, bins=np.logspace(-4, 2, 64), histtype="step", label="ECAL")
    ax.hist(hcal_e, bins=np.logspace(-4, 2, 64), histtype="step", label="HCAL")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Particle Calo Hit Energy Contribution [MeV]")
    ax.set_ylabel("Count")

    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_particle_hit_calo_energy.png"))

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)

    ecal_hit_e = inputs["ecal_energy"] * 1000
    ecal_sum_e = np.nan_to_num(particle_ecal_energy).sum(-2) * 1000

    hcal_hit_e = inputs["hcal_energy"] * 1000
    hcal_sum_e = np.nan_to_num(particle_hcal_energy).sum(-2) * 1000

    bins = (np.logspace(0.5, 2.5, 128), np.logspace(-1, 1, 128))

    ax[0].hist2d(ecal_hit_e.flatten(), ecal_sum_e.flatten(), bins=bins)

    ax[0].set_xscale("log")
    ax[0].set_yscale("log")

    ax[0].set_xlabel("ECAL Hit Digitized Energy [MeV]")
    ax[0].set_ylabel("ECAL Hit Sum of Particle Energy Contributions [MeV]")

    bins = (np.logspace(1, 2.5, 128), np.logspace(-1, 1, 128))

    ax[1].hist2d(hcal_hit_e.flatten(), hcal_sum_e.flatten(), bins=bins)

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")

    ax[1].set_xlabel("HCAL Hit Digitized Energy [MeV]")
    ax[1].set_ylabel("HCAL Hit Sum of Particle Energy Contributions [MeV]")

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_calo_energy_sum_hist.png"))

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)

    ecal_x = inputs["ecal_pos.x"].flatten()
    ecal_y = inputs["ecal_pos.y"].flatten()
    ecal_z = inputs["ecal_pos.z"].flatten()
    ecal_ratio = (ecal_hit_e / ecal_sum_e).flatten()

    ax[0].scatter(ecal_x, ecal_y, c=ecal_ratio, alpha=0.1, s=0.1)
    ax[1].scatter(ecal_z, ecal_y, c=ecal_ratio, alpha=0.1, s=0.1)

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_calo_ratio_coords.png"))

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 3)

    ax.hist(ecal_ratio, bins=np.linspace(0, 80, 128), histtype="step")

    ax.set_yscale("log")

    ax.set_xlabel("ECAL Hit / Sum of Contributions Ratio")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_calo_ratio_hist.png"))

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)

    ecal_x = inputs["ecal_pos.x"].flatten()
    ecal_y = inputs["ecal_pos.y"].flatten()
    ecal_z = inputs["ecal_pos.z"].flatten()
    ecal_ratio = (ecal_hit_e / ecal_sum_e).flatten()

    for cut, label in [
        ((ecal_ratio > 0.0) & (ecal_ratio < 25.0), "0-25"),
        ((ecal_ratio > 25.0) & (ecal_ratio < 40.0), "25-40"),
        ((ecal_ratio > 40.0) & (ecal_ratio < 80.0), "40-80"),
    ]:
        ax[0].scatter(ecal_x[cut], ecal_y[cut], alpha=0.5, s=0.05, label=label)
        ax[1].scatter(ecal_z[cut], ecal_y[cut], alpha=0.5, s=0.05, label=label)

    ax[0].set_xlabel(r"ECAL Hit $x$")
    ax[0].set_ylabel(r"ECAL Hit $y$")

    ax[1].set_xlabel(r"ECAL Hit $z$")
    ax[1].set_ylabel(r"ECAL Hit $y$")

    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_calo_ratio_coords_cut.png"))


@pytest.mark.requiresdata
def test_plot_cld_hit_distance(cld_datamodule):
    dataloader = cld_datamodule.train_dataloader()
    data_iterator = iter(dataloader)
    inputs, targets = next(data_iterator)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 3)

    for item_name in ["vtxd", "trkr"]:
        mask = targets[f"particle_{item_name}_valid"]

        x = np.ma.masked_array(mask * inputs[f"{item_name}_pos.x"][..., None, :], mask=~mask)
        y = np.ma.masked_array(mask * inputs[f"{item_name}_pos.y"][..., None, :], mask=~mask)
        z = np.ma.masked_array(mask * inputs[f"{item_name}_pos.z"][..., None, :], mask=~mask)

        t = np.ma.masked_array(mask * inputs[f"{item_name}_time"][..., None, :], mask=~mask)

        dx = np.ma.diff(np.take_along_axis(x, np.ma.argsort(t, axis=-1), axis=-1), axis=-1)
        dy = np.ma.diff(np.take_along_axis(y, np.ma.argsort(t, axis=-1), axis=-1), axis=-1)
        dz = np.ma.diff(np.take_along_axis(z, np.ma.argsort(t, axis=-1), axis=-1), axis=-1)

        dr = np.sqrt(dx**2 + dy**2 + dz**2)

        ax.hist(dr.flatten(), bins=np.logspace(-3, 0.25, 64), histtype="step", label=item_name.upper())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax.set_xlabel("Distance between Consecutive Hits")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_hit_dist.png"))


@pytest.mark.requiresdata
def test_plot_cld_hit_dr(cld_datamodule):
    dataloader = cld_datamodule.train_dataloader()
    data_iterator = iter(dataloader)
    inputs, targets = next(data_iterator)

    for item_name in ["vtxd", "trkr"]:
        mask = targets[f"particle_{item_name}_valid"]
        hit_r = inputs[f"{item_name}_pos.r"]

        # Only consider differences for hits that are away from the IP
        mask &= hit_r[..., None, :] >= 0.25

        hit_phi = inputs[f"{item_name}_pos.phi"]
        hit_eta = inputs[f"{item_name}_pos.eta"]
        hit_time = inputs[f"{item_name}_time"]

        phi = np.ma.masked_array(mask * hit_phi[..., None, :], mask=~mask)
        eta = np.ma.masked_array(mask * hit_eta[..., None, :], mask=~mask)
        time = np.ma.masked_array(mask * hit_time[..., None, :], mask=~mask)

        std_phi = np.ma.std(phi, axis=-1)
        std_eta = np.ma.std(eta, axis=-1)

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(8, 3)

        ax[0].hist(std_eta.flatten(), bins=32, histtype="step")
        ax[1].hist(std_phi.flatten(), bins=32, histtype="step")

        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

        fig.tight_layout()
        fig.savefig(Path(f"tests/outputs/cld/cld_{item_name}_std_eta_phi.png"))

        idx = np.ma.argsort(time, axis=-1)

        phi = np.take_along_axis(phi, idx, axis=-1)
        eta = np.take_along_axis(eta, idx, axis=-1)
        time = np.take_along_axis(time, idx, axis=-1)

        dphi = np.ma.diff(phi, axis=-1)
        dphi[dphi < -np.pi] += 2 * np.pi
        dphi[dphi > np.pi] -= 2 * np.pi

        deta = np.ma.diff(eta, axis=-1)
        dr = np.sqrt(deta**2 + dphi**2)

        fig, ax = plt.subplots(3, 2)
        fig.set_size_inches(8, 6)

        ax[0, 0].hist(deta.flatten(), bins=64, histtype="step")
        ax[1, 0].hist(dphi.flatten(), bins=64, histtype="step")
        ax[2, 0].hist(dr.flatten(), bins=64, histtype="step")

        ax[0, 0].set_yscale("log")
        ax[1, 0].set_yscale("log")
        ax[2, 0].set_yscale("log")

        ax[0, 0].set_xlabel(r"Hit $\Delta \eta$")
        ax[1, 0].set_xlabel(r"Hit $\Delta \phi$")
        ax[2, 0].set_xlabel(r"Hit $\Delta R$")

        ax[0, 0].set_ylabel("Count")

        ax[0, 1].hist(deta.flatten(), bins=np.linspace(-0.05, 0.05, 64), histtype="step")
        ax[1, 1].hist(dphi.flatten(), bins=np.linspace(-0.05, 0.05, 64), histtype="step")
        ax[2, 1].hist(dr.flatten(), bins=np.linspace(0.0, 0.05, 64), histtype="step")

        ax[0, 1].set_yscale("log")
        ax[1, 1].set_yscale("log")
        ax[2, 1].set_yscale("log")

        ax[0, 1].set_xlabel(r"Hit $\Delta \eta$")
        ax[1, 1].set_xlabel(r"Hit $\Delta \phi$")
        ax[2, 1].set_xlabel(r"Hit $\Delta R$")

        ax[0, 0].set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(Path(f"tests/outputs/cld/cld_{item_name}_deta_dphi_dr.png"))
