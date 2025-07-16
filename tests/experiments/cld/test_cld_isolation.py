from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import yaml
from matplotlib.colors import LogNorm

from hepattn.experiments.cld.data import CLDDataModule

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

    _inputs, targets = next(data_iterator)

    pts = []
    dphis = []
    detas = []
    isolations = []

    for i in range(len(targets["particle_valid"])):
        particle_valid = targets["particle_valid"][i]

        particle_pt = targets["particle_mom.r"][i][particle_valid]
        particle_phi = 1000 * targets["particle_mom.phi"][i][particle_valid]
        particle_eta = 1000 * targets["particle_mom.eta"][i][particle_valid]
        particle_valid = particle_valid[particle_valid]

        particle_dphi = particle_phi[:, None] - particle_phi[None, :]
        particle_deta = particle_eta[:, None] - particle_eta[None, :]
        particle_dr = (particle_dphi**2 + particle_deta**2).sqrt()

        particle_abs_dphi = particle_dphi.abs()
        particle_abs_deta = particle_dphi.abs()

        diag_idx = torch.arange(len(particle_abs_dphi))

        particle_abs_dphi[diag_idx, diag_idx] = torch.inf
        particle_abs_deta[diag_idx, diag_idx] = torch.inf
        particle_dr[diag_idx, diag_idx] = torch.inf

        particle_min_abs_dphi = torch.min(particle_abs_dphi, dim=-1)[0]
        particle_min_abs_deta = torch.min(particle_abs_deta, dim=-1)[0]
        particle_min_dr = torch.min(particle_dr, dim=-1)[0]

        pts.append(particle_pt)
        dphis.append(particle_min_abs_dphi)
        detas.append(particle_min_abs_deta)
        isolations.append(particle_min_dr)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12, 4)

    ax[0].hist2d(
        torch.cat(pts), torch.cat(dphis), bins=(np.geomspace(0.1, 100.0, 32), np.geomspace(1.0, 1000 * np.pi, 32)), norm=LogNorm(vmin=1, vmax=None)
    )
    ax[1].hist2d(
        torch.cat(pts), torch.cat(detas), bins=(np.geomspace(0.1, 100.0, 32), np.geomspace(1.0, 1000 * np.pi, 32)), norm=LogNorm(vmin=1, vmax=None)
    )
    ax[2].hist2d(
        torch.cat(pts),
        torch.cat(isolations),
        bins=(np.geomspace(0.1, 100.0, 32), np.geomspace(0.1, 1000 * np.pi, 32)),
        norm=LogNorm(vmin=1, vmax=None),
    )

    ax[0].set_xscale("log")
    ax[0].set_yscale("log")

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")

    ax[2].set_xscale("log")
    ax[2].set_yscale("log")

    ax[0].set_xlabel(r"Particle $p_T$")
    ax[0].set_ylabel(r"Particle $\Delta \phi$ Isolation")

    ax[1].set_xlabel(r"Particle $p_T$")
    ax[1].set_ylabel(r"Particle $\Delta \eta$ Isolation")

    ax[2].set_xlabel(r"Particle $p_T$")
    ax[2].set_ylabel(r"Particle $\Delta R$ Isolation")

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_particle_isolation.png"))
