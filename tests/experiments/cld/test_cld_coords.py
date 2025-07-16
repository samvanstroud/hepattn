from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

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

    _, targets = next(data_iterator)

    # Plot the particle fields
    fields = ["mom.r", "mom.theta", "mom.phi", "mom.qopt"]

    fig, ax = plt.subplots(1, len(fields))
    fig.set_size_inches(12, 2)

    for j, field in enumerate(fields):
        ax[j].hist(targets[f"particle_{field}"][targets["particle_valid"]], bins=32, histtype="step")
        ax[j].set_yscale("log")
        ax[j].set_xlabel(f"particle {field}")

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_particle_fields.png"))

    # Plot the particle hit position fields
    hits = ["vtxd", "trkr"]
    fields = ["pos.x", "pos.y", "pos.z", "pos.r", "pos.theta", "pos.phi"]

    fig, ax = plt.subplots(len(hits), len(fields))
    fig.set_size_inches(12, 3)

    for i, hit in enumerate(hits):
        for j, field in enumerate(fields):
            ax[i, j].hist(targets[f"particle_{hit}_{field}"][targets[f"particle_{hit}_valid"]], bins=32, histtype="step")
            ax[i, j].set_yscale("log")
            ax[i, j].set_xlabel(f"{hit} {field}")

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_particle_hit_pos.png"))

    # Plot the particle hit momentum fields

    fields = ["mom.x", "mom.y", "mom.z", "mom.r", "mom.theta", "mom.phi"]

    fig, ax = plt.subplots(len(hits), len(fields))
    fig.set_size_inches(12, 3)

    for i, hit in enumerate(hits):
        for j, field in enumerate(fields):
            ax[i, j].hist(targets[f"particle_{hit}_{field}"][targets[f"particle_{hit}_valid"]], bins=32, histtype="step")
            ax[i, j].set_yscale("log")
            ax[i, j].set_xlabel(f"{hit} {field}")

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_particle_hit_mom.png"))

    # Plot the particle ecal fields

    hits = ["ecal", "hcal"]

    fig, ax = plt.subplots(len(hits), 3)
    fig.set_size_inches(12, 3)

    for i, hit in enumerate(hits):
        ax[i, 0].hist(targets[f"particle_{hit}_energy"][targets[f"particle_{hit}_valid"]], bins=np.logspace(-6, 3, 64), histtype="step")
        ax[i, 0].set_yscale("log")
        ax[i, 0].set_xlabel(f"{hit} energy")
        ax[i, 0].set_xscale("log")

        ax[i, 1].hist(targets[f"particle_{hit}_log_energy"][targets[f"particle_{hit}_valid"]], bins=64, histtype="step")
        ax[i, 1].set_yscale("log")
        ax[i, 1].set_xlabel(f"{hit} log energy")
        # ax[i,1].set_xscale("log")

        ax[i, 2].hist(targets[f"particle_{hit}_log_energy"][targets[f"particle_{hit}_valid"]], bins=np.logspace(-1, 2, 64), histtype="step")
        ax[i, 2].set_yscale("log")
        ax[i, 2].set_xlabel(f"{hit} log energy")
        ax[i, 2].set_xscale("log")

    fig.tight_layout()
    fig.savefig(Path("tests/outputs/cld/cld_particle_calo_energy.png"))

    # Plot the particle hcal fields
