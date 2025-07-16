from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import yaml

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.plot_event import plot_cld_event_reconstruction

plt.rcParams["figure.dpi"] = 300

torch.manual_seed(42)


@pytest.fixture
def cld_datamodule():
    config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
    config = yaml.safe_load(config_path.read_text())["data"]
    config["num_workers"] = 0

    datamodule = CLDDataModule(**config)
    datamodule.setup(stage="test")

    return datamodule


@pytest.mark.requiresdata
def test_cld_event_display(cld_datamodule):
    # Plot an event display directly from dataloader with merged
    # inputs to verify things look correct

    test_dataloader = cld_datamodule.test_dataloader()

    Path("tests/outputs/cld/").mkdir(parents=True, exist_ok=True)

    for _i in range(1):
        inputs, targets = test_dataloader.dataset[0]

        # Plot the full event with all subsytems
        axes_spec = [
            {
                "x": "pos.x",
                "y": "pos.y",
                "px": "mom.x",
                "py": "mom.y",
                "input_names": [
                    "trkr",
                    "ecal",
                    "hcal",
                    "muon",
                ],
            },
            {
                "x": "pos.z",
                "y": "pos.y",
                "px": "mom.z",
                "py": "mom.y",
                "input_names": [
                    "trkr",
                    "ecal",
                    "hcal",
                    "muon",
                ],
            },
        ]

        fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
        fig.savefig(Path("tests/outputs/cld/cld_event.png"))

        # Plot just the inner and outter tracker
        axes_spec = [
            {
                "x": "pos.x",
                "y": "pos.y",
                "px": "mom.x",
                "py": "mom.y",
                "input_names": [
                    "trkr",
                ],
            },
            {
                "x": "pos.z",
                "y": "pos.y",
                "px": "mom.z",
                "py": "mom.y",
                "input_names": [
                    "trkr",
                ],
            },
        ]

        fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
        fig.savefig(Path("tests/outputs/cld/cld_event_trkr.png"))

        # Plot just the vertex detector
        axes_spec = [
            {
                "x": "pos.x",
                "y": "pos.y",
                "px": "mom.x",
                "py": "mom.y",
                "input_names": [
                    "vtxd",
                ],
            },
            {
                "x": "pos.z",
                "y": "pos.y",
                "px": "mom.z",
                "py": "mom.y",
                "input_names": [
                    "vtxd",
                ],
            },
        ]

        fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
        fig.savefig(Path("tests/outputs/cld/cld_event_vtxd.png"))
