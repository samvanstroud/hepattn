from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import yaml

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.plot_event import plot_cld_event_reconstruction

plt.rcParams["figure.dpi"] = 300

torch.manual_seed(42)


class TestCLDDataModule:
    @pytest.fixture
    def cld_datamodule(self):
        config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
        config = yaml.safe_load(config_path.read_text())["data"]
        config["num_workers"] = 0

        datamodule = CLDDataModule(**config)
        datamodule.setup(stage="fit")

        return datamodule

    @pytest.mark.requiresdata
    def test_cld_masks(self, cld_datamodule):
        dataloader = cld_datamodule.train_dataloader()
        dataset = dataloader.dataset
        data_iterator = iter(dataloader)

        for _i in range(10):
            inputs, targets = next(data_iterator)

            # Valid particles should have no nan fields
            for field in dataset.targets["particle"]:
                assert torch.all(~torch.isnan(targets[f"particle_{field}"][targets["particle_valid"]]))
            for hit in ["trkr", "vtxd", "ecal", "hcal", "muon"]:
                # Any invalid particle slot should have no mask entries
                mask = targets[f"particle_{hit}_valid"]
                particles_num_hits = mask.sum(-1)

                targets["particle_valid"]
                invalid_particles = ~targets["particle_valid"]

                # Invalid particles should have no hits
                assert torch.all((particles_num_hits[invalid_particles]) == 0)

                # Check that the truth filtering does indeed remove all of the hits
                if hit in dataset.truth_filter_hits:
                    hit_with_no_particle = targets[f"particle_{hit}_valid"].sum(-2) == 0
                    hit_with_particle = targets[f"particle_{hit}_valid"].sum(-2) > 0

                    hit_valid = inputs[f"{hit}_valid"]
                    hit_invalid = ~inputs[f"{hit}_valid"]

                    # All valid hits should have a particle
                    assert torch.all(hit_valid[hit_with_particle])

                    # All invalid hits should not have a particle
                    assert torch.all(hit_invalid[hit_with_no_particle])

                    # All valid hits

    @pytest.mark.requiresdata
    def test_cld_event_display_merged_inputs(self, cld_datamodule):
        # Plot an event display directly from dataloader with merged
        # inputs to verify things look correct

        train_dataloader = cld_datamodule.train_dataloader()

        data_iterator = iter(train_dataloader)

        for _i in range(1):
            batch = next(data_iterator)

            inputs, targets = batch

            # Plot the full event with all subsytems
            axes_spec = [
                {
                    "x": "pos.phi",
                    "y": "pos.eta",
                    "input_names": [
                        "ecal",
                        "hcal",
                        "muon",
                    ],
                },
            ]

            fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
            fig.savefig(Path("tests/outputs/cld/cld_event_calo_etaphi.png"))

            axes_spec = [
                {
                    "x": "pos.phi",
                    "y": "pos.eta",
                    "input_names": [
                        "trkr",
                        "vtxd",
                    ],
                },
            ]

            fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
            fig.savefig(Path("tests/outputs/cld/cld_event_trkr_vtxd_etaphi.png"))

            axes_spec = [
                {
                    "x": "pos.u",
                    "y": "pos.v",
                    "input_names": [
                        "trkr",
                        "vtxd",
                    ],
                },
            ]

            fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
            fig.savefig(Path("tests/outputs/cld/cld_event_trkr_vtxd_uv.png"))

            # Plot the full event with all subsytems
            axes_spec = [
                {
                    "x": "pos.x",
                    "y": "pos.y",
                    "input_names": [
                        "trkr",
                        "vtxd",
                        "ecal",
                        "hcal",
                        "muon",
                    ],
                },
                {
                    "x": "pos.z",
                    "y": "pos.y",
                    "input_names": [
                        "trkr",
                        "vtxd",
                        "ecal",
                        "hcal",
                        "muon",
                    ],
                },
            ]

            fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
            fig.savefig(Path("tests/outputs/cld/cld_event.png"))

            # Plot just the si tracker
            axes_spec = [
                {
                    "x": "pos.x",
                    "y": "pos.y",
                    "input_names": [
                        "trkr",
                        "vtxd",
                    ],
                },
                {
                    "x": "pos.z",
                    "y": "pos.y",
                    "input_names": [
                        "trkr",
                        "vtxd",
                    ],
                },
            ]

            fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
            fig.savefig(Path("tests/outputs/cld/cld_event_tracker.png"))
