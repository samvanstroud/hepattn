from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import yaml

from hepattn.experiments.cld.data import CLDDataset
from hepattn.experiments.cld.plot_event import plot_cld_event_reconstruction

plt.rcParams["figure.dpi"] = 300


class TestCLDEvent:
    @pytest.fixture
    def cld_event(self):
        config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
        config = yaml.safe_load(config_path.read_text())["data"]

        dirpath = "/share/rcifdata/maxhart/data/cld/prepped"
        num_events = -1
        particle_min_pt = 0.1
        event_max_num_particles = 1000

        merge_inputs = {
            "sihit": [
                "vtb",
                "vte",
                "itb",
                "ite",
                "otb",
                "ote",
            ],
            "ecal": ["ecb", "ece"],
            "hcal": [
                "hcb",
                "hce",
                "hco",
            ],
        }

        dataset = CLDDataset(
            dirpath=dirpath,
            inputs=config["inputs"],
            targets=config["targets"],
            num_events=num_events,
            particle_min_pt=particle_min_pt,
            event_max_num_particles=event_max_num_particles,
            merge_inputs=merge_inputs,
        )

        return dataset[0]

    def test_cld_event_masks(self, cld_event):
        # Some basic sanity checks on the event data
        _inputs, _targets = cld_event

        # Every valid particle should have a unique particle id

        # Particle id should be a long

    def test_cld_event_display_merged_inputs(self, cld_event):
        # Plot an event display directly from dataloader with merged
        # inputs to verify things look correct
        inputs, targets = cld_event

        # Plot the full event with all subsytems
        axes_spec = [
            {
                "x": "pos.x",
                "y": "pos.y",
                "input_names": [
                    "sihit",
                    "ecal",
                    "hcal",
                    "muon",
                ],
            },
            {
                "x": "pos.z",
                "y": "pos.y",
                "input_names": [
                    "sihit",
                    "ecal",
                    "hcal",
                    "muon",
                ],
            },
        ]

        fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
        fig.savefig(Path("tests/outputs/cld/cld_event_full_merged.png"))

        # Plot just the si tracker
        axes_spec = [
            {
                "x": "pos.x",
                "y": "pos.y",
                "input_names": [
                    "sihit",
                ],
            },
            {
                "x": "pos.z",
                "y": "pos.y",
                "input_names": [
                    "sihit",
                ],
            },
        ]

        fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
        fig.savefig(Path("tests/outputs/cld/cld_event_tracker_merged.png"))

    def test_cld_event_display(self, cld_event):
        # Plot an event display directly from dataloader to verify things look correct
        inputs, targets = cld_event

        # Plot the full event with all subsytems
        axes_spec = [
            # Barrel view, so only plot barrel hits
            {
                "x": "pos.x",
                "y": "pos.y",
                "input_names": [
                    "vtb",
                    "itb",
                    "otb",
                    "ecb",
                    "hcb",
                    "muon",
                ],
            },
            # Side view, so plot barrel and endcap hits
            {
                "x": "pos.z",
                "y": "pos.y",
                "input_names": [
                    "vtb",
                    "vte",
                    "itb",
                    "ite",
                    "otb",
                    "ote",
                    "ecb",
                    "ece",
                    "hcb",
                    "hce",
                    "muon",
                ],
            },
        ]

        fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
        fig.savefig(Path("tests/outputs/cld/cld_event_full.png"))

        # Now plot just the tracker systems
        axes_spec = [
            {
                "x": "pos.x",
                "y": "pos.y",
                "input_names": [
                    "vtb",
                    "itb",
                    "otb",
                ],
            },
            {
                "x": "pos.z",
                "y": "pos.y",
                "input_names": [
                    "vtb",
                    "vte",
                    "itb",
                    "ite",
                    "otb",
                    "ote",
                ],
            },
        ]

        fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
        fig.savefig(Path("tests/outputs/cld/cld_event_tracker.png"))

        # Now plot just the vertex tracker detector
        axes_spec = [
            {
                "x": "pos.x",
                "y": "pos.y",
                "input_names": [
                    "vtb",
                ],
            },
            {
                "x": "pos.z",
                "y": "pos.y",
                "input_names": [
                    "vtb",
                    "vte",
                ],
            },
        ]

        fig = plot_cld_event_reconstruction(inputs, targets, axes_spec)
        fig.savefig(Path("tests/outputs/cld/cld_event_vtxd.png"))
