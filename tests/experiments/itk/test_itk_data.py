from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from hepattn.experiments.itk.data import ITkDataset
from hepattn.experiments.itk.plot_event import plot_itk_event_reconstruction
from hepattn.models.matcher import Matcher

plt.rcParams["figure.dpi"] = 300


class TestITkEvent:
    @pytest.fixture
    def itk_dataset(self):
        input_fields = {
            "pixel": [
                "x",
                "y",
                "z",
                "r",
                "eta",
                "phi",
                "u",
                "v",
            ],
            "strip": [
                "x",
                "y",
                "z",
                "r",
                "eta",
                "phi",
                "u",
                "v",
            ],
        }

        target_fields = {
            "particle": ["pt", "eta", "phi"],
            "pixel": ["on_valid_particle"],
            "strip": ["on_valid_particle"],
            "particle_pixel": [],
            "particle_strip": [],
        }

        dirpath = "/share/rcifdata/maxhart/data/itk/val"
        num_events = -1
        hit_regions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        particle_min_pt = 1.0
        particle_max_abs_eta = 1.0
        particle_min_num_hits = {"pixel": 3, "strip": 3}
        event_max_num_particles = 2000

        dataset = ITkDataset(
            dirpath=dirpath,
            inputs=input_fields,
            targets=target_fields,
            num_events=num_events,
            hit_regions=hit_regions,
            particle_min_pt=particle_min_pt,
            particle_max_abs_eta=particle_max_abs_eta,
            particle_min_num_hits=particle_min_num_hits,
            event_max_num_particles=event_max_num_particles,
        )

        return dataset


    def test_itk_event_display(self, itk_event):
        # Plot an event display directly from dataloader to verify things look correct
        inputs, targets = itk_event

        fig = plot_itk_event_reconstruction(inputs, targets)
        fig.savefig(Path("tests/outputs/itk/itk_event.png"))

    def test_itk_matcher(self):
        Matcher(
            default_solver="scipy",
            adaptive_solver=False,
        )
