from pathlib import Path

import pytest

from hepattn.experiments.trackml.data import TrackMLDataset
from hepattn.experiments.trackml.eval.plot_kinematics import plot_trackml_kinematics


class TestTrackMLEvent:
    @pytest.fixture
    def trackml_dataset(self):
        input_fields = {
            "hit": [
                "x",
                "y",
                "z",
                "r",
                "eta",
                "phi",
            ]
        }

        target_fields = {
            "particle": ["pt", "eta", "phi"],
        }

        dirpath = "data/trackml/prepped/"
        num_events = 2
        hit_volume_ids = [7, 8, 9]
        particle_min_pt = 0.0
        particle_max_abs_eta = 4.0
        particle_min_num_hits = 1
        event_max_num_particles = 100000

        return TrackMLDataset(
            dirpath=dirpath,
            inputs=input_fields,
            targets=target_fields,
            num_events=num_events,
            hit_volume_ids=hit_volume_ids,
            particle_min_pt=particle_min_pt,
            particle_max_abs_eta=particle_max_abs_eta,
            particle_min_num_hits=particle_min_num_hits,
            event_max_num_particles=event_max_num_particles,
        )

    def test_trackml_plot_kinematics(self, trackml_dataset):
        fig = plot_trackml_kinematics(trackml_dataset)
        output_dir = Path("tests/outputs/trackml/")
        output_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_dir / Path("trackml_kinematics.png"))
