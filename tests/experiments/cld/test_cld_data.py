from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import yaml

from hepattn.experiments.cld.data import CLDDataModule

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
