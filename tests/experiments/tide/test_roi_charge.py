from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import yaml

from hepattn.experiments.tide.data import ROIDataModule

plt.rcParams["figure.dpi"] = 300

torch.manual_seed(42)


class TestROIDataModule:
    @pytest.fixture
    def roi_datamodule(self):
        config_path = Path("src/hepattn/experiments/tide/configs/base.yaml")
        config = yaml.safe_load(config_path.read_text())["data"]
        config["num_workers"] = 0
        config["batch_size"] = 1000
        config["num_test"] = 10000

        datamodule = ROIDataModule(**config)
        datamodule.setup(stage="test")

        return datamodule

    @pytest.mark.requiresdata
    def test_roi_data(self, roi_datamodule):
        dataloader = roi_datamodule.test_dataloader()
        data_iterator = iter(dataloader)

        output_dir = Path("tests/outputs/tide/")

        inputs, _ = next(data_iterator)

        # Plot the histogram for the values of the charge matrices

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 3)

        charge_matrices = inputs["pix_log_charge_matrix"][inputs["pix_valid"]]

        ax.hist(charge_matrices.flatten(), bins=64, histtype="step")
        ax.set_yscale("log")

        ax.set_xlabel("Logarithm of pixel charge count")
        ax.set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_pix_chargemats_hist.png"))

        # Now plot the histogram for the total charge

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 3)

        ax.hist(inputs["pix_log_charge"][inputs["pix_valid"]], bins=64, histtype="step")
        ax.set_yscale("log")

        ax.set_xlabel("Logarithm of total pixel charge")
        ax.set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_pix_charge_hist.png"))
