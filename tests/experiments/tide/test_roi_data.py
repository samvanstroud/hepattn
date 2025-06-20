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

        datamodule = ROIDataModule(**config)
        datamodule.setup(stage="fit")

        return datamodule

    @pytest.mark.requiresdata
    def test_roi_data(self, roi_datamodule):
        dataloader = roi_datamodule.train_dataloader()
        data_iterator = iter(dataloader)

        for _i in range(10):
            inputs, targets = next(data_iterator)

            for k, v in inputs.items():
                print(k, v.dtype, v.shape)

            for k, v in targets.items():
                print(k, v.dtype, v.shape)
