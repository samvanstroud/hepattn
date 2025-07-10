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
        config["batch_size"] = 2
        config["num_test"] = 10

        datamodule = ROIDataModule(**config)
        datamodule.setup(stage="test")

        return datamodule

    def test_plot_roi_hit_vars(self, roi_datamodule):
        dataloader = roi_datamodule.test_dataloader()
        data_iterator = iter(dataloader)

        output_dir = Path("tests/outputs/tide/")
        output_dir.mkdir(exist_ok=True, parents=True)

        inputs, _ = next(data_iterator)

        # Plot the global hit cluster coordinates

        hits = ["pix", "sct"]
        hit_aliases = {"pix": "Pixel", "sct": "SCT"}
        field_aliases = {
            "x": r"$x$",
            "y": r"$y$",
            "z": r"$z$",
            "r": r"$r$",
            "theta": r"$\theta$",
            "eta": r"$\eta$",
            "phi": r"$\phi$",
            "dtheta": r"$\Delta \theta$",
            "deta": r"$\Delta \eta$",
            "dphi": r"$\Delta \phi$",
        }

        fields = ["x", "y", "z", "r", "theta", "eta", "phi"]

        fig, ax = plt.subplots(len(hits), len(fields))
        fig.set_size_inches(12, 4)

        for i, hit in enumerate(hits):
            for j, field in enumerate(fields):
                ax[i, j].hist(inputs[f"{hit}_{field}"][inputs[f"{hit}_valid"]], bins=32, histtype="step")
                ax[i, j].set_xlabel(rf"{hit_aliases[hit]} {field_aliases[field]}")
                ax[i, j].set_ylabel("Count")
                ax[i, j].set_yscale("log")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_hit_global_coords.png"))

        # Plot the global module coordinates
        fields = ["x", "y", "z", "r", "theta", "eta", "phi"]

        fig, ax = plt.subplots(len(hits), len(fields))
        fig.set_size_inches(12, 4)

        for i, hit in enumerate(hits):
            for j, field in enumerate(fields):
                ax[i, j].hist(inputs[f"{hit}_mod_{field}"][inputs[f"{hit}_valid"]], bins=32, histtype="step")
                ax[i, j].set_xlabel(rf"{hit_aliases[hit]} Module {field_aliases[field]}")
                ax[i, j].set_ylabel("Count")
                ax[i, j].set_yscale("log")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_module_global_coords.png"))

        # Plot the hit cluster coordinates in the local ROI frame
        fields = ["dtheta", "deta", "dphi"]

        fig, ax = plt.subplots(len(hits), len(fields))
        fig.set_size_inches(12, 4)

        for i, hit in enumerate(hits):
            for j, field in enumerate(fields):
                ax[i, j].hist(inputs[f"{hit}_{field}"][inputs[f"{hit}_valid"]], bins=32, histtype="step")
                ax[i, j].set_xlabel(rf"{hit_aliases[hit]} ROI {field_aliases[field]}")
                ax[i, j].set_ylabel("Count")
                ax[i, j].set_yscale("log")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_hit_local_coords.png"))

    def test_plot_roi_track_vars(self, roi_datamodule):
        dataloader = roi_datamodule.test_dataloader()
        data_iterator = iter(dataloader)

        output_dir = Path("tests/outputs/tide/")

        _, targets = next(data_iterator)

        tracks = ["sudo", "sisp", "reco"]
        track_aliases = {"sudo": "Pseudo Track", "sisp": "SiSp", "reco": "Reco"}
        field_aliases = {
            "vx": r"$v_x$",
            "vy": r"$v_y$",
            "vz": r"$v_z$",
            "z0": r"$z_0$",
            "d0": r"$d_0$",
            "pt": r"$p_T$",
            "theta": r"$\theta$",
            "eta": r"$\eta$",
            "phi": r"$\phi$",
            "qopt": r"$q/p_T$",
            "deta": r"$\Delta \eta$",
            "dtheta": r"$\Delta \theta$",
            "dphi": r"$\Delta \phi$",
            "dz0": r"$\Delta z_0$",
        }

        fields = ["pt", "eta", "theta", "phi", "z0", "d0", "qopt"]

        fig, ax = plt.subplots(len(tracks), len(fields))
        fig.set_size_inches(16, 4)

        for i, track in enumerate(tracks):
            for j, field in enumerate(fields):
                ax[i, j].hist(targets[f"{track}_{field}"][targets[f"{track}_valid"]], bins=24, histtype="step")
                ax[i, j].set_xlabel(rf"{track_aliases[track]} {field_aliases[field]}")
                ax[i, j].set_ylabel("Count")
                ax[i, j].set_yscale("log")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_track_global_coords.png"))

        fields = ["deta", "dtheta", "dphi", "dz0"]

        fig, ax = plt.subplots(len(tracks), len(fields))
        fig.set_size_inches(12, 4)

        for i, track in enumerate(tracks):
            for j, field in enumerate(fields):
                ax[i, j].hist(targets[f"{track}_{field}"][targets[f"{track}_valid"]], bins=24, histtype="step")
                ax[i, j].set_xlabel(rf"{track_aliases[track]} {field_aliases[field]}")
                ax[i, j].set_ylabel("Count")
                ax[i, j].set_yscale("log")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_track_local_coords.png"))

        fields = ["loc_x", "loc_y", "phi", "theta", "energy"]

        fig, ax = plt.subplots(nrows=1, ncols=len(fields))
        fig.set_size_inches(12, 3)

        for j, field in enumerate(fields):
            ax[j].hist(targets[f"sudo_pix_{field}"][targets["sudo_pix_valid"]], bins=32, histtype="step")
            ax[j].set_xlabel(rf"{field}")
            ax[j].set_ylabel("Count")
            ax[j].set_yscale("log")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_sudo_pix_fields.png"))
