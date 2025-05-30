import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from jsonargparse.typing import register_type
from lightning.pytorch.cli import LightningCLI


# For info on why this is needed, see here
# https://docs.pytorch.org/docs/stable/multiprocessing.html#file-system-file-system
torch.multiprocessing.set_sharing_strategy("file_system")

# Similarly, see here for this
# https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")


# Add support for converting yaml lists to tensors
def serializer(x: torch.Tensor) -> list:
    return x.tolist()


def deserializer(x: list) -> torch.Tensor:
    return torch.tensor(x)


register_type(torch.Tensor, serializer, deserializer)


def get_best_epoch(config_path: Path) -> Path:
    """Find the best perfoming epoch.

    Parameters
    ----------
    config_path : Path
        Path to saved training config file.

    Returns
    -------
    Path
        Path to best checkpoint for the training run.
    """
    ckpt_dir = Path(config_path.parent / "ckpts")
    print(f"No --ckpt_path specified, looking for best checkpoint in {ckpt_dir.resolve()!r}")
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir.resolve()!r}")
    exp = r"(?<=loss=)(?:(?:\d+(?:\.\d*)?|\.\d+))"
    losses = [float(re.findall(exp, Path(ckpt).name)[0]) for ckpt in ckpts]
    ckpt = ckpts[np.argmin(losses)]
    print(f"Using checkpoint {ckpt.resolve()!r}")
    return ckpt


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--name", default="hepformer", help="Name for this training run.")
        parser.link_arguments("name", "trainer.logger.init_args.experiment_name")
        parser.link_arguments("name", "model.name")
        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")

    def before_instantiate_classes(self) -> None:
        sc = self.config[self.subcommand]

        if self.subcommand == "fit":
            # Get timestamped output dir for this run
            timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")  # noqa: DTZ005
            log = "trainer.logger"
            name = sc["name"]
            log_dir = Path(sc["trainer.default_root_dir"])

            # Handle case where we re-use an existing config: use parent of timestampped dir
            try:
                datetime.strptime(log_dir.name.split("_")[-1], "%Y%m%d-T%H%M%S")  # noqa: DTZ007
                log_dir = log_dir.parent
            except ValueError:
                pass

            # Set the timestampped dir
            dirname = f"{name}_{timestamp}"
            log_dir_timestamp = str(Path(log_dir / dirname).resolve())
            sc["trainer.default_root_dir"] = log_dir_timestamp
            if sc[log]:
                sc[f"{log}.init_args.save_dir"] = log_dir_timestamp

        if self.subcommand == "test":
            # Modify callbacks when testing
            self.save_config_callback = None
            sc["trainer.logger"] = False
            for c in sc["trainer.callbacks"]:
                if hasattr(c, "init_args") and hasattr(c.init_args, "refresh_rate"):
                    c.init_args.refresh_rate = 1

            # Use the best epoch for testing
            if sc["ckpt_path"] is None:
                config = sc["config"]
                assert len(config) == 1
                best_epoch_path = get_best_epoch(Path(config[0].rel_path))
                sc["ckpt_path"] = best_epoch_path

            # Ensure only one device is used for testing
            n_devices = sc["trainer.devices"]
            if (isinstance(n_devices, str | int)) and int(n_devices) > 1:
                print("Setting --trainer.devices=1")
                sc["trainer.devices"] = "1"
            if isinstance(n_devices, list) and len(n_devices) > 1:
                raise ValueError("Testing requires --trainer.devices=1")
