import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from jsonargparse.typing import register_type
from lightning.pytorch.cli import LightningCLI

torch._dynamo.config.capture_scalar_outputs = True  # noqa: SLF001


# Add support for converting yaml lists to tensors
def serializer(x: torch.Tensor) -> list:
    return x.tolist()


def deserializer(x: list) -> torch.Tensor:
    return torch.tensor(x)


register_type(torch.Tensor, serializer, deserializer)


def get_best_epoch(config_path: Path) -> Path:
    """Find the best perfoming epoch.

    Args:
        config_path (Path): Path to saved training config file.

    Returns:
        Path: Path to best checkpoint for the training run.

    Raises:
        FileNotFoundError: If no checkpoints are found in the expected directory.
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
        parser.add_argument("--name", type=str, default="hepattn", help="Name for this training run.")

        parser.add_argument(
            "--matmul_precision",
            type=str,
            choices=["highest", "high", "medium", "low"],
            help="Precision setting for float32 matrix multiplications.",
        )

        parser.link_arguments("name", "model.name")
        parser.link_arguments("name", "trainer.logger.init_args.name")

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
                sc[f"{log}.init_args.offline_directory"] = log_dir_timestamp

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
                best_epoch_path = get_best_epoch(Path(config[0].relative))
                sc["ckpt_path"] = best_epoch_path

            # Ensure only one device is used for testing
            n_devices = sc["trainer.devices"]
            if (isinstance(n_devices, str | int)) and int(n_devices) > 1:
                print("Setting --trainer.devices=1")
                sc["trainer.devices"] = "1"
            if isinstance(n_devices, list) and len(n_devices) > 1:
                raise ValueError("Testing requires --trainer.devices=1")

        # Set the matmul precision
        if sc.get("matmul_precision"):
            torch.set_float32_matmul_precision(sc["matmul_precision"])

    def after_instantiate_classes(self) -> None:
        sc = self.config[self.subcommand]

        if self.subcommand == "test":
            ckpt_path = sc["ckpt_path"] or get_best_epoch(Path(sc["config"][0].relative))
            # Workaround to store ckpt dir for prediction writer since trainer.ckpt_path gets set to none somewhere
            # TODO: Figure out what causes trainer.ckpt_path to be set to none
            self.trainer.ckpt_path = ckpt_path
            self.trainer.ckpt_dir = Path(ckpt_path).parent
            self.trainer.ckpt_name = str(Path(ckpt_path).stem)
