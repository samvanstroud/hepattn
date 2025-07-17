"""Top level training script, powered by the lightning CLI."""

import pathlib

from lightning.pytorch.cli import ArgsType

from hepattn.experiments.clic.lightning_module import MPflow
from hepattn.experiments.clic.pflow_data import PflowDataModule
from hepattn.utils.cli import CLI

config_dir = pathlib.Path(__file__).parent / "configs"


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=MPflow,
        datamodule_class=PflowDataModule,
        args=args,
        parser_kwargs={"default_env": True, "fit": {"default_config_files": [f"{config_dir}/base.yaml"]}},
    )


if __name__ == "__main__":
    main()
