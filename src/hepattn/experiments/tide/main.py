import torch
from lightning.pytorch.cli import ArgsType

from hepattn.experiments.tide.data import ROIDataModule
from hepattn.experiments.tide.model import TIDEModel
from hepattn.utils.cli import CLI

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TIDEModel,
        datamodule_class=ROIDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
