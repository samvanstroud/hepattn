from lightning.pytorch.cli import ArgsType

from hepattn.experiments.pixel.data import PixelClusterDataModule
from hepattn.experiments.pixel.model import PixelClusterSplitter
from hepattn.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=PixelClusterSplitter,
        datamodule_class=PixelClusterDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
