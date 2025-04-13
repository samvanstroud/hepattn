from lightning.pytorch.cli import ArgsType

from hepattn.experiments.trackml.data import TrackMLDataModule
from hepattn.experiments.trackml.filter import TrackMLFilter
from hepattn.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLFilter,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
