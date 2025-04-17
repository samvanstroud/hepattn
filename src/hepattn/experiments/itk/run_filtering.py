from lightning.pytorch.cli import ArgsType

from hepattn.experiments.itk.data import ITkDataModule
from hepattn.experiments.itk.filter import ITkFilter
from hepattn.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=ITkFilter,
        datamodule_class=ITkDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
