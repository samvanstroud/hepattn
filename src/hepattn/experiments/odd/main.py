from lightning.pytorch.cli import ArgsType

from hepattn.experiments.odd.data import ODDDataModule
from hepattn.experiments.odd.model import ODDModel
from hepattn.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=ODDModel,
        datamodule_class=ODDDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
