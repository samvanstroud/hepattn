from lightning.pytorch.cli import ArgsType

from hepattn.experiments.colliderml.data import ColliderMLDataModule
from hepattn.experiments.colliderml.model import ColliderMLModel
from hepattn.utils.cli import CLI


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=ColliderMLModel,
        datamodule_class=ColliderMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
