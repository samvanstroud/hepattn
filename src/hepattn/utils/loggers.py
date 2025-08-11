from lightning.pytorch.loggers import CometLogger


class MyCometLogger(CometLogger):
    """Wrap CometLogger to fix issues with CLI arguments."""

    def __init__(self, name: str, offline_directory: str | None = None, **kwargs):
        assert offline_directory is not None, "offline_directory must be specified for MyCometLogger"
        super().__init__(name=name, offline_directory=offline_directory, **kwargs)
