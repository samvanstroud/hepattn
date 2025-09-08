from hepattn.experiments.clic import main

from ..utils import run_test  # noqa: TID252


def run_test_pflow() -> None:
    run_test(main, "tests/experiments/clic/test_clic.yaml")
