from hepattn.experiments.clic import main
from tests.experiments.test_utils import test_run


def test_run_pflow() -> None:
    test_run(main, "tests/experiments/clic/test_clic.yaml")
