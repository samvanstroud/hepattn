from hepattn.experiments.clic import main
from tests.experiments.test_utils import run_test


def run_test_pflow() -> None:
    run_test(main, "tests/experiments/clic/test_clic.yaml")
