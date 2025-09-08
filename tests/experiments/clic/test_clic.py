from tests.experiments.test_utils import run_experiment_test

from hepattn.experiments.clic import main


def test_run_pflow():
    run_experiment_test(main, "tests/experiments/clic/test_clic.yaml", "clic_test_")
