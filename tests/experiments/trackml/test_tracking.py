from tests.experiments.test_utils import run_experiment_test

from hepattn.experiments.trackml import run_tracking


def test_run_tracking():
    run_experiment_test(run_tracking, "tests/experiments/trackml/test_tracking.yaml", "tracking-test_", "test_logs")


def test_run_tracking_old_sort():
    run_experiment_test(run_tracking, "tests/experiments/trackml/test_tracking_old_sort.yaml", "tracking-test-old-sort_", "test_logs")
