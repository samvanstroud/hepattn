from hepattn.experiments.trackml import run_tracking

from ..test_utils import run_test  # noqa: TID252


def run_test_tracking() -> None:
    run_test(run_tracking, "tests/experiments/trackml/test_tracking.yaml")


def run_test_tracking_old_sort() -> None:
    run_test(run_tracking, "tests/experiments/trackml/test_tracking_old_sort.yaml")
