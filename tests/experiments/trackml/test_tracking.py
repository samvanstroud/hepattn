from hepattn.experiments.trackml import run_tracking

from ..utils import run_test  # noqa: TID252


def test_tracking() -> None:
    run_test(run_tracking, "tests/experiments/trackml/test_tracking.yaml")


def test_tracking_old_sort() -> None:
    run_test(run_tracking, "tests/experiments/trackml/test_tracking_old_sort.yaml")
