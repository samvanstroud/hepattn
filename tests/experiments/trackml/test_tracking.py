from hepattn.experiments.trackml import run_tracking

from ..test_utils import test_run  # noqa: TID252


def test_run_tracking() -> None:
    test_run(run_tracking, "tests/experiments/trackml/test_tracking.yaml", "tracking-test_", "test_logs")


def test_run_tracking_old_sort() -> None:
    test_run(run_tracking, "tests/experiments/trackml/test_tracking_old_sort.yaml", "tracking-test-old-sort_", "test_logs")
