from hepattn.experiments.atlas_muon import run_tracking

from ..utils import run_test  # noqa: TID252


def test_tracking() -> None:
    run_test(run_tracking, "tests/experiments/atlas_muon/test_tracking.yaml")
