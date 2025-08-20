from hepattn.experiments.trackml import run_tracking


def test_run_tracking():
    args = ["fit", "--config", "tests/experiments/trackml/test_tracking.yaml"]
    run_tracking.main(args)


def test_run_tracking_old_sort():
    args = ["fit", "--config", "tests/experiments/trackml/test_tracking_old_sort.yaml"]
    run_tracking.main(args)
