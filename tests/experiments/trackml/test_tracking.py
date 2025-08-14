from hepattn.experiments.trackml import run_tracking


def test_run_tracking():
    args = ["fit", "--config", "tests/experiments/trackml/test_tracking.yaml"]
    run_tracking.main(args)
