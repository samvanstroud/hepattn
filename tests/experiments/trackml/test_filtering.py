from hepattn.experiments.trackml import run_filtering


def test_run_filtering():
    args = ["fit", "--config", "tests/experiments/trackml/test_filtering.yaml"]
    run_filtering.main(args)
