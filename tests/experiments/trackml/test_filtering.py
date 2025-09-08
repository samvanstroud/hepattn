from hepattn.experiments.trackml import run_filtering


def run_test_filtering():
    args = ["fit", "--config", "tests/experiments/trackml/test_filtering.yaml"]
    run_filtering.main(args)
