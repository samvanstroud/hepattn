from hepattn.experiments.trackml import run_filtering


def test_run_filtering():
    args = [
        "fit",
        "--config",
        "tests/experiments/trackml/test_filtering.yaml",
        "--trainer.fast_dev_run",
        "1",
        "--data.dummy_data",
        "True",
        "--trainer.logger",
        "False",
    ]
    run_filtering.main(args)
