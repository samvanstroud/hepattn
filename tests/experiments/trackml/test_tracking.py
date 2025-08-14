from hepattn.experiments.trackml import run_tracking


def test_run_tracking():
    args = [
        "fit",
        "--config",
        "tests/experiments/trackml/test_tracking.yaml",
        "--trainer.fast_dev_run",
        "1",
        "--data.dummy_data",
        "True",
        "--trainer.logger",
        "False",
    ]
    run_tracking.main(args)
