from hepattn.experiments.clic import main


def test_run_tracking():
    args = ["fit", "--config", "tests/experiments/clic/test_clic.yaml"]
    main.main(args)
