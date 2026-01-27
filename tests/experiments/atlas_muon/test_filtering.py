from hepattn.experiments.atlas_muon import run_filtering


def test_filtering():
    args = ["fit", "--config", "tests/experiments/atlas_muon/test_filtering.yaml"]
    run_filtering.main(args)
