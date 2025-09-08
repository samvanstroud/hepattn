import os
from pathlib import Path

from hepattn.experiments.trackml import run_tracking


def test_run_tracking():
    args = ["fit", "--config", "tests/experiments/trackml/test_tracking.yaml"]
    run_tracking.main(args)

    # Find the most recent experiment directory
    test_logs_dir = Path("test_logs")
    assert test_logs_dir.exists(), f"test_logs directory not found at {test_logs_dir.absolute()}"

    # Get all directories that start with 'tracking-test_'
    exp_dirs = [d for d in test_logs_dir.iterdir() if d.is_dir() and d.name.startswith("tracking-test_")]
    assert exp_dirs, f"No experiment directories found in {test_logs_dir.absolute()} starting with 'tracking-test_'"

    # Get the most recent directory by modification time
    latest_dir = max(exp_dirs, key=os.path.getmtime)
    config_path = latest_dir / "config.yaml"
    assert config_path.exists(), f"Config file not found at {config_path.absolute()}"

    args = ["test", "--config", str(config_path)]
    run_tracking.main(args)


def test_run_tracking_old_sort():
    args = ["fit", "--config", "tests/experiments/trackml/test_tracking_old_sort.yaml"]
    run_tracking.main(args)

    # Find the most recent experiment directory
    test_logs_dir = Path("test_logs")
    assert test_logs_dir.exists(), f"test_logs directory not found at {test_logs_dir.absolute()}"

    # Get all directories that start with 'tracking-test_'
    exp_dirs = [d for d in test_logs_dir.iterdir() if d.is_dir() and d.name.startswith("tracking-test-old-sort_")]
    assert exp_dirs, f"No experiment directories found in {test_logs_dir.absolute()} starting with 'tracking-test-old-sort_'"

    # Get the most recent directory by modification time
    latest_dir = max(exp_dirs, key=os.path.getmtime)
    config_path = latest_dir / "config.yaml"
    assert config_path.exists(), f"Config file not found at {config_path.absolute()}"

    args = ["test", "--config", str(config_path)]
    run_tracking.main(args)
