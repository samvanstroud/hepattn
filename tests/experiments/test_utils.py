import os
from pathlib import Path
from typing import Any


def test_run(main_module: Any, config_path: str, dir_prefix: str, logs_dir: str = "logs"):
    """Run an experiment test with the given main module, config, and directory prefix."""
    args = ["fit", "--config", config_path]
    main_module.main(args)

    # Find the most recent experiment directory
    test_logs_dir = Path(logs_dir)
    assert test_logs_dir.exists(), f"{logs_dir} directory not found at {test_logs_dir.absolute()}"

    # Get all directories that start with the given prefix
    exp_dirs = [d for d in test_logs_dir.iterdir() if d.is_dir() and d.name.startswith(dir_prefix)]
    assert exp_dirs, f"No experiment directories found in {test_logs_dir.absolute()} starting with '{dir_prefix}'"

    # Get the most recent directory by modification time
    latest_dir = max(exp_dirs, key=os.path.getmtime)
    config_path = latest_dir / "config.yaml"
    assert config_path.exists(), f"Config file not found at {config_path.absolute()}"

    args = ["test", "--config", str(config_path)]
    main_module.main(args)
