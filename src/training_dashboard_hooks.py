"""Optional stop-file integration for dashboard-launched training."""
import os
from pathlib import Path


def should_stop_training() -> bool:
    control_dir = os.environ.get("AMC_TRAINING_CONTROL_DIR")
    if not control_dir:
        return False
    return (Path(control_dir) / "stop.request").is_file()