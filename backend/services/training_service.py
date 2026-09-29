from pathlib import Path
from typing import Any

class TrainingService:
    """Reports training/checkpoint state without starting training automatically."""

    def __init__(self, checkpoints_dir: Path, src_dir: Path):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.src_dir = Path(src_dir)

    def status(self) -> dict[str, Any]:
        checkpoints = []
        if self.checkpoints_dir.exists():
            for path in sorted(self.checkpoints_dir.rglob("*")):
                if path.is_file():
                    stat = path.stat()
                    checkpoints.append({
                        "name": str(path.relative_to(self.checkpoints_dir)),
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                    })
        return {
            "training_control": "external",
            "src_exists": self.src_dir.exists(),
            "checkpoints_dir": str(self.checkpoints_dir),
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
        }
