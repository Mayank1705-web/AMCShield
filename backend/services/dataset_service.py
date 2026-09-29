from pathlib import Path
from typing import Any

class DatasetService:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def overview(self) -> dict[str, Any]:
        files = []
        if self.data_dir.exists():
            files = sorted(
                str(p.relative_to(self.data_dir))
                for p in self.data_dir.rglob("*")
                if p.is_file()
            )
        return {
            "path": str(self.data_dir),
            "exists": self.data_dir.exists(),
            "files": files,
        }
