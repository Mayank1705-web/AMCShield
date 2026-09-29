from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

class ResultsService:
    """Read-only access to AMCShield's existing results artifacts."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)

    def exists(self) -> bool:
        return self.results_dir.exists()

    def list_files(self) -> list[str]:
        if not self.results_dir.exists():
            return []
        return sorted(
            str(p.relative_to(self.results_dir))
            for p in self.results_dir.rglob("*")
            if p.is_file()
        )

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.results_dir),
            "exists": self.exists(),
            "file_count": len(self.list_files()),
            "files": self.list_files(),
        }

    def read_json(self, filename: str) -> Any:
        path = self._safe_path(filename)
        return json.loads(path.read_text(encoding="utf-8"))

    def read_csv(self, filename: str) -> list[dict[str, Any]]:
        path = self._safe_path(filename)
        frame = pd.read_csv(path)
        return frame.where(pd.notna(frame), None).to_dict(orient="records")

    def _safe_path(self, filename: str) -> Path:
        root = self.results_dir.resolve()
        path = (root / filename).resolve()
        if root != path and root not in path.parents:
            raise ValueError("Invalid results path")
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path
