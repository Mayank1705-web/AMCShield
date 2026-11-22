
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ResultsService:
    """Read-only analytics service for AMCShield evaluation artifacts."""

    ATTACKS = ("fgsm", "pgd", "mim", "cw", "blackbox")

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)

    def exists(self) -> bool:
        return self.results_dir.is_dir()

    def list_files(self) -> list[str]:
        if not self.exists():
            return []
        return sorted(
            str(path.relative_to(self.results_dir))
            for path in self.results_dir.rglob("*")
            if path.is_file()
        )

    def summary(self) -> dict[str, Any]:
        return {
            "exists": self.exists(),
            "file_count": len(self.list_files()),
            "files": self.list_files(),
        }

    def _safe_path(self, filename: str) -> Path:
        root = self.results_dir.resolve()
        path = (root / filename).resolve()

        if root != path and root not in path.parents:
            raise ValueError("Invalid results path")

        if not path.is_file():
            raise FileNotFoundError(filename)

        return path

    def read_json(self, filename: str) -> Any:
        path = self._safe_path(filename)
        return json.loads(path.read_text(encoding="utf-8"))

    def read_csv(self, filename: str) -> list[dict[str, Any]]:
        frame = pd.read_csv(self._safe_path(filename))
        return frame.where(pd.notna(frame), None).to_dict(orient="records")

    def _load(self) -> pd.DataFrame:
        """Load and validate the consolidated, percentage-scale results."""
        frame = pd.read_csv(self._safe_path("master_results.csv"))

        required = {
            "snr",
            "clean_accuracy",
            "baseline_clean_accuracy",
        }

        for attack in self.ATTACKS:
            required.update({
                f"{attack}_accuracy",
                f"{attack}_asr",
                f"baseline_{attack}_accuracy",
                f"baseline_{attack}_asr",
                f"{attack}_gap",
            })

        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"master_results.csv is missing required columns: "
                f"{sorted(missing)}"
            )

        if frame.empty:
            raise ValueError("master_results.csv contains no evaluation rows")

        frame = frame.sort_values("snr").reset_index(drop=True)

        if frame["snr"].duplicated().any():
            raise ValueError("master_results.csv contains duplicate SNR levels")

        return frame

    def _clean_counts(self) -> dict[str, int] | None:
        """Return sample counts only when the original clean CSV provides them."""
        path = self.results_dir / "clean_results.csv"

        if not path.is_file():
            return None

        frame = pd.read_csv(path)
        if not {"correct", "total"}.issubset(frame.columns):
            return None

        return {
            "correct": int(frame["correct"].sum()),
            "total": int(frame["total"].sum()),
        }

    def _metadata(self) -> dict[str, Any]:
        """Read optional metadata without requiring a nonexistent metadata.json."""
        path = self.results_dir / "blackbox_results_metadata.json"

        if not path.is_file():
            return {}

        try:
            return self.read_json("blackbox_results_metadata.json")
        except (ValueError, OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _mean(frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None

        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return frame.where(pd.notna(frame), None).to_dict(orient="records")

    def evaluation_summary(self) -> dict[str, Any]:
        frame = self._load()
        counts = self._clean_counts()
        metadata = self._metadata()

        snrs = sorted(int(value) for value in frame["snr"].unique())

        attack_scope = list(self.ATTACKS)

        robust_asrs = [
            self._mean(frame, f"{attack}_asr")
            for attack in attack_scope
        ]
        baseline_asrs = [
            self._mean(frame, f"baseline_{attack}_asr")
            for attack in attack_scope
        ]

        robust_asrs = [value for value in robust_asrs if value is not None]
        baseline_asrs = [value for value in baseline_asrs if value is not None]

        return {
            "robust_clean_accuracy": self._mean(frame, "clean_accuracy"),
            "robust_samples": counts["total"] if counts else None,
            "baseline_clean_accuracy": self._mean(
                frame, "baseline_clean_accuracy"
            ),
            "baseline_samples": None,
            "mean_robust_asr": (
                sum(robust_asrs) / len(robust_asrs)
                if robust_asrs else None
            ),
            "mean_baseline_asr": (
                sum(baseline_asrs) / len(baseline_asrs)
                if baseline_asrs else None
            ),
            "attack_scope": attack_scope,
            "mean_gap": (
                (
                    sum(baseline_asrs) / len(baseline_asrs)
                    - sum(robust_asrs) / len(robust_asrs)
                )
                if robust_asrs and baseline_asrs else None
            ),
            "gap_definition": (
                "Mean baseline ASR minus mean robust ASR across SNR levels "
                "and attack methods; percentage points."
            ),
            "snr_levels": snrs,
            "snr_min": min(snrs) if snrs else None,
            "snr_max": max(snrs) if snrs else None,
            "source": "master_results.csv",
            "status": "available",
            "message": (
                "Accuracy and attack success rates are percentages. "
                "ASR summaries are arithmetic means across SNR levels "
                "and attack methods, not pooled sample-weighted rates."
            ),
            "test_samples": counts["total"] if counts else None,
            "num_classes": 24,
            "dataset": metadata.get("dataset"),
            "attacks": attack_scope,
            "total_attack_samples": None,
            "total_originally_correct": None,
            "total_flipped": None,
        }

    def evaluation_filters(self) -> dict[str, Any]:
        frame = self._load()
        return {
            "models": ["baseline", "robust"],
            "attacks": list(self.ATTACKS),
            "snr_levels": sorted(int(value) for value in frame["snr"].unique()),
        }

    def evaluation_comparison(self) -> list[dict[str, Any]]:
        frame = self._load()
        rows = []

        for attack in self.ATTACKS:
            baseline_asr = self._mean(frame, f"baseline_{attack}_asr")
            robust_asr = self._mean(frame, f"{attack}_asr")

            rows.append({
                "attack": attack,
                "baseline_asr": baseline_asr,
                "robust_asr": robust_asr,
            })

        return rows

    def evaluation_attack_summary(self) -> list[dict[str, Any]]:
        frame = self._load()
        rows = []

        for attack in self.ATTACKS:
            baseline_asr = self._mean(frame, f"baseline_{attack}_asr")
            robust_asr = self._mean(frame, f"{attack}_asr")
            gap = self._mean(frame, f"{attack}_gap")

            rows.append({
                "attack": attack,
                "baseline_asr": baseline_asr,
                "robust_asr": robust_asr,
                "gap": gap,
                "generalization_gap": gap,
                "mean_per_snr_gap": gap,
            })

        return rows

    def evaluation_results(
        self,
        attack: str | None = None,
        snr: int | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        frame = self._load()
        rows = []

        selected_attacks = (
            [attack.lower()]
            if attack and attack.lower() != "all"
            else list(self.ATTACKS)
        )

        if any(item not in self.ATTACKS for item in selected_attacks):
            return []

        if model and model.lower() not in ("all", "baseline", "robust"):
            return []

        if snr is not None:
            frame = frame[frame["snr"] == snr]

        for _, source_row in frame.iterrows():
            snr_value = int(source_row["snr"])

            for attack_name in selected_attacks:
                for model_name in ("baseline", "robust"):
                    if model and model.lower() != "all":
                        if model.lower() != model_name:
                            continue

                    prefix = "baseline_" if model_name == "baseline" else ""

                    accuracy_column = f"{prefix}{attack_name}_accuracy"
                    asr_column = f"{prefix}{attack_name}_asr"

                    result = {
                        "model": model_name,
                        "attack": attack_name,
                        "snr": snr_value,
                        "samples": None,
                        "baseline_accuracy": (
                            self._number(source_row["baseline_clean_accuracy"])
                            if model_name == "baseline" else None
                        ),
                        "robust_accuracy": (
                            self._number(source_row["clean_accuracy"])
                            if model_name == "robust" else None
                        ),
                        "clean_accuracy": (
                            self._number(
                                source_row[
                                    "baseline_clean_accuracy"
                                    if model_name == "baseline"
                                    else "clean_accuracy"
                                ]
                            )
                        ),
                        "accuracy_under_attack": self._number(
                            source_row[accuracy_column]
                        ),
                        "attack_success_rate": self._number(
                            source_row[asr_column]
                        ),
                        "baseline_asr": self._number(
                            source_row[f"baseline_{attack_name}_asr"]
                        ),
                        "robust_asr": self._number(
                            source_row[f"{attack_name}_asr"]
                        ),
                        "gap": self._number(source_row[f"{attack_name}_gap"]),
                        "mean_linf": None,
                        "mean_l2": None,
                    }

                    for metric in ("mean_linf", "mean_l2"):
                        column = f"{attack_name}_{metric}"
                        if model_name == "baseline":
                            column = f"baseline_{attack_name}_{metric}"

                        if column in source_row.index:
                            result[metric] = self._number(source_row[column])

                    rows.append(result)

        return rows

    def evaluation_accuracy_by_snr(self) -> list[dict[str, Any]]:
        frame = self._load()

        return [
            {
                "snr": int(row["snr"]),
                "baseline_accuracy": self._number(
                    row["baseline_clean_accuracy"]
                ),
                "robust_accuracy": self._number(row["clean_accuracy"]),
            }
            for _, row in frame.iterrows()
        ]

    def evaluation_generalization_gap(self) -> list[dict[str, Any]]:
        frame = self._load()
        gap_columns = [f"{attack}_gap" for attack in self.ATTACKS]

        rows = []
        for _, row in frame.iterrows():
            values = [
                self._number(row[column])
                for column in gap_columns
            ]
            values = [value for value in values if value is not None]

            rows.append({
                "snr": int(row["snr"]),
                "gap": sum(values) / len(values) if values else None,
            })

        return rows

    def evaluation_insights(self) -> list[dict[str, Any]]:
        comparison = self.evaluation_comparison()
        clean_summary = self.evaluation_summary()
        findings = []

        for row in comparison:
            baseline = row["baseline_asr"]
            robust = row["robust_asr"]

            if baseline is None or robust is None:
                continue

            difference = robust - baseline
            direction = "lower" if difference < 0 else "higher"

            findings.append({
                "type": "comparison",
                "title": f"{row['attack'].upper()} mean ASR comparison",
                "description": (
                    f"Robust model mean ASR is {abs(difference):.2f} "
                    f"percentage points {direction} than baseline across "
                    f"the evaluated SNR levels."
                ),
            })

        baseline_clean = clean_summary["baseline_clean_accuracy"]
        robust_clean = clean_summary["robust_clean_accuracy"]

        if baseline_clean is not None and robust_clean is not None:
            findings.append({
                "type": "clean_accuracy",
                "title": "Clean accuracy comparison",
                "description": (
                    f"Mean clean accuracy across SNR levels: baseline "
                    f"{baseline_clean:.2f}%, robust {robust_clean:.2f}%."
                ),
            })

        return findings

    def dashboard_performance(self) -> list[dict[str, Any]]:
        return self.evaluation_accuracy_by_snr()

    def dashboard_attacks(self) -> list[dict[str, Any]]:
        frame = self._load()
        rows = []

        for _, row in frame.iterrows():
            for attack in self.ATTACKS:
                rows.append({
                    "attack": attack,
                    "snr": int(row["snr"]),
                    "baseline_asr": self._number(
                        row[f"baseline_{attack}_asr"]
                    ),
                    "robust_asr": self._number(row[f"{attack}_asr"]),
                })

        return rows

    def dashboard_generalization_gap(self) -> list[dict[str, Any]]:
        frame = self._load()
        rows = []

        for _, row in frame.iterrows():
            for attack in self.ATTACKS:
                rows.append({
                    "attack": attack,
                    "snr": int(row["snr"]),
                    "gap": self._number(row[f"{attack}_gap"]),
                })

        return rows

    def dashboard_snr(self) -> list[dict[str, Any]]:
        frame = self._load()
        rows = []

        for _, source_row in frame.iterrows():
            row = {
                "snr": int(source_row["snr"]),
                "baseline_accuracy": self._number(
                    source_row["baseline_clean_accuracy"]
                ),
                "robust_accuracy": self._number(
                    source_row["clean_accuracy"]
                ),
            }

            for attack in self.ATTACKS:
                row[f"baseline_{attack}_asr"] = self._number(
                    source_row[f"baseline_{attack}_asr"]
                )
                row[f"robust_{attack}_asr"] = self._number(
                    source_row[f"{attack}_asr"]
                )
                row[f"{attack}_asr"] = self._number(
                    source_row[f"{attack}_asr"]
                )

            rows.append(row)

        return rows

    def dashboard_summary(self) -> dict[str, Any]:
        frame = self._load()
        metadata = self._metadata()
        counts = self._clean_counts()
        snrs = sorted(int(value) for value in frame["snr"].unique())

        return {
            "dataset": {
                "name": metadata.get("dataset"),
                "num_classes": 24,
                "snr_min": min(snrs) if snrs else None,
                "snr_max": max(snrs) if snrs else None,
            },
            "evaluation": {
                "test_samples": counts["total"] if counts else None,
                "snr_levels": len(snrs),
                "attack_methods": list(self.ATTACKS),
            },
            "attacks": list(self.ATTACKS),
        }

    def dashboard_recent_updates(self) -> list[dict[str, Any]]:
        # The evaluation artifacts do not contain dated activity records.
        return []
