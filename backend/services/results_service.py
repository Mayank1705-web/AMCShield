from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ResultsService:
    """Read-only analytics service for existing AMCShield evaluation artifacts."""

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

    def _load(self) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        metadata = self.read_json("metadata.json")
        clean = pd.DataFrame(self.read_csv("clean_per_snr.csv"))
        attacks = pd.DataFrame(self.read_csv("attack_results_per_snr.csv"))
        gaps = pd.DataFrame(self.read_csv("generalization_gap.csv"))

        required = {
            "clean": {"model", "snr_db", "n_samples", "n_correct", "accuracy"},
            "attacks": {
                "model", "attack", "snr_db", "n_samples",
                "n_originally_correct", "n_flipped",
                "accuracy_under_attack", "attack_success_rate",
            },
            "gaps": {
                "attack", "snr_db", "baseline_asr",
                "robust_asr", "gap_robust_minus_baseline",
            },
        }

        for name, frame in [
            ("clean", clean),
            ("attacks", attacks),
            ("gaps", gaps),
        ]:
            missing = required[name] - set(frame.columns)
            if missing:
                raise ValueError(
                    f"{name} CSV is missing required columns: {sorted(missing)}"
                )

        return metadata, clean, attacks, gaps

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return frame.where(pd.notna(frame), None).to_dict(orient="records")

    @staticmethod
    def _pooled_asr(frame: pd.DataFrame) -> float | None:
        denominator = int(frame["n_originally_correct"].sum())
        if denominator == 0:
            return None
        return float(frame["n_flipped"].sum() / denominator)

    def evaluation_summary(self) -> dict[str, Any]:
        metadata, clean, attacks, gaps = self._load()

        def clean_mean(model: str) -> float | None:
            rows = clean.loc[clean["model"] == model, "accuracy"]
            return float(rows.mean()) if not rows.empty else None

        def sample_total(model: str) -> int:
            rows = clean.loc[clean["model"] == model, "n_samples"]
            return int(rows.sum())

        baseline = attacks[attacks["model"] == "baseline"]
        robust = attacks[attacks["model"] == "robust"]

        baseline_asr = self._pooled_asr(baseline)
        robust_asr = self._pooled_asr(robust)

        all_originally_correct = int(attacks["n_originally_correct"].sum())
        all_flipped = int(attacks["n_flipped"].sum())
        mean_gap = (
            robust_asr - baseline_asr
            if baseline_asr is not None and robust_asr is not None
            else None
        )

        snrs = sorted(int(value) for value in clean["snr_db"].unique())

        return {
            "robust_clean_accuracy": clean_mean("robust"),
            "robust_samples": sample_total("robust"),
            "baseline_clean_accuracy": clean_mean("baseline"),
            "baseline_samples": sample_total("baseline"),
            "mean_robust_asr": robust_asr,
            "mean_baseline_asr": baseline_asr,
            "attack_scope": sorted(str(value) for value in attacks["attack"].unique()),
            "mean_gap": mean_gap,
            "gap_definition": "pooled robust ASR minus pooled baseline ASR",
            "snr_levels": snrs,
            "snr_min": min(snrs) if snrs else None,
            "snr_max": max(snrs) if snrs else None,
            "source": "evaluation CSV artifacts",
            "status": metadata.get("status", "unknown"),
            "message": "Metrics are calculated from the stored evaluation subset.",
            "test_samples": int(metadata.get("total_samples", 0)),
            "num_classes": int(metadata.get("num_classes", 0)),
            "dataset": metadata.get("dataset"),
            "attacks": sorted(str(value) for value in attacks["attack"].unique()),
            "total_attack_samples": int(attacks["n_samples"].sum()),
            "total_originally_correct": all_originally_correct,
            "total_flipped": all_flipped,
        }

    def evaluation_filters(self) -> dict[str, Any]:
        _, clean, attacks, _ = self._load()
        return {
            "models": sorted(str(value) for value in clean["model"].unique()),
            "attacks": sorted(str(value) for value in attacks["attack"].unique()),
            "snr_levels": sorted(int(value) for value in clean["snr_db"].unique()),
        }

    def evaluation_comparison(self) -> list[dict[str, Any]]:
        _, _, attacks, _ = self._load()
        rows = []

        for attack in sorted(attacks["attack"].unique()):
            baseline = attacks[
                (attacks["model"] == "baseline") &
                (attacks["attack"] == attack)
            ]
            robust = attacks[
                (attacks["model"] == "robust") &
                (attacks["attack"] == attack)
            ]

            rows.append({
                "attack": str(attack),
                "baseline_asr": self._pooled_asr(baseline),
                "robust_asr": self._pooled_asr(robust),
            })

        return rows

    def evaluation_attack_summary(self) -> list[dict[str, Any]]:
        _, _, attacks, gaps = self._load()
        rows = []

        for attack in sorted(attacks["attack"].unique()):
            baseline = attacks[
                (attacks["model"] == "baseline") &
                (attacks["attack"] == attack)
            ]
            robust = attacks[
                (attacks["model"] == "robust") &
                (attacks["attack"] == attack)
            ]

            baseline_asr = self._pooled_asr(baseline)
            robust_asr = self._pooled_asr(robust)
            gap_rows = gaps[gaps["attack"] == attack]

            rows.append({
                "attack": str(attack),
                "baseline_asr": baseline_asr,
                "robust_asr": robust_asr,
                "gap": (
                    robust_asr - baseline_asr
                    if baseline_asr is not None and robust_asr is not None
                    else None
                ),
                "generalization_gap": (
                    robust_asr - baseline_asr
                    if baseline_asr is not None and robust_asr is not None
                    else None
                ),
                "mean_per_snr_gap": (
                    float(gap_rows["gap_robust_minus_baseline"].mean())
                    if not gap_rows.empty else None
                ),
            })

        return rows

    def evaluation_results(
        self,
        attack: str | None = None,
        snr: int | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        _, clean, attacks, gaps = self._load()
        rows = []

        selected_attacks = (
            [attack] if attack and attack != "all"
            else sorted(attacks["attack"].unique())
        )

        for attack_name in selected_attacks:
            attack_rows = attacks[attacks["attack"] == attack_name]

            if snr is not None:
                attack_rows = attack_rows[attack_rows["snr_db"] == snr]

            for _, attack_row in attack_rows.iterrows():
                model_name = str(attack_row["model"])

                if model and model != "all" and model != model_name:
                    continue

                snr_value = int(attack_row["snr_db"])
                clean_row = clean[
                    (clean["model"] == model_name) &
                    (clean["snr_db"] == snr_value)
                ]

                baseline_gap = gaps[
                    (gaps["attack"] == attack_name) &
                    (gaps["snr_db"] == snr_value)
                ]

                accuracy = (
                    float(clean_row.iloc[0]["accuracy"])
                    if not clean_row.empty else None
                )

                gap_value = None
                if not baseline_gap.empty:
                    gap_value = float(
                        baseline_gap.iloc[0]["gap_robust_minus_baseline"]
                    )

                rows.append({
                    "model": model_name,
                    "attack": str(attack_name),
                    "snr": snr_value,
                    "samples": int(attack_row["n_samples"]),
                    "baseline_accuracy": (
                        accuracy if model_name == "baseline" else None
                    ),
                    "robust_accuracy": (
                        accuracy if model_name == "robust" else None
                    ),
                    "clean_accuracy": accuracy,
                    "accuracy_under_attack": float(
                        attack_row["accuracy_under_attack"]
                    ),
                    "attack_success_rate": float(
                        attack_row["attack_success_rate"]
                    ),
                    "baseline_asr": (
                        float(baseline_gap.iloc[0]["baseline_asr"])
                        if not baseline_gap.empty else None
                    ),
                    "robust_asr": (
                        float(baseline_gap.iloc[0]["robust_asr"])
                        if not baseline_gap.empty else None
                    ),
                    "gap": gap_value,
                    "mean_linf": float(attack_row["mean_linf"]),
                    "mean_l2": float(attack_row["mean_l2"]),
                })

        return rows

    def evaluation_accuracy_by_snr(self) -> list[dict[str, Any]]:
        _, clean, _, _ = self._load()
        pivot = clean.pivot(
            index="snr_db",
            columns="model",
            values="accuracy",
        ).reset_index()

        rows = []
        for _, row in pivot.iterrows():
            rows.append({
                "snr": int(row["snr_db"]),
                "baseline_accuracy": (
                    float(row["baseline"])
                    if "baseline" in pivot.columns and pd.notna(row.get("baseline"))
                    else None
                ),
                "robust_accuracy": (
                    float(row["robust"])
                    if "robust" in pivot.columns and pd.notna(row.get("robust"))
                    else None
                ),
            })

        return rows

    def evaluation_generalization_gap(self) -> list[dict[str, Any]]:
        _, _, _, gaps = self._load()
        result = gaps.groupby("snr_db", as_index=False)[
            "gap_robust_minus_baseline"
        ].mean()

        return [
            {"snr": int(row["snr_db"]), "gap": float(row["gap_robust_minus_baseline"])}
            for _, row in result.iterrows()
        ]

    def evaluation_insights(self) -> list[dict[str, Any]]:
        comparison = self.evaluation_comparison()
        clean_summary = self.evaluation_summary()
        findings = []

        for row in comparison:
            if row["baseline_asr"] is None or row["robust_asr"] is None:
                continue

            difference = row["robust_asr"] - row["baseline_asr"]
            direction = "lower" if difference < 0 else "higher"

            findings.append({
                "type": "comparison",
                "title": f"{row['attack'].upper()} pooled ASR comparison",
                "description": (
                    f"Robust model pooled ASR is {abs(difference) * 100:.2f} "
                    f"percentage points {direction} than baseline on this evaluation subset."
                ),
            })

        findings.append({
            "type": "clean_accuracy",
            "title": "Clean accuracy comparison",
            "description": (
                f"Mean clean accuracy across SNR levels: baseline "
                f"{clean_summary['baseline_clean_accuracy'] * 100:.2f}%, "
                f"robust {clean_summary['robust_clean_accuracy'] * 100:.2f}%."
            ),
        })

        return findings

    def dashboard_performance(self) -> list[dict[str, Any]]:
        return self.evaluation_accuracy_by_snr()

    def dashboard_attacks(self) -> list[dict[str, Any]]:
        _, _, attacks, _ = self._load()
        rows = []

        for attack in sorted(attacks["attack"].unique()):
            baseline = attacks[
                (attacks["model"] == "baseline") &
                (attacks["attack"] == attack)
            ]
            robust = attacks[
                (attacks["model"] == "robust") &
                (attacks["attack"] == attack)
            ]

            for _, row in baseline.iterrows():
                snr_value = int(row["snr_db"])
                robust_row = robust[robust["snr_db"] == snr_value]

                rows.append({
                    "attack": str(attack),
                    "snr": snr_value,
                    "baseline_asr": float(row["attack_success_rate"]),
                    "robust_asr": (
                        float(robust_row.iloc[0]["attack_success_rate"])
                        if not robust_row.empty else None
                    ),
                })

        return rows

    def dashboard_generalization_gap(self) -> list[dict[str, Any]]:
        _, _, _, gaps = self._load()
        return [
            {
                "attack": str(row["attack"]),
                "snr": int(row["snr_db"]),
                "gap": float(row["gap_robust_minus_baseline"]),
            }
            for _, row in gaps.iterrows()
        ]

    def dashboard_snr(self) -> list[dict[str, Any]]:
        _, clean, attacks, _ = self._load()
        rows = []

        for snr_value in sorted(clean["snr_db"].unique()):
            baseline = clean[
                (clean["model"] == "baseline") &
                (clean["snr_db"] == snr_value)
            ]
            robust = clean[
                (clean["model"] == "robust") &
                (clean["snr_db"] == snr_value)
            ]

            row: dict[str, Any] = {
                "snr": int(snr_value),
                "baseline_accuracy": (
                    float(baseline.iloc[0]["accuracy"]) if not baseline.empty else None
                ),
                "robust_accuracy": (
                    float(robust.iloc[0]["accuracy"]) if not robust.empty else None
                ),
            }

            for attack_name in ["fgsm", "pgd", "mim", "cw", "blackbox"]:
                for model_name, prefix in [
                    ("baseline", "baseline"),
                    ("robust", "robust"),
                ]:
                    match = attacks[
                        (attacks["model"] == model_name) &
                        (attacks["attack"] == attack_name) &
                        (attacks["snr_db"] == snr_value)
                    ]
                    row[f"{prefix}_{attack_name}_asr"] = (
                        float(match.iloc[0]["attack_success_rate"])
                        if not match.empty else None
                    )

                robust_match = attacks[
                    (attacks["model"] == "robust") &
                    (attacks["attack"] == attack_name) &
                    (attacks["snr_db"] == snr_value)
                ]
                row[f"{attack_name}_asr"] = (
                    float(robust_match.iloc[0]["attack_success_rate"])
                    if not robust_match.empty else None
                )

            rows.append(row)

        return rows

    def dashboard_summary(self) -> dict[str, Any]:
        metadata, clean, attacks, _ = self._load()
        snrs = sorted(int(value) for value in clean["snr_db"].unique())

        return {
            "dataset": {
                "name": metadata.get("dataset"),
                "num_classes": int(metadata.get("num_classes", 0)),
                "snr_min": min(snrs) if snrs else None,
                "snr_max": max(snrs) if snrs else None,
            },
            "evaluation": {
                "test_samples": int(metadata.get("total_samples", 0)),
                "snr_levels": len(snrs),
                "attack_methods": sorted(
                    str(value) for value in attacks["attack"].unique()
                ),
            },
            "attacks": sorted(
                str(value) for value in attacks["attack"].unique()
            ),
        }

    def dashboard_recent_updates(self) -> list[dict[str, Any]]:
        # Evaluation metadata does not contain dated activity records.
        return []
