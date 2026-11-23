
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


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

    # ------------------------------------------------------------------
    # Shared helpers for the visualization / reports / attacks pages
    # ------------------------------------------------------------------

    CONFIG_RELATIVE_PATH = Path("configs") / "config.yaml"

    ATTACK_METHODS = (
        {
            "name": "FGSM",
            "key": "fgsm",
            "type": "white-box",
            "description": "Single-step fast gradient sign method.",
        },
        {
            "name": "PGD",
            "key": "pgd",
            "type": "white-box",
            "description": "Projected gradient descent, iterative L-infinity attack.",
        },
        {
            "name": "MIM",
            "key": "mim",
            "type": "white-box",
            "description": "Momentum iterative method with gradient momentum.",
        },
        {
            "name": "C&W",
            "key": "cw",
            "type": "white-box",
            "description": "Carlini-Wagner optimisation attack on a fixed subset per SNR.",
        },
        {
            "name": "Black-box transfer",
            "key": "blackbox",
            "type": "black-box",
            "description": "Surrogate-model transfer attack against the baseline model.",
        },
    )

    def _config(self) -> dict[str, Any]:
        """Read configs/config.yaml (read-only). Returns {} when absent."""
        path = self.results_dir.parent / self.CONFIG_RELATIVE_PATH
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
        except (OSError, yaml.YAMLError):
            return {}
        return data if isinstance(data, dict) else {}

    def _try(self, builder, default):
        """Return builder() or default when result artifacts are unavailable."""
        try:
            return builder()
        except (FileNotFoundError, ValueError):
            return default

    def visualization_summary(self) -> dict[str, Any]:
        frame = self._load()
        summary = self.evaluation_summary()
        return {
            "data_points": int(len(frame) * len(self.ATTACKS)),
            "snr_levels": len(summary["snr_levels"]),
            "snr_min": summary["snr_min"],
            "snr_max": summary["snr_max"],
            "mean_robust_asr": summary["mean_robust_asr"],
            "mean_baseline_asr": summary["mean_baseline_asr"],
            "source": summary["source"],
        }

    def visualization_filters(self) -> dict[str, Any]:
        filters = self.evaluation_filters()
        if "snr_levels" not in filters and "snr" in filters:
            filters["snr_levels"] = filters["snr"]
        return filters

    def visualization_class_accuracy(self) -> list[dict[str, Any]]:
        """Per-class accuracy only when a results file really provides it."""
        for name in ("class_accuracy.csv", "per_class_accuracy.csv"):
            path = self.results_dir / name
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            label = next(
                (c for c in ("class_name", "class", "label") if c in frame.columns),
                None,
            )
            if label is None or "accuracy" not in frame.columns:
                continue
            rows = frame[[label, "accuracy"]].rename(columns={label: "class_name"})
            return self._records(rows)
        return []

    def visualization_model_comparison(self) -> list[dict[str, Any]]:
        summary = self.evaluation_summary()
        return [
            {"model": "baseline", "accuracy": summary["baseline_clean_accuracy"]},
            {"model": "robust", "accuracy": summary["robust_clean_accuracy"]},
        ]

    # ------------------------------ reports ---------------------------

    def reports_summary(self) -> dict[str, Any]:
        summary = self.evaluation_summary()
        generated = self.results_dir.parent / "reports" / "generated"
        generated_count = (
            sum(1 for p in generated.glob("*") if p.is_file())
            if generated.is_dir() else 0
        )
        return {
            "available_reports": len(self.list_files()) + generated_count,
            "report_count": len(self.list_files()) + generated_count,
            "robust_clean_accuracy": summary["robust_clean_accuracy"],
            "baseline_clean_accuracy": summary["baseline_clean_accuracy"],
            "mean_robust_asr": summary["mean_robust_asr"],
            "mean_baseline_asr": summary["mean_baseline_asr"],
            "mean_gap": summary["mean_gap"],
            "source": summary["source"],
        }

    def reports_executive_summary(self) -> dict[str, Any]:
        summary = self.evaluation_summary()
        robust = summary["mean_robust_asr"]
        baseline = summary["mean_baseline_asr"]
        gap = summary["mean_gap"]
        lead = None
        if robust is not None and baseline is not None and gap is not None:
            lead = (
                f"Across {len(summary['snr_levels'])} SNR levels and "
                f"{len(summary['attack_scope'])} attack methods, mean attack "
                f"success rate is {baseline:.2f}% on the baseline model and "
                f"{robust:.2f}% on the robust model, a gap of {gap:.2f} "
                f"percentage points."
            )
        highlights = self.evaluation_insights()
        return {"lead": lead, "highlights": highlights}

    def reports_metrics(self) -> list[dict[str, Any]]:
        summary = self.evaluation_summary()
        rows = []

        def add(name, baseline, robust):
            delta = (
                robust - baseline
                if baseline is not None and robust is not None else None
            )
            rows.append({
                "metric": name,
                "baseline": baseline,
                "robust": robust,
                "delta": delta,
            })

        add(
            "Clean accuracy",
            summary["baseline_clean_accuracy"],
            summary["robust_clean_accuracy"],
        )
        for row in self.evaluation_comparison():
            add(
                f"{row['attack'].upper()} ASR",
                row["baseline_asr"],
                row["robust_asr"],
            )
        return rows

    def reports_methodology(self) -> list[dict[str, Any]]:
        """Whitelisted, read-only view of configs/config.yaml."""
        config = self._config()
        dataset = config.get("dataset") or {}
        epsilon = config.get("epsilon")
        items = [
            ("Seed", config.get("seed")),
            ("Dataset", dataset.get("name")),
            ("SNR range (dB)", dataset.get("snr_range")),
            ("Epsilons", epsilon),
            ("PGD steps", config.get("pgd_steps")),
            ("MIM steps", config.get("mim_steps")),
            ("Adversarial training ratio", config.get("adv_train_ratio")),
        ]
        return [
            {"key": label, "label": label, "value": value}
            for label, value in items
            if value is not None
        ]

    REQUIRED_RESULT_FILES = (
        "master_results.csv",
        "clean_results.csv",
        "generalization_gap.csv",
        "metrics.json",
    )

    def reports_checks(self) -> list[dict[str, Any]]:
        checks = []
        for name in self.REQUIRED_RESULT_FILES:
            present = (self.results_dir / name).is_file()
            checks.append({
                "name": name,
                "status": "ok" if present else "warning",
                "message": "Present" if present else "Missing from results/",
            })
        for attack in self.ATTACKS:
            for prefix in ("", "baseline_"):
                name = f"{prefix}{attack}_results.csv"
                present = (self.results_dir / name).is_file()
                checks.append({
                    "name": name,
                    "status": "ok" if present else "warning",
                    "message": "Present" if present else "Missing from results/",
                })
        return checks

    REPORT_TYPES = ("executive", "metrics", "full")
    REPORT_SCOPES = ("all", "baseline", "robust")

    def reports_options(self) -> dict[str, Any]:
        return {
            "types": [
                {"value": "executive", "label": "Executive summary"},
                {"value": "metrics", "label": "Metrics comparison"},
                {"value": "full", "label": "Full evaluation report"},
            ],
            "scopes": [
                {"value": "all", "label": "All models"},
                {"value": "baseline", "label": "Baseline model"},
                {"value": "robust", "label": "Robust model"},
            ],
            "versions": ["latest"],
        }

    def reports_export(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": self.reports_summary(),
            "executive_summary": self.reports_executive_summary(),
            "metrics": self.reports_metrics(),
            "methodology": self.reports_methodology(),
            "checks": self.reports_checks(),
            "findings": self.evaluation_insights(),
        }

    def reports_build(self, report_type: str, scope: str) -> dict[str, Any]:
        report = self.reports_export()
        report["report_type"] = report_type
        report["scope"] = scope
        if report_type == "executive":
            keep = ("generated_at", "report_type", "scope", "executive_summary")
        elif report_type == "metrics":
            keep = ("generated_at", "report_type", "scope", "metrics")
        else:
            return self._scope_report(report, scope)
        return self._scope_report({k: report[k] for k in keep}, scope)

    @staticmethod
    def _scope_report(report: dict[str, Any], scope: str) -> dict[str, Any]:
        if scope in ("baseline", "robust") and "metrics" in report:
            other = "robust" if scope == "baseline" else "baseline"
            report["metrics"] = [
                {k: v for k, v in row.items() if k not in (other, "delta")}
                for row in report["metrics"]
            ]
        return report

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "", value)[:40] or "report"

    def reports_generate(
        self,
        report_type: str,
        scope: str,
        fmt: str,
        version: str,
    ) -> dict[str, Any]:
        report = self.reports_build(report_type, scope)
        out_dir = (self.results_dir.parent / "reports" / "generated").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        job_id = f"{self._safe_name(report_type)}-{self._safe_name(scope)}-{stamp}"
        extension = "md" if fmt == "markdown" else "json"
        target = (out_dir / f"{job_id}.{extension}").resolve()

        if out_dir not in target.parents:
            raise ValueError("Invalid report path")

        if extension == "json":
            text = json.dumps(report, indent=2, default=str)
        else:
            text = self._report_markdown(report)

        target.write_text(text, encoding="utf-8")
        return {
            "status": "completed",
            "job_id": job_id,
            "message": f"Report saved as reports/generated/{target.name}",
        }

    @staticmethod
    def _report_markdown(report: dict[str, Any]) -> str:
        lines = [
            "# AMCShield evaluation report",
            "",
            f"Generated: {report.get('generated_at')}",
            f"Type: {report.get('report_type')} | Scope: {report.get('scope')}",
            "",
        ]
        executive = report.get("executive_summary") or {}
        if executive.get("lead"):
            lines += ["## Executive summary", "", executive["lead"], ""]
        for item in executive.get("highlights") or []:
            lines.append(f"- **{item['title']}**: {item['description']}")
        if executive.get("highlights"):
            lines.append("")
        metrics = report.get("metrics") or []
        if metrics:
            lines += ["## Metrics", ""]
            for row in metrics:
                values = ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in row.items() if k != "metric"
                )
                lines.append(f"- {row['metric']}: {values}")
            lines.append("")
        return "\n".join(lines)

    def reports_preview(self, report_type: str, scope: str) -> dict[str, Any]:
        report = self.reports_build(report_type, scope)
        executive = report.get("executive_summary") or {}
        lead = executive.get("lead")
        return {
            "message": lead or "Preview ready; see summary for the report content.",
            "summary": {
                "report_type": report_type,
                "scope": scope,
                "metric_rows": len(report.get("metrics") or []),
                "finding_count": len(executive.get("highlights") or []),
            },
        }

    # ------------------------------ attacks ---------------------------

    def attacks_overview(self) -> dict[str, Any]:
        config = self._config()
        epsilon = (config.get("epsilon") or {}).get("pgd")
        frame = self._load()
        summary = self.evaluation_summary()
        robust_asr = summary["mean_robust_asr"]
        accuracies = [
            self._mean(frame, f"{attack}_accuracy") for attack in self.ATTACKS
        ]
        accuracies = [v for v in accuracies if v is not None]
        return {
            "attack": {
                "name": "Precomputed evaluation",
                "status": "idle",
                "message": (
                    "Attacks are precomputed offline; the dashboard shows "
                    "stored results only."
                ),
                "mean_asr": robust_asr,
                "attack_accuracy": (
                    sum(accuracies) / len(accuracies) if accuracies else None
                ),
                "clean_accuracy": summary["robust_clean_accuracy"],
                "epsilon": epsilon,
                "norm": "L\u221e",
                "query_budget": config.get("surrogate_query_budget"),
                "progress_percent": 0,
                "phase": "No active attack",
                "eta": None,
                "processed": None,
                "remaining": None,
                "current_snr": None,
                "queries": None,
                "target_model": "robust",
            }
        }

    def attacks_results(self) -> list[dict[str, Any]]:
        frame = self._load()
        rows = []
        for attack in self.ATTACKS:
            asr = self._mean(frame, f"{attack}_asr")
            rows.append({
                "attack": attack,
                "clean_accuracy": self._mean(frame, "clean_accuracy"),
                "attack_accuracy": self._mean(frame, f"{attack}_accuracy"),
                "attack_success_rate": asr,
                "mean_linf": self._mean(frame, f"{attack}_mean_linf"),
                "mean_l2": self._mean(frame, f"{attack}_mean_l2"),
                "status": "EVALUATED",
            })
        return rows

    def attacks_config(self) -> dict[str, Any]:
        config = self._config()
        epsilon = config.get("epsilon") or {}
        return {
            "models": ["baseline", "robust"],
            "methods": [
                {"name": m["key"], "label": m["name"]}
                for m in self.ATTACK_METHODS
            ],
            "epsilon": epsilon.get("pgd"),
            "epsilons": epsilon,
            "steps": config.get("pgd_steps"),
            "pgd_steps": config.get("pgd_steps"),
            "mim_steps": config.get("mim_steps"),
            "cw_eval_subset_size": config.get("cw_eval_subset_size"),
            "samples_per_snr": config.get("cw_eval_subset_size"),
            "snr_levels": self._try(
                lambda: self.evaluation_filters()["snr_levels"], []
            ),
            "source": "configs/config.yaml",
        }

    def attacks_snr_profile(self) -> list[dict[str, Any]]:
        frame = self._load()
        rows = []
        for _, row in frame.iterrows():
            values = [
                self._number(row[f"{attack}_asr"]) for attack in self.ATTACKS
            ]
            values = [v for v in values if v is not None]
            rows.append({
                "snr": int(row["snr"]),
                "attack_success_rate": (
                    sum(values) / len(values) if values else None
                ),
            })
        return rows

    def attacks_methods(self) -> list[dict[str, Any]]:
        return [
            {
                "name": m["name"],
                "type": m["type"],
                "description": m["description"],
            }
            for m in self.ATTACK_METHODS
        ]

    def attacks_constraints(self) -> list[dict[str, Any]]:
        config = self._config()
        epsilon = (config.get("epsilon") or {}).get("pgd")
        entries = [
            ("Norm", "L\u221e"),
            ("Epsilon", epsilon),
            ("PGD steps", config.get("pgd_steps")),
            ("MIM steps", config.get("mim_steps")),
            ("C&W subset per SNR", config.get("cw_eval_subset_size")),
            ("Surrogate query budget", config.get("surrogate_query_budget")),
        ]
        return [
            {"label": label, "value": value}
            for label, value in entries if value is not None
        ]

    # ------------------------------ project (About page) --------------

    def project_overview(self) -> dict[str, Any]:
        config = self._config()
        dataset = config.get("dataset") or {}
        snr_range = dataset.get("snr_range") or [None, None]
        return {
            "status": "available" if config else "unavailable",
            "num_classes": dataset.get("num_classes"),
            "class_count": dataset.get("num_classes"),
            "snr_min": snr_range[0] if len(snr_range) > 0 else None,
            "snr_max": snr_range[1] if len(snr_range) > 1 else None,
            "model_count": len(self.MODELS),
            "source": "configs/config.yaml" if config else None,
        }

    MODELS = (
        ("baseline", "Baseline classifier trained on clean signals only."),
        (
            "robust",
            "Classifier adversarially trained with PGD on a mixed "
            "clean/adversarial batch.",
        ),
    )

    def project_mission(self) -> dict[str, Any]:
        config = self._config()
        dataset = config.get("dataset") or {}
        name = dataset.get("name")
        classes = dataset.get("num_classes")
        if not name:
            return {}
        return {
            "mission": (
                f"Evaluate the adversarial robustness of automatic modulation "
                f"classification on {name}"
                + (f" ({classes} modulation classes)." if classes else ".")
            ),
            "objective": (
                "Compare a baseline model with a PGD adversarially trained "
                "robust model across SNR levels."
            ),
            "scope": (
                "Attacks evaluated: "
                + ", ".join(m["name"] for m in self.ATTACK_METHODS)
                + ". Attacks are precomputed offline."
            ),
        }

    def project_facts(self) -> list[dict[str, Any]]:
        config = self._config()
        dataset = config.get("dataset") or {}
        items = [
            ("Dataset", dataset.get("name")),
            ("Modulation classes", dataset.get("num_classes")),
            ("SNR range (dB)", dataset.get("snr_range")),
            ("Signal length", dataset.get("signal_length")),
            ("Input channels", dataset.get("input_channels")),
            ("Train / val / test split", self._split_text(config)),
            ("Seed", config.get("seed")),
        ]
        return [
            {"label": label, "value": value}
            for label, value in items if value is not None
        ]

    @staticmethod
    def _split_text(config: dict[str, Any]) -> str | None:
        parts = [config.get(k) for k in ("train_size", "val_size", "test_size")]
        if any(part is None for part in parts):
            return None
        return " / ".join(f"{float(part) * 100:g}%" for part in parts)

    def project_models(self) -> list[dict[str, Any]]:
        config = self._config()
        ratio = config.get("adv_train_ratio")
        rows = []
        for name, description in self.MODELS:
            text = description
            if name == "robust" and ratio is not None:
                text = f"{description} Adversarial ratio: {ratio}."
            rows.append({"name": name, "description": text})
        return rows

    def project_attacks(self) -> list[dict[str, Any]]:
        epsilon = (self._config().get("epsilon") or {})
        rows = []
        for method in self.ATTACK_METHODS:
            row = {
                "name": method["name"],
                "type": method["type"],
                "description": method["description"],
            }
            if method["key"] in epsilon:
                row["epsilon"] = epsilon[method["key"]]
            rows.append(row)
        return rows

    def project_context(self) -> list[dict[str, Any]]:
        config = self._config()
        counts = self._try(self._clean_counts, None)
        snrs = self._try(lambda: self.evaluation_filters()["snr_levels"], [])
        items = [
            ("Evaluated SNR levels", len(snrs) or None),
            ("Clean test samples", counts["total"] if counts else None),
            ("C&W subset per SNR", config.get("cw_eval_subset_size")),
            ("Surrogate query budget", config.get("surrogate_query_budget")),
            ("Results source", "master_results.csv" if snrs else None),
        ]
        return [
            {"label": label, "value": value}
            for label, value in items if value is not None
        ]
