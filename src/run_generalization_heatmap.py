"""
AMCShield - Baseline vs Robust Generalization Gap Analysis

Canonical scale:
    All accuracy / ASR values are normalized to percent [0, 100]
    before the generalization gap is calculated.

Generalization Gap = Baseline ASR - Robust ASR
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

OUTPUT_CSV = RESULTS_DIR / "generalization_gap.csv"
OUTPUT_JSON = RESULTS_DIR / "generalization_gap.json"
OUTPUT_PLOT = PLOTS_DIR / "09_generalization_gap_heatmap.png"

EXPECTED_SNR = list(range(-20, 31, 2))

ATTACK_FILES = {
    "FGSM": ("baseline_fgsm_results.csv", "fgsm_results.csv", "fraction"),
    "PGD": ("baseline_pgd_results.csv", "pgd_results.csv", "fraction"),
    "MIM": ("baseline_mim_results.csv", "mim_results.csv", "fraction"),
    "C&W": ("baseline_cw_results.csv", "cw_results.csv", "fraction"),
    "Black-box": ("baseline_blackbox_results.csv", "blackbox_results.csv", "percent"),
}


def load_and_normalize(filename, scale, label):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required result file does not exist: {path}")

    df = pd.read_csv(path)

    required = {"snr", "attack_success_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing required columns: {sorted(missing)}")

    if len(df) != 26 or sorted(df["snr"].astype(int).tolist()) != EXPECTED_SNR:
        raise ValueError(
            f"{label}: expected exactly 26 rows containing SNR values "
            f"{EXPECTED_SNR}"
        )

    asr = pd.to_numeric(df["attack_success_rate"], errors="raise")

    if scale == "fraction":
        asr = asr * 100.0
    elif scale != "percent":
        raise ValueError(f"{label}: unknown source scale '{scale}'")

    if ((asr < 0) | (asr > 100)).any():
        bad = asr[(asr < 0) | (asr > 100)].iloc[0]
        raise ValueError(
            f"{label}: normalized attack_success_rate contains {bad}, "
            "outside [0, 100]"
        )

    return pd.DataFrame({
        "snr": df["snr"].astype(int),
        "asr": asr.astype(float),
    }).sort_values("snr").reset_index(drop=True)


def build_generalization_gap():
    rows = []

    for attack, (baseline_file, robust_file, robust_scale) in ATTACK_FILES.items():
        baseline = load_and_normalize(
            baseline_file, "percent", f"{attack} baseline"
        )
        robust = load_and_normalize(
            robust_file, robust_scale, f"{attack} robust"
        )

        merged = baseline.merge(
            robust,
            on="snr",
            how="inner",
            validate="one_to_one",
            suffixes=("_baseline", "_robust"),
        )

        if len(merged) != 26:
            raise RuntimeError(
                f"{attack}: expected 26 merged rows, got {len(merged)}"
            )

        merged["attack"] = attack
        merged["baseline_asr"] = merged["asr_baseline"]
        merged["robust_asr"] = merged["asr_robust"]
        merged["gap"] = merged["baseline_asr"] - merged["robust_asr"]

        # Source-level arithmetic consistency check.
        expected_gap = merged["baseline_asr"] - merged["robust_asr"]
        if not (merged["gap"].sub(expected_gap).abs() <= 1e-10).all():
            raise AssertionError(f"{attack}: gap is not arithmetically consistent")

        rows.append(
            merged[["attack", "snr", "baseline_asr", "robust_asr", "gap"]]
        )

    result = pd.concat(rows, ignore_index=True)

    result["attack"] = pd.Categorical(
        result["attack"],
        categories=list(ATTACK_FILES.keys()),
        ordered=True,
    )
    result = result.sort_values(["attack", "snr"]).reset_index(drop=True)
    result["attack"] = result["attack"].astype(str)

    # Final canonical-scale validation.
    for column in ["baseline_asr", "robust_asr"]:
        if ((result[column] < 0) | (result[column] > 100)).any():
            raise AssertionError(
                f"Column {column} contains values outside [0, 100]"
            )

    recomputed = result["baseline_asr"] - result["robust_asr"]
    if not (result["gap"].sub(recomputed).abs() <= 1e-10).all():
        raise AssertionError("Final gap column does not match baseline_asr - robust_asr")

    result[["baseline_asr", "robust_asr", "gap"]] = result[
        ["baseline_asr", "robust_asr", "gap"]
    ].round(4)

    if len(result) != 130:
        raise RuntimeError(f"Expected 130 rows, generated {len(result)}")

    return result


def save_outputs(df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)

    summary = {}
    for attack in ATTACK_FILES:
        subset = df[df["attack"] == attack]
        summary[attack] = {
            "rows": int(len(subset)),
            "mean_baseline_asr": round(float(subset["baseline_asr"].mean()), 4),
            "mean_robust_asr": round(float(subset["robust_asr"].mean()), 4),
            "mean_gap": round(float(subset["gap"].mean()), 4),
            "min_gap": round(float(subset["gap"].min()), 4),
            "max_gap": round(float(subset["gap"].max()), 4),
        }

    output = {
        "description": "Baseline versus robust attack-success-rate generalization gap across SNR.",
        "formula": "generalization_gap = baseline_asr - robust_asr",
        "scale": "percent [0, 100]; gap in percentage points",
        "dataset": "RadioML2018.01A",
        "snr_values": EXPECTED_SNR,
        "num_attacks": 5,
        "num_snr_levels": 26,
        "total_rows": int(len(df)),
        "attacks": list(ATTACK_FILES.keys()),
        "attack_summary": summary,
        "data": df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return summary


def create_heatmap(df):
    heatmap_data = (
        df.pivot(index="attack", columns="snr", values="gap")
        .reindex(list(ATTACK_FILES.keys()))
        .reindex(columns=EXPECTED_SNR)
    )

    fig, ax = plt.subplots(figsize=(18, 6))
    image = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(EXPECTED_SNR)))
    ax.set_xticklabels(EXPECTED_SNR)
    ax.set_yticks(range(len(ATTACK_FILES)))
    ax.set_yticklabels(list(ATTACK_FILES.keys()))

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Attack")
    ax.set_title(
        "Baseline vs Robust Generalization Gap\n"
        "(Baseline ASR − Robust ASR)"
    )

    for row in range(heatmap_data.shape[0]):
        for col in range(heatmap_data.shape[1]):
            ax.text(
                col,
                row,
                f"{heatmap_data.iloc[row, col]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("ASR Gap (percentage points)")

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("AMCShield - Corrected Generalization Gap Analysis")
    print("=" * 70)

    df = build_generalization_gap()
    summary = save_outputs(df)
    create_heatmap(df)

    print("\nNormalized source scales:")
    for attack, (_, _, robust_scale) in ATTACK_FILES.items():
        print(f"  {attack:10s} robust ASR source scale: {robust_scale}")

    print("\nSummary:")
    print(pd.DataFrame(summary).T.round(2).to_string())

    print("\nArithmetic validation: PASSED")
    print(f"Rows: {len(df)}")
    print(f"SNR levels: {df['snr'].nunique()}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Output JSON: {OUTPUT_JSON}")
    print(f"Output heatmap: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()