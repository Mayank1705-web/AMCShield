"""
AMCShield - Baseline vs Robust Generalization Gap Analysis

Calculates:

    Generalization Gap = Baseline ASR - Robust ASR

for:
    FGSM
    PGD
    MIM
    C&W
    Black-box

across all 26 SNR levels.

This script ONLY reads existing result CSVs.
It does NOT retrain models.
It does NOT rerun attacks.

Outputs:
    results/generalization_gap.csv
    results/generalization_gap.json
    results/plots/09_generalization_gap_heatmap.png
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

OUTPUT_CSV = RESULTS_DIR / "generalization_gap.csv"
OUTPUT_JSON = RESULTS_DIR / "generalization_gap.json"
OUTPUT_PLOT = PLOTS_DIR / "09_generalization_gap_heatmap.png"


# ============================================================
# EXPECTED SNR VALUES
# ============================================================

EXPECTED_SNR = list(range(-20, 31, 2))


# ============================================================
# INPUT FILES
# ============================================================

ATTACK_FILES = {
    "FGSM": {
        "baseline": RESULTS_DIR / "baseline_fgsm_results.csv",
        "robust": RESULTS_DIR / "fgsm_results.csv",
    },

    "PGD": {
        "baseline": RESULTS_DIR / "baseline_pgd_results.csv",
        "robust": RESULTS_DIR / "pgd_results.csv",
    },

    "MIM": {
        "baseline": RESULTS_DIR / "baseline_mim_results.csv",
        "robust": RESULTS_DIR / "mim_results.csv",
    },

    "C&W": {
        "baseline": RESULTS_DIR / "baseline_cw_results.csv",
        "robust": RESULTS_DIR / "cw_results.csv",
    },

    "Black-box": {
        "baseline": RESULTS_DIR / "baseline_blackbox_results.csv",
        "robust": RESULTS_DIR / "blackbox_results.csv",
    },
}


# ============================================================
# VALIDATION
# ============================================================

def validate_file(path):
    """Validate that a result CSV contains all 26 SNR levels."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file does not exist:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "snr",
        "attack_success_rate",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if len(df) != 26:
        raise ValueError(
            f"{path.name} has {len(df)} rows. "
            f"Expected 26."
        )

    actual_snr = sorted(
        df["snr"].astype(int).tolist()
    )

    if actual_snr != EXPECTED_SNR:
        raise ValueError(
            f"{path.name} has unexpected SNR values.\n"
            f"Expected: {EXPECTED_SNR}\n"
            f"Actual:   {actual_snr}"
        )

    if df["snr"].nunique() != 26:
        raise ValueError(
            f"{path.name} contains duplicate SNR values."
        )

    return df


# ============================================================
# LOAD ASR DATA
# ============================================================

def load_asr(path):
    """
    Load SNR and attack success rate.

    Returns a normalized dataframe:
        snr
        asr
    """

    df = validate_file(path)

    result = df[
        [
            "snr",
            "attack_success_rate",
        ]
    ].copy()

    result["snr"] = result["snr"].astype(int)

    result["attack_success_rate"] = (
        pd.to_numeric(
            result["attack_success_rate"],
            errors="raise",
        )
    )

    result = result.sort_values(
        "snr"
    ).reset_index(drop=True)

    return result


# ============================================================
# BUILD GENERALIZATION GAP
# ============================================================

def build_generalization_gap():
    """Build the complete 5 × 26 generalization-gap table."""

    print("=" * 70)
    print("AMCShield - Generalization Gap Analysis")
    print("=" * 70)

    print()
    print("Reading existing result CSVs...")
    print("No attacks will be rerun.")
    print()

    all_rows = []

    for attack, files in ATTACK_FILES.items():

        print(f"[{attack}]")

        baseline_df = load_asr(
            files["baseline"]
        )

        robust_df = load_asr(
            files["robust"]
        )

        # ----------------------------------------------------
        # Rename ASR columns
        # ----------------------------------------------------

        baseline_df = baseline_df.rename(
            columns={
                "attack_success_rate":
                    "baseline_asr"
            }
        )

        robust_df = robust_df.rename(
            columns={
                "attack_success_rate":
                    "robust_asr"
            }
        )

        # ----------------------------------------------------
        # Merge by SNR
        # ----------------------------------------------------

        merged = pd.merge(
            baseline_df,
            robust_df,
            on="snr",
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != 26:
            raise ValueError(
                f"{attack}: expected 26 merged rows, "
                f"got {len(merged)}."
            )

        # ----------------------------------------------------
        # Generalization gap
        # ----------------------------------------------------

        merged["gap"] = (
            merged["baseline_asr"]
            - merged["robust_asr"]
        )

        merged["attack"] = attack

        # ----------------------------------------------------
        # Reorder columns
        # ----------------------------------------------------

        merged = merged[
            [
                "attack",
                "snr",
                "baseline_asr",
                "robust_asr",
                "gap",
            ]
        ]

        all_rows.append(
            merged
        )

        print(
            f"  ✓ Baseline : {len(baseline_df)} rows"
        )

        print(
            f"  ✓ Robust   : {len(robust_df)} rows"
        )

        print(
            f"  ✓ Gap rows : {len(merged)}"
        )

    # --------------------------------------------------------
    # Combine all attacks
    # --------------------------------------------------------

    final_df = pd.concat(
        all_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    attack_order = [
        "FGSM",
        "PGD",
        "MIM",
        "C&W",
        "Black-box",
    ]

    final_df["attack"] = pd.Categorical(
        final_df["attack"],
        categories=attack_order,
        ordered=True,
    )

    final_df = final_df.sort_values(
        [
            "attack",
            "snr",
        ]
    ).reset_index(drop=True)

    # Convert categorical back to string before saving.
    final_df["attack"] = (
        final_df["attack"].astype(str)
    )

    # --------------------------------------------------------
    # Round numerical values
    # --------------------------------------------------------

    final_df[
        [
            "baseline_asr",
            "robust_asr",
            "gap",
        ]
    ] = final_df[
        [
            "baseline_asr",
            "robust_asr",
            "gap",
        ]
    ].round(4)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_rows = 5 * 26

    if len(final_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"generated {len(final_df)}."
        )

    if final_df["attack"].nunique() != 5:
        raise RuntimeError(
            "Expected exactly 5 attacks."
        )

    if final_df["snr"].nunique() != 26:
        raise RuntimeError(
            "Expected exactly 26 SNR levels."
        )

    # Every attack must have all 26 SNR values.
    counts = (
        final_df
        .groupby("attack", observed=True)["snr"]
        .count()
    )

    if not (counts == 26).all():
        raise RuntimeError(
            "Not every attack contains 26 SNR rows."
        )

    return final_df


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(df):
    """Save generalization-gap CSV."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print(
        f"✓ Saved CSV: {OUTPUT_CSV}"
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(df):
    """Save JSON representation and summary metadata."""

    attacks = [
        "FGSM",
        "PGD",
        "MIM",
        "C&W",
        "Black-box",
    ]

    attack_summary = {}

    for attack in attacks:

        subset = df[
            df["attack"] == attack
        ]

        attack_summary[attack] = {
            "rows": int(len(subset)),
            "mean_baseline_asr": round(
                float(
                    subset["baseline_asr"].mean()
                ),
                4,
            ),
            "mean_robust_asr": round(
                float(
                    subset["robust_asr"].mean()
                ),
                4,
            ),
            "mean_gap": round(
                float(
                    subset["gap"].mean()
                ),
                4,
            ),
            "max_gap": round(
                float(
                    subset["gap"].max()
                ),
                4,
            ),
            "min_gap": round(
                float(
                    subset["gap"].min()
                ),
                4,
            ),
        }

    output = {
        "description": (
            "Baseline versus robust attack-success-rate "
            "generalization gap across SNR."
        ),

        "formula": (
            "generalization_gap = "
            "baseline_asr - robust_asr"
        ),

        "dataset": "RadioML2018.01A",

        "snr_values": EXPECTED_SNR,

        "num_attacks": 5,

        "num_snr_levels": 26,

        "total_rows": int(len(df)),

        "attacks": attacks,

        "attack_summary": attack_summary,

        "data": df.to_dict(
            orient="records"
        ),
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"✓ Saved JSON: {OUTPUT_JSON}"
    )


# ============================================================
# HEATMAP
# ============================================================

def create_heatmap(df):
    """Create and save the 5 × 26 generalization-gap heatmap."""

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    attack_order = [
        "FGSM",
        "PGD",
        "MIM",
        "C&W",
        "Black-box",
    ]

    # --------------------------------------------------------
    # Pivot:
    #
    # Rows    = attacks
    # Columns = SNR
    # Values  = gap
    # --------------------------------------------------------

    heatmap_data = (
        df.pivot(
            index="attack",
            columns="snr",
            values="gap",
        )
        .reindex(attack_order)
        .reindex(columns=EXPECTED_SNR)
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(18, 6)
    )

    image = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        interpolation="nearest",
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xticks(
        range(len(EXPECTED_SNR))
    )

    ax.set_xticklabels(
        EXPECTED_SNR
    )

    ax.set_yticks(
        range(len(attack_order))
    )

    ax.set_yticklabels(
        attack_order
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Attack"
    )

    ax.set_title(
        "Baseline vs Robust Generalization Gap\n"
        "(Baseline ASR − Robust ASR)"
    )

    # --------------------------------------------------------
    # Cell annotations
    # --------------------------------------------------------

    for row in range(
        heatmap_data.shape[0]
    ):

        for col in range(
            heatmap_data.shape[1]
        ):

            value = heatmap_data.iloc[
                row,
                col,
            ]

            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "ASR Gap (percentage points)"
    )

    # --------------------------------------------------------
    # Layout and save
    # --------------------------------------------------------

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"✓ Saved heatmap: {OUTPUT_PLOT}"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(df):
    """Print a concise summary of the generated analysis."""

    print()
    print("=" * 70)
    print("GENERALIZATION GAP SUMMARY")
    print("=" * 70)

    summary = (
        df.groupby(
            "attack",
            observed=True,
        )
        .agg(
            baseline_mean_asr=(
                "baseline_asr",
                "mean",
            ),
            robust_mean_asr=(
                "robust_asr",
                "mean",
            ),
            mean_gap=(
                "gap",
                "mean",
            ),
            min_gap=(
                "gap",
                "min",
            ),
            max_gap=(
                "gap",
                "max",
            ),
        )
    )

    print(
        summary.round(2).to_string()
    )

    print()
    print(
        f"Total rows : {len(df)}"
    )

    print(
        f"Attacks    : {df['attack'].nunique()}"
    )

    print(
        f"SNR levels : {df['snr'].nunique()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = build_generalization_gap()

    save_csv(df)

    save_json(df)

    create_heatmap(df)

    print_summary(df)

    print()
    print("=" * 70)
    print("SECTION 3 COMPLETED")
    print("=" * 70)

    print(
        "✓ 5 attacks"
    )

    print(
        "✓ 26 SNR levels"
    )

    print(
        "✓ 130 total gap records"
    )

    print(
        "✓ CSV generated"
    )

    print(
        "✓ JSON generated"
    )

    print(
        "✓ Heatmap generated"
    )


if __name__ == "__main__":
    main()