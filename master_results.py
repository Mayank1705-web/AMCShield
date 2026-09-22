"""
AMCShield - Master Results & Plot Generator

Combines:
    - Clean
    - FGSM
    - PGD
    - MIM
    - C&W

into:
    results/master_results.csv
    results/master_results.json

and generates:
    results/plots/
        01_clean_accuracy_vs_snr.png
        02_adversarial_accuracy_vs_snr.png
        03_cw_accuracy_vs_snr.png
        04_attack_success_rate_vs_snr.png
        05_cw_mean_linf_vs_snr.png
        06_cw_max_linf_vs_snr.png
        07_cw_mean_l2_vs_snr.png
        08_cw_max_l2_vs_snr.png
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Change these paths if your files have different names.
# ------------------------------------------------------------

INPUT_FILES = {
    "clean": RESULTS_DIR / "clean_results.csv",
    "fgsm": RESULTS_DIR / "fgsm_results.csv",
    "pgd": RESULTS_DIR / "pgd_results.csv",
    "mim": RESULTS_DIR / "mim_results.csv",
    "cw": RESULTS_DIR / "cw_results.csv",
}


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(name):
    """
    Convert column names to a standard form.

    Examples:
        "SNR" -> "snr"
        "Clean Accuracy" -> "clean_accuracy"
        "Attack Success Rate" -> "attack_success_rate"
    """

    name = str(name).strip().lower()

    name = name.replace("∞", "inf")
    name = name.replace("∞", "inf")

    name = re.sub(r"[^a-z0-9]+", "_", name)

    name = re.sub(r"_+", "_", name)

    return name.strip("_")


def normalize_dataframe_columns(df):
    df = df.copy()

    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    return df


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, candidates, required=True):
    """
    Find the first matching column from a list of candidates.
    """

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:

        candidate = normalize_column_name(candidate)

        if candidate in normalized:
            return normalized[candidate]

    if required:
        raise ValueError(
            f"Could not find any of these columns:\n"
            f"{candidates}\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    return None


# ============================================================
# PERCENTAGE CONVERSION
# ============================================================

def percentage_to_fraction(series):
    """
    Convert percentage values to fractions.

    Handles both:
        55.18  -> 0.5518
        0.5518 -> 0.5518
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = series.dropna()

    if len(valid) == 0:
        return series

    # If values look like percentages, convert them.
    if valid.abs().max() > 1.0:
        return series / 100.0

    return series


def fraction_to_percentage(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ) * 100.0


# ============================================================
# GENERIC RESULT LOADER
# ============================================================

def load_result_file(path, attack_name):
    """
    Load one attack result CSV.

    Expected minimum:
        SNR

    Accuracy:
        clean_accuracy / accuracy / clean

    ASR:
        attack_success_rate / asr

    C&W distortion:
        mean_linf
        max_linf
        mean_l2
        max_l2
    """

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing result file for {attack_name.upper()}:\n"
            f"{path}\n\n"
            f"Create/export this file first."
        )

    print(
        f"Loading {attack_name.upper():<6}: "
        f"{path}"
    )

    df = pd.read_csv(path)

    df = normalize_dataframe_columns(df)

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    snr_col = find_column(
        df,
        [
            "snr",
            "snr_db",
            "snr_value",
        ]
    )

    df["snr"] = pd.to_numeric(
        df[snr_col],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy_col = find_column(
        df,
        [
            "accuracy",
            "clean_accuracy",
            "attack_accuracy",
            "cw_accuracy",
            "adversarial_accuracy",
            "adv_accuracy",
        ],
        required=False
    )

    if accuracy_col is not None:

        df[f"{attack_name}_accuracy"] = (
            percentage_to_fraction(
                df[accuracy_col]
            )
        )

    # --------------------------------------------------------
    # ASR
    # --------------------------------------------------------

    asr_col = find_column(
        df,
        [
            "attack_success_rate",
            "asr",
            "attack_success",
        ],
        required=False
    )

    if asr_col is not None:

        df[f"{attack_name}_asr"] = (
            percentage_to_fraction(
                df[asr_col]
            )
        )

    # --------------------------------------------------------
    # C&W distortion
    # --------------------------------------------------------

    if attack_name == "cw":

        distortion_columns = {
            "cw_mean_linf": [
                "mean_linf",
                "mean_l_inf",
                "mean_linf_distortion",
            ],
            "cw_max_linf": [
                "max_linf",
                "max_l_inf",
                "max_linf_distortion",
            ],
            "cw_mean_l2": [
                "mean_l2",
                "mean_l_2",
                "mean_l2_distortion",
            ],
            "cw_max_l2": [
                "max_l2",
                "max_l_2",
                "max_l2_distortion",
            ],
        }

        for output_name, candidates in distortion_columns.items():

            column = find_column(
                df,
                candidates,
                required=False
            )

            if column is not None:

                df[output_name] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

    # --------------------------------------------------------
    # Keep relevant columns
    # --------------------------------------------------------

    columns = ["snr"]

    if f"{attack_name}_accuracy" in df:
        columns.append(
            f"{attack_name}_accuracy"
        )

    if f"{attack_name}_asr" in df:
        columns.append(
            f"{attack_name}_asr"
        )

    for column in [
        "cw_mean_linf",
        "cw_max_linf",
        "cw_mean_l2",
        "cw_max_l2",
    ]:

        if column in df:
            columns.append(column)

    return df[columns].copy()


# ============================================================
# LOAD ALL RESULTS
# ============================================================

def load_all_results():

    dataframes = {}

    for attack_name, path in INPUT_FILES.items():

        dataframes[attack_name] = load_result_file(
            path,
            attack_name
        )

    return dataframes


# ============================================================
# MERGE RESULTS
# ============================================================

def merge_results(dataframes):

    print("\nMerging results...")

    master = None

    for attack_name in [
        "clean",
        "fgsm",
        "pgd",
        "mim",
        "cw",
    ]:

        df = dataframes[attack_name]

        # Remove duplicate SNR rows if any.
        df = (
            df
            .groupby("snr", as_index=False)
            .mean(numeric_only=True)
        )

        if master is None:

            master = df

        else:

            master = master.merge(
                df,
                on="snr",
                how="outer"
            )

    master = master.sort_values(
        "snr"
    ).reset_index(drop=True)

    return master


# ============================================================
# ADD DERIVED METRICS
# ============================================================

def add_derived_metrics(master):

    # --------------------------------------------------------
    # Accuracy in percentage
    # --------------------------------------------------------

    for attack in [
        "clean",
        "fgsm",
        "pgd",
        "mim",
        "cw",
    ]:

        column = f"{attack}_accuracy"

        if column in master:

            master[
                f"{attack}_accuracy_pct"
            ] = fraction_to_percentage(
                master[column]
            )

    # --------------------------------------------------------
    # ASR in percentage
    # --------------------------------------------------------

    for attack in [
        "fgsm",
        "pgd",
        "mim",
        "cw",
    ]:

        column = f"{attack}_asr"

        if column in master:

            master[
                f"{attack}_asr_pct"
            ] = fraction_to_percentage(
                master[column]
            )

    return master


# ============================================================
# SAVE MASTER CSV
# ============================================================

def save_master_csv(master):

    path = RESULTS_DIR / "master_results.csv"

    master.to_csv(
        path,
        index=False,
        float_format="%.8f"
    )

    print(
        f"\n✓ Master CSV saved:\n"
        f"  {path}"
    )


# ============================================================
# SAVE MASTER JSON
# ============================================================

def save_master_json(master):

    path = RESULTS_DIR / "master_results.json"

    # Replace NaN with None for valid JSON.
    json_df = master.replace(
        {np.nan: None}
    )

    records = json_df.to_dict(
        orient="records"
    )

    output = {
        "project": "AMCShield",
        "description": (
            "Master clean and adversarial robustness "
            "results by SNR"
        ),
        "attacks": [
            "clean",
            "fgsm",
            "pgd",
            "mim",
            "cw",
        ],
        "results": records,
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            allow_nan=False
        )

    print(
        f"✓ Master JSON saved:\n"
        f"  {path}"
    )


# ============================================================
# PLOT HELPER
# ============================================================

def save_plot(filename):

    path = PLOTS_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ {path}"
    )


def setup_plot(title, xlabel, ylabel):

    plt.figure(
        figsize=(11, 6)
    )

    plt.title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        xlabel,
        fontsize=11
    )

    plt.ylabel(
        ylabel,
        fontsize=11
    )

    plt.grid(
        True,
        alpha=0.25
    )


# ============================================================
# PLOT 1
# Clean Accuracy vs SNR
# ============================================================

def plot_clean_accuracy(master):

    if "clean_accuracy_pct" not in master:
        print(
            "⚠ Skipping clean accuracy plot."
        )
        return

    setup_plot(
        "AMCShield - Clean Accuracy vs SNR",
        "SNR (dB)",
        "Accuracy (%)"
    )

    plt.plot(
        master["snr"],
        master["clean_accuracy_pct"],
        marker="o",
        linewidth=2,
        label="Clean"
    )

    plt.legend()

    save_plot(
        "01_clean_accuracy_vs_snr.png"
    )


# ============================================================
# PLOT 2
# Adversarial Accuracy vs SNR
# ============================================================

def plot_adversarial_accuracy(master):

    setup_plot(
        "AMCShield - Adversarial Accuracy vs SNR",
        "SNR (dB)",
        "Accuracy (%)"
    )

    plotted = False

    for attack in [
        "fgsm",
        "pgd",
        "mim",
    ]:

        column = f"{attack}_accuracy_pct"

        if column not in master:
            continue

        plt.plot(
            master["snr"],
            master[column],
            marker="o",
            linewidth=2,
            label=attack.upper()
        )

        plotted = True

    if not plotted:

        plt.close()

        print(
            "⚠ No FGSM/PGD/MIM accuracy columns found."
        )

        return

    plt.legend()

    save_plot(
        "02_adversarial_accuracy_vs_snr.png"
    )


# ============================================================
# PLOT 3
# C&W Accuracy vs SNR
# ============================================================

def plot_cw_accuracy(master):

    column = "cw_accuracy_pct"

    if column not in master:

        print(
            "⚠ Skipping C&W accuracy plot."
        )

        return

    setup_plot(
        "AMCShield - C&W Accuracy vs SNR",
        "SNR (dB)",
        "Accuracy (%)"
    )

    plt.plot(
        master["snr"],
        master[column],
        marker="o",
        linewidth=2,
        label="C&W"
    )

    plt.legend()

    save_plot(
        "03_cw_accuracy_vs_snr.png"
    )


# ============================================================
# PLOT 4
# Attack Success Rate vs SNR
# ============================================================

def plot_attack_success_rate(master):

    setup_plot(
        "AMCShield - Attack Success Rate vs SNR",
        "SNR (dB)",
        "Attack Success Rate (%)"
    )

    plotted = False

    for attack in [
        "fgsm",
        "pgd",
        "mim",
        "cw",
    ]:

        column = f"{attack}_asr_pct"

        if column not in master:
            continue

        plt.plot(
            master["snr"],
            master[column],
            marker="o",
            linewidth=2,
            label=attack.upper()
        )

        plotted = True

    if not plotted:

        plt.close()

        print(
            "⚠ No ASR columns found."
        )

        return

    plt.legend()

    save_plot(
        "04_attack_success_rate_vs_snr.png"
    )


# ============================================================
# PLOT 5
# C&W Mean L_inf
# ============================================================

def plot_cw_mean_linf(master):

    column = "cw_mean_linf"

    if column not in master:
        print(
            "⚠ Skipping C&W mean L_inf plot."
        )
        return

    setup_plot(
        "AMCShield - C&W Mean L∞ Distortion",
        "SNR (dB)",
        "Mean L∞ Distortion"
    )

    plt.plot(
        master["snr"],
        master[column],
        marker="o",
        linewidth=2,
        label="Mean L∞"
    )

    plt.legend()

    save_plot(
        "05_cw_mean_linf_vs_snr.png"
    )


# ============================================================
# PLOT 6
# C&W Max L_inf
# ============================================================

def plot_cw_max_linf(master):

    column = "cw_max_linf"

    if column not in master:
        print(
            "⚠ Skipping C&W max L_inf plot."
        )
        return

    setup_plot(
        "AMCShield - C&W Maximum L∞ Distortion",
        "SNR (dB)",
        "Maximum L∞ Distortion"
    )

    plt.plot(
        master["snr"],
        master[column],
        marker="o",
        linewidth=2,
        label="Max L∞"
    )

    plt.legend()

    save_plot(
        "06_cw_max_linf_vs_snr.png"
    )


# ============================================================
# PLOT 7
# C&W Mean L2
# ============================================================

def plot_cw_mean_l2(master):

    column = "cw_mean_l2"

    if column not in master:
        print(
            "⚠ Skipping C&W mean L2 plot."
        )
        return

    setup_plot(
        "AMCShield - C&W Mean L2 Distortion",
        "SNR (dB)",
        "Mean L2 Distortion"
    )

    plt.plot(
        master["snr"],
        master[column],
        marker="o",
        linewidth=2,
        label="Mean L2"
    )

    plt.legend()

    save_plot(
        "07_cw_mean_l2_vs_snr.png"
    )


# ============================================================
# PLOT 8
# C&W Max L2
# ============================================================

def plot_cw_max_l2(master):

    column = "cw_max_l2"

    if column not in master:
        print(
            "⚠ Skipping C&W max L2 plot."
        )
        return

    setup_plot(
        "AMCShield - C&W Maximum L2 Distortion",
        "SNR (dB)",
        "Maximum L2 Distortion"
    )

    plt.plot(
        master["snr"],
        master[column],
        marker="o",
        linewidth=2,
        label="Max L2"
    )

    plt.legend()

    save_plot(
        "08_cw_max_l2_vs_snr.png"
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def print_summary(master):

    print()
    print("=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    for attack in [
        "clean",
        "fgsm",
        "pgd",
        "mim",
        "cw",
    ]:

        accuracy_column = (
            f"{attack}_accuracy_pct"
        )

        asr_column = (
            f"{attack}_asr_pct"
        )

        print(
            f"\n{attack.upper()}"
        )

        if accuracy_column in master:

            print(
                f"  Mean accuracy : "
                f"{master[accuracy_column].mean():.2f}%"
            )

        if asr_column in master:

            print(
                f"  Mean ASR      : "
                f"{master[asr_column].mean():.2f}%"
            )

    # --------------------------------------------------------
    # C&W distortion
    # --------------------------------------------------------

    if "cw_mean_linf" in master:

        print(
            "\nC&W DISTORTION"
        )

        print(
            f"  Mean of mean L∞ : "
            f"{master['cw_mean_linf'].mean():.6f}"
        )

        print(
            f"  Maximum L∞      : "
            f"{master['cw_max_linf'].max():.6f}"
        )

    if "cw_mean_l2" in master:

        print(
            f"  Mean of mean L2  : "
            f"{master['cw_mean_l2'].mean():.6f}"
        )

        print(
            f"  Maximum L2       : "
            f"{master['cw_max_l2'].max():.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("AMCShield - Master Results Pipeline")
    print("=" * 100)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    dataframes = load_all_results()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    master = merge_results(
        dataframes
    )

    # --------------------------------------------------------
    # Derived metrics
    # --------------------------------------------------------

    master = add_derived_metrics(
        master
    )

    # --------------------------------------------------------
    # Display master table
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("MASTER RESULTS")
    print("=" * 100)

    print(
        master.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_master_csv(
        master
    )

    save_master_json(
        master
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("GENERATING PLOTS")
    print("=" * 100)

    plot_clean_accuracy(
        master
    )

    plot_adversarial_accuracy(
        master
    )

    plot_cw_accuracy(
        master
    )

    plot_attack_success_rate(
        master
    )

    plot_cw_mean_linf(
        master
    )

    plot_cw_max_linf(
        master
    )

    plot_cw_mean_l2(
        master
    )

    plot_cw_max_l2(
        master
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        master
    )

    print()
    print("=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)

    print(
        f"\nMaster CSV : "
        f"{RESULTS_DIR / 'master_results.csv'}"
    )

    print(
        f"Master JSON: "
        f"{RESULTS_DIR / 'master_results.json'}"
    )

    print(
        f"Plots      : "
        f"{PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()