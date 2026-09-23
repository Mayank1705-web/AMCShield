from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# ------------------------------------------------------------
# Load helper
# ------------------------------------------------------------

def load(name):
    path = RESULTS / name

    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    return pd.read_csv(path)


def keep_without_clean(df, columns):
    """
    Select requested columns while explicitly excluding any
    duplicate clean_accuracy column from attack result files.
    """
    return df[columns].copy()


# ------------------------------------------------------------
# Robust results
# ------------------------------------------------------------

clean = load("clean_results.csv")[[
    "snr",
    "accuracy"
]].rename(
    columns={"accuracy": "clean_accuracy"}
)


fgsm = load("fgsm_results.csv")[[
    "snr",
    "fgsm_accuracy",
    "attack_success_rate"
]].rename(
    columns={"attack_success_rate": "fgsm_asr"}
)


pgd = load("pgd_results.csv")[[
    "snr",
    "pgd_accuracy",
    "attack_success_rate"
]].rename(
    columns={"attack_success_rate": "pgd_asr"}
)


mim = load("mim_results.csv")[[
    "snr",
    "mim_accuracy",
    "attack_success_rate"
]].rename(
    columns={"attack_success_rate": "mim_asr"}
)


cw = load("cw_results.csv")[[
    "snr",
    "cw_accuracy",
    "attack_success_rate",
    "mean_linf",
    "max_linf",
    "mean_l2",
    "max_l2",
]].rename(
    columns={
        "attack_success_rate": "cw_asr",
        "mean_linf": "cw_mean_linf",
        "max_linf": "cw_max_linf",
        "mean_l2": "cw_mean_l2",
        "max_l2": "cw_max_l2",
    }
)


blackbox = load("blackbox_results.csv")[[
    "snr",
    "blackbox_accuracy",
    "attack_success_rate",
    "mean_linf",
    "max_linf",
    "mean_l2",
    "max_l2",
]].rename(
    columns={
        "attack_success_rate": "blackbox_asr",
        "mean_linf": "blackbox_mean_linf",
        "max_linf": "blackbox_max_linf",
        "mean_l2": "blackbox_mean_l2",
        "max_l2": "blackbox_max_l2",
    }
)


# ------------------------------------------------------------
# Baseline results
# ------------------------------------------------------------

b_clean = load("baseline_clean_results.csv")[[
    "snr",
    "clean_accuracy"
]].rename(
    columns={"clean_accuracy": "baseline_clean_accuracy"}
)


b_fgsm = load("baseline_fgsm_results.csv")[[
    "snr",
    "attack_accuracy",
    "attack_success_rate"
]].rename(
    columns={
        "attack_accuracy": "baseline_fgsm_accuracy",
        "attack_success_rate": "baseline_fgsm_asr",
    }
)


b_pgd = load("baseline_pgd_results.csv")[[
    "snr",
    "attack_accuracy",
    "attack_success_rate"
]].rename(
    columns={
        "attack_accuracy": "baseline_pgd_accuracy",
        "attack_success_rate": "baseline_pgd_asr",
    }
)


b_mim = load("baseline_mim_results.csv")[[
    "snr",
    "attack_accuracy",
    "attack_success_rate"
]].rename(
    columns={
        "attack_accuracy": "baseline_mim_accuracy",
        "attack_success_rate": "baseline_mim_asr",
    }
)


b_cw = load("baseline_cw_results.csv")[[
    "snr",
    "cw_accuracy",
    "attack_success_rate",
    "mean_linf",
    "max_linf",
    "mean_l2",
    "max_l2",
]].rename(
    columns={
        "cw_accuracy": "baseline_cw_accuracy",
        "attack_success_rate": "baseline_cw_asr",
        "mean_linf": "baseline_cw_mean_linf",
        "max_linf": "baseline_cw_max_linf",
        "mean_l2": "baseline_cw_mean_l2",
        "max_l2": "baseline_cw_max_l2",
    }
)


b_black = load("baseline_blackbox_results.csv")[[
    "snr",
    "blackbox_accuracy",
    "attack_success_rate"
]].rename(
    columns={
        "blackbox_accuracy": "baseline_blackbox_accuracy",
        "attack_success_rate": "baseline_blackbox_asr",
    }
)


# ------------------------------------------------------------
# Generalization gap
# ------------------------------------------------------------

gap_raw = load("generalization_gap.csv")

gap = gap_raw.pivot(
    index="snr",
    columns="attack",
    values="gap",
).rename(
    columns={
        "FGSM": "fgsm_gap",
        "PGD": "pgd_gap",
        "MIM": "mim_gap",
        "C&W": "cw_gap",
        "Black-box": "blackbox_gap",
    }
).reset_index()


# ------------------------------------------------------------
# Merge everything
# ------------------------------------------------------------

master = clean.copy()

datasets = [
    fgsm,
    pgd,
    mim,
    cw,
    blackbox,

    b_clean,
    b_fgsm,
    b_pgd,
    b_mim,
    b_cw,
    b_black,

    gap,
]

for df in datasets:
    master = master.merge(
        df,
        on="snr",
        how="left"
    )


# ------------------------------------------------------------
# Percentage fields
# ------------------------------------------------------------

# Preserve original robust-model percentage fields
master["clean_accuracy_pct"] = (
    master["clean_accuracy"] * 100
)

for attack in [
    "fgsm",
    "pgd",
    "mim",
    "cw",
    "blackbox",
]:
    master[f"{attack}_accuracy_pct"] = (
        master[f"{attack}_accuracy"] * 100
    )

    master[f"{attack}_asr_pct"] = (
        master[f"{attack}_asr"] * 100
    )


# Baseline percentage fields
master["baseline_clean_accuracy_pct"] = (
    master["baseline_clean_accuracy"] * 100
)

for attack in [
    "fgsm",
    "pgd",
    "mim",
    "cw",
    "blackbox",
]:
    master[f"baseline_{attack}_accuracy_pct"] = (
        master[f"baseline_{attack}_accuracy"] * 100
    )

    master[f"baseline_{attack}_asr_pct"] = (
        master[f"baseline_{attack}_asr"] * 100
    )


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

master = master.sort_values("snr").reset_index(drop=True)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

required_columns = [
    "snr",

    "clean_accuracy",
    "fgsm_accuracy",
    "fgsm_asr",
    "pgd_accuracy",
    "pgd_asr",
    "mim_accuracy",
    "mim_asr",
    "cw_accuracy",
    "cw_asr",
    "blackbox_accuracy",
    "blackbox_asr",

    "baseline_clean_accuracy",
    "baseline_fgsm_accuracy",
    "baseline_fgsm_asr",
    "baseline_pgd_accuracy",
    "baseline_pgd_asr",
    "baseline_mim_accuracy",
    "baseline_mim_asr",
    "baseline_cw_accuracy",
    "baseline_cw_asr",
    "baseline_blackbox_accuracy",
    "baseline_blackbox_asr",

    "fgsm_gap",
    "pgd_gap",
    "mim_gap",
    "cw_gap",
    "blackbox_gap",
]

missing = [
    column for column in required_columns
    if column not in master.columns
]

if missing:
    raise RuntimeError(
        f"Missing required master columns: {missing}"
    )


if len(master) != 26:
    raise RuntimeError(
        f"Expected 26 SNR rows, found {len(master)}"
    )


if master["snr"].nunique() != 26:
    raise RuntimeError(
        "Expected 26 unique SNR values."
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

csv_path = RESULTS / "master_results.csv"
json_path = RESULTS / "master_results.json"

master.to_csv(
    csv_path,
    index=False
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        master.to_dict(orient="records"),
        f,
        indent=2
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("=" * 70)
print("AMCShield - Master Results Pipeline")
print("=" * 70)

print(f"Rows    : {len(master)}")
print(f"Columns : {len(master.columns)}")

print("\nRequired fields verified:")
print("  ✓ Robust clean/FGSM/PGD/MIM/CW")
print("  ✓ Robust black-box")
print("  ✓ Baseline clean/FGSM/PGD/MIM/CW")
print("  ✓ Baseline black-box")
print("  ✓ Generalization gaps")
print("  ✓ Percentage fields")

print("\nFiles:")
print(f"  CSV  : {csv_path}")
print(f"  JSON : {json_path}")

print("\nSNR range:")
print(
    f"  {master['snr'].min()} to "
    f"{master['snr'].max()} dB"
)

print("\nMaster results generated successfully.")
print("=" * 70)