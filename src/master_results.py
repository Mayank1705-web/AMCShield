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


def normalize_percent(df, columns, source_scale, label):
    """Normalize known accuracy/ASR columns to canonical percent scale [0, 100]."""
    out = df.copy()
    if source_scale == "fraction":
        for column in columns:
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="raise") * 100.0
    elif source_scale == "percent":
        for column in columns:
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="raise")
    else:
        raise ValueError(f"Unknown source scale for {label}: {source_scale}")

    for column in columns:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{label}: non-numeric/NaN values found in {column}")
        if ((values < 0) | (values > 100)).any():
            bad = values[(values < 0) | (values > 100)].iloc[0]
            raise ValueError(
                f"{label}: column '{column}' contains value {bad}, outside [0, 100]"
            )
    return out


def validate_percent_columns(df, columns, label):
    """Sanity-check canonical percent columns after all source normalization."""
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 100)).any():
            bad = values[values.isna() | (values < 0) | (values > 100)]
            example = bad.iloc[0] if len(bad) else "unknown"
            raise ValueError(
                f"{label}: normalized column '{column}' has invalid value {example}; "
                "expected [0, 100] percent scale."
            )


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
clean = normalize_percent(
    clean, ["clean_accuracy"], "fraction", "robust clean"
)


fgsm = load("fgsm_results.csv")[[
    "snr",
    "fgsm_accuracy",
    "attack_success_rate"
]].rename(
    columns={"attack_success_rate": "fgsm_asr"}
)
fgsm = normalize_percent(
    fgsm, ["fgsm_accuracy", "fgsm_asr"], "fraction", "robust FGSM"
)


pgd = load("pgd_results.csv")[[
    "snr",
    "pgd_accuracy",
    "attack_success_rate"
]].rename(
    columns={"attack_success_rate": "pgd_asr"}
)
pgd = normalize_percent(
    pgd, ["pgd_accuracy", "pgd_asr"], "fraction", "robust PGD"
)


mim = load("mim_results.csv")[[
    "snr",
    "mim_accuracy",
    "attack_success_rate"
]].rename(
    columns={"attack_success_rate": "mim_asr"}
)
mim = normalize_percent(
    mim, ["mim_accuracy", "mim_asr"], "fraction", "robust MIM"
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
cw = normalize_percent(
    cw, ["cw_accuracy", "cw_asr"], "fraction", "robust C&W"
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
blackbox = normalize_percent(
    blackbox, ["blackbox_accuracy", "blackbox_asr"], "percent", "robust Black-box"
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
b_clean = normalize_percent(
    b_clean, ["baseline_clean_accuracy"], "percent", "baseline clean"
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
b_fgsm = normalize_percent(
    b_fgsm, ["baseline_fgsm_accuracy", "baseline_fgsm_asr"], "percent", "baseline FGSM"
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
b_pgd = normalize_percent(
    b_pgd, ["baseline_pgd_accuracy", "baseline_pgd_asr"], "percent", "baseline PGD"
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
b_mim = normalize_percent(
    b_mim, ["baseline_mim_accuracy", "baseline_mim_asr"], "percent", "baseline MIM"
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
b_cw = normalize_percent(
    b_cw, ["baseline_cw_accuracy", "baseline_cw_asr"], "percent", "baseline C&W"
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
b_black = normalize_percent(
    b_black, ["baseline_blackbox_accuracy", "baseline_blackbox_asr"], "percent", "baseline Black-box"
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
# Canonical scale validation
# ------------------------------------------------------------

PERCENT_COLUMNS = [
    "clean_accuracy", "fgsm_accuracy", "fgsm_asr",
    "pgd_accuracy", "pgd_asr", "mim_accuracy", "mim_asr",
    "cw_accuracy", "cw_asr", "blackbox_accuracy", "blackbox_asr",
    "baseline_clean_accuracy",
    "baseline_fgsm_accuracy", "baseline_fgsm_asr",
    "baseline_pgd_accuracy", "baseline_pgd_asr",
    "baseline_mim_accuracy", "baseline_mim_asr",
    "baseline_cw_accuracy", "baseline_cw_asr",
    "baseline_blackbox_accuracy", "baseline_blackbox_asr",
]

validate_percent_columns(master, PERCENT_COLUMNS, "master")

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
print("  ✓ Canonical percent scale [0, 100] validated")

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