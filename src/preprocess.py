"""
preprocess.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 1 (Week 1-2)

Responsibility:
    Create a stratified train/validation/test split for
    RadioML 2018.01A.

Important:
    Stratification is performed jointly by:
        (modulation, SNR)

    The complete 21+ GB HDF5 signal dataset is NOT copied.
    Only sample indices are saved to data/processed/.
"""

from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "data/raw/GOLD_XYZ_OSC.0001_1024.hdf5"
OUTPUT_DIR = "data/processed"

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

SEED = 42


# ============================================================
# Normalization
# ============================================================

def normalize(X: np.ndarray) -> np.ndarray:
    """
    Per-sample unit-energy normalization.

    Input:
        X: (N, 2, 1024)

    Output:
        normalized X with the same shape.
    """

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X with shape (N, 2, 1024), got {X.shape}"
        )

    energy = np.sqrt(
        np.sum(X ** 2, axis=(1, 2), keepdims=True)
    )

    epsilon = 1e-12

    return X / np.maximum(energy, epsilon)


# ============================================================
# Build stratification labels
# ============================================================

def build_stratification_key(
    y: np.ndarray,
    snr: np.ndarray,
) -> np.ndarray:
    """
    Create a joint stratification key:

        modulation + SNR

    Example:

        class 0, -20 dB -> "0_-20"
        class 0, -18 dB -> "0_-18"
        class 1, -20 dB -> "1_-20"
    """

    y = np.asarray(y).reshape(-1)
    snr = np.asarray(snr).reshape(-1)

    if len(y) != len(snr):
        raise ValueError(
            f"y and snr must have equal lengths. "
            f"Got {len(y)} and {len(snr)}."
        )

    return np.array(
        [
            f"{int(label)}_{int(snr_value)}"
            for label, snr_value in zip(y, snr)
        ]
    )


# ============================================================
# Stratified split
# ============================================================

def stratified_split(
    y: np.ndarray,
    snr: np.ndarray,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> dict:
    """
    Create a stratified 70/15/15 split.

    IMPORTANT:
        This function returns INDICES rather than signal arrays.

    Returns:
        {
            "train": indices,
            "val": indices,
            "test": indices
        }
    """

    if not np.isclose(
        train_size + val_size + test_size,
        1.0,
    ):
        raise ValueError(
            "train_size + val_size + test_size must equal 1.0"
        )

    y = np.asarray(y).reshape(-1)
    snr = np.asarray(snr).reshape(-1)

    if len(y) != len(snr):
        raise ValueError(
            "y and snr must have the same number of samples."
        )

    indices = np.arange(
        len(y),
        dtype=np.int32,
    )

    # Joint modulation + SNR stratification
    stratify_key = build_stratification_key(y, snr)

    # --------------------------------------------------------
    # First split:
    #
    # Train = 70%
    # Temporary = 30%
    # --------------------------------------------------------

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_size,
        random_state=seed,
        stratify=stratify_key,
    )

    # --------------------------------------------------------
    # Second split:
    #
    # Validation = 15%
    # Test = 15%
    #
    # 15 / (15 + 15) = 0.5
    # --------------------------------------------------------

    temp_key = stratify_key[temp_indices]

    relative_val_size = val_size / (
        val_size + test_size
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=relative_val_size,
        random_state=seed,
        stratify=temp_key,
    )

    # Sorting helps later when reading HDF5 samples.
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


# ============================================================
# Save split indices
# ============================================================

def save_processed(
    splits: dict,
    out_dir: str = OUTPUT_DIR,
) -> None:
    """
    Save train/validation/test indices as .npy files.
    """

    output_path = Path(out_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path / "train_indices.npy",
        splits["train"],
    )

    np.save(
        output_path / "val_indices.npy",
        splits["val"],
    )

    np.save(
        output_path / "test_indices.npy",
        splits["test"],
    )

    print()
    print("Saved split indices:")
    print(
        f"  train: {output_path / 'train_indices.npy'}"
    )
    print(
        f"  val  : {output_path / 'val_indices.npy'}"
    )
    print(
        f"  test : {output_path / 'test_indices.npy'}"
    )


# ============================================================
# Verify split
# ============================================================

def verify_split(
    splits: dict,
    y: np.ndarray,
    snr: np.ndarray,
) -> None:
    """
    Verify that:

    1. There is no overlap.
    2. Every sample appears exactly once.
    3. Split sizes are correct.
    4. Modulation/SNR strata are represented in all splits.
    """

    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]

    total = len(y)

    # --------------------------------------------------------
    # Size
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT VERIFICATION")
    print("=" * 70)

    print(f"Total      : {total:,}")
    print(
        f"Train      : {len(train_indices):,} "
        f"({len(train_indices) / total:.2%})"
    )
    print(
        f"Validation : {len(val_indices):,} "
        f"({len(val_indices) / total:.2%})"
    )
    print(
        f"Test       : {len(test_indices):,} "
        f"({len(test_indices) / total:.2%})"
    )

    # --------------------------------------------------------
    # No overlap
    # --------------------------------------------------------

    train_set = set(train_indices.tolist())
    val_set = set(val_indices.tolist())
    test_set = set(test_indices.tolist())

    assert not train_set.intersection(val_set), (
        "Train/validation overlap detected!"
    )

    assert not train_set.intersection(test_set), (
        "Train/test overlap detected!"
    )

    assert not val_set.intersection(test_set), (
        "Validation/test overlap detected!"
    )

    print()
    print(" No overlap between train/validation/test")

    # --------------------------------------------------------
    # Complete coverage
    # --------------------------------------------------------

    combined = np.concatenate(
        [
            train_indices,
            val_indices,
            test_indices,
        ]
    )

    assert len(combined) == total, (
        "Split does not contain all samples."
    )

    assert len(np.unique(combined)) == total, (
        "Duplicate sample indices detected."
    )

    assert combined.min() == 0
    assert combined.max() == total - 1

    print("Every sample appears exactly once")

    # --------------------------------------------------------
    # Joint distribution
    # --------------------------------------------------------

    full_key = build_stratification_key(y, snr)

    train_key = full_key[train_indices]
    val_key = full_key[val_indices]
    test_key = full_key[test_indices]

    unique_keys = np.unique(full_key)

    for key in unique_keys:

        total_count = np.sum(full_key == key)

        train_count = np.sum(train_key == key)
        val_count = np.sum(val_key == key)
        test_count = np.sum(test_key == key)

        assert (
            train_count
            + val_count
            + test_count
            == total_count
        ), f"Stratum mismatch: {key}"

    print(
        "Joint (modulation, SNR) distribution verified"
    )

    print()
    print("=" * 70)
    print("SPLIT VERIFICATION PASSED")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RadioML 2018.01A Preprocessing")
    print("=" * 70)

    dataset_path = Path(DATASET_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    print(f"Dataset: {dataset_path}")

    # --------------------------------------------------------
    # Read only Y and Z
    #
    # We DO NOT load X.
    # --------------------------------------------------------

    print()
    print("Opening HDF5 dataset...")

    with h5py.File(dataset_path, "r") as h5_file:

        print("Reading labels...")

        # Y shape:
        # (2,555,904, 24)
        #
        # Convert one-hot labels to class indices.
        y = np.argmax(
            h5_file["Y"][:],
            axis=1,
        ).astype(np.uint8)

        print("Reading SNR...")

        # Z shape:
        # (2,555,904, 1)
        snr = np.asarray(
            h5_file["Z"][:]
        ).reshape(-1).astype(np.int16)

        total_samples = h5_file["X"].shape[0]

    # --------------------------------------------------------
    # Verify metadata
    # --------------------------------------------------------

    if len(y) != total_samples:
        raise ValueError(
            "Y and X sample counts do not match."
        )

    if len(snr) != total_samples:
        raise ValueError(
            "Z and X sample counts do not match."
        )

    print()
    print(f"Total samples: {total_samples:,}")
    print(f"Labels shape : {y.shape}")
    print(f"SNR shape    : {snr.shape}")

    print()
    print(
        f"Unique modulation classes: {len(np.unique(y))}"
    )

    print(
        f"Unique SNR values: {np.unique(snr)}"
    )

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    print()
    print(
        "Creating stratified 70/15/15 split..."
    )

    splits = stratified_split(
        y=y,
        snr=snr,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_split(
        splits=splits,
        y=y,
        snr=snr,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_processed(
        splits=splits,
        out_dir=OUTPUT_DIR,
    )

    print()
    print("Preprocessing completed successfully.")