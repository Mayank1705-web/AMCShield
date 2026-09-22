"""
data_loader.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 1

Responsibility:
    Lazy loading of the RadioML 2018.01A HDF5 dataset.

Dataset format:
    X: (N, 1024, 2)
    Y: (N, 24)
    Z: (N, 1)

PyTorch format:
    signal: (2, 1024)
    label : scalar class index
    snr   : scalar SNR value

Important:
    The complete 21+ GB HDF5 dataset is never loaded into RAM.
    Samples are read only when requested by the DataLoader.
"""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

DEFAULT_DATASET_PATH = (
    "data/raw/GOLD_XYZ_OSC.0001_1024.hdf5"
)

DEFAULT_PROCESSED_DIR = "data/processed"

DEFAULT_BATCH_SIZE = 128


# ============================================================
# RadioML Dataset
# ============================================================

class RadioMLDataset(Dataset):
    """
    Lazy PyTorch Dataset for RadioML 2018.01A.

    Only the requested samples are read from the HDF5 file.
    """

    def __init__(
        self,
        hdf5_path: str = DEFAULT_DATASET_PATH,
        indices_path: str | None = None,
    ):
        self.hdf5_path = Path(hdf5_path)

        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"HDF5 dataset not found: {self.hdf5_path}"
            )

        # ----------------------------------------------------
        # Load only the split indices.
        # ----------------------------------------------------

        if indices_path is None:
            self.indices = np.arange(
                self._get_dataset_length(),
                dtype=np.int32,
            )
        else:
            self.indices_path = Path(indices_path)

            if not self.indices_path.exists():
                raise FileNotFoundError(
                    f"Index file not found: {self.indices_path}"
                )

            self.indices = np.load(
                self.indices_path
            ).astype(np.int32)

        # ----------------------------------------------------
        # HDF5 file handle.
        #
        # It is opened lazily when the first sample is
        # requested. This is important for PyTorch workers.
        # ----------------------------------------------------

        self._h5_file = None

    # --------------------------------------------------------
    # HDF5 handling
    # --------------------------------------------------------

    def _get_dataset_length(self) -> int:
        """Read the number of samples without loading X."""

        with h5py.File(
            self.hdf5_path,
            "r",
        ) as h5_file:
            return h5_file["X"].shape[0]

    def _open_hdf5(self):
        """Open the HDF5 file lazily."""

        if self._h5_file is None:
            self._h5_file = h5py.File(
                self.hdf5_path,
                "r",
            )

    # --------------------------------------------------------
    # Dataset interface
    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self._open_hdf5()

        # Map DataLoader index to original HDF5 index.
        real_index = int(
            self.indices[index]
        )

        # ----------------------------------------------------
        # Read one signal.
        #
        # HDF5:
        #     (1024, 2)
        #
        # PyTorch:
        #     (2, 1024)
        # ----------------------------------------------------

        signal = self._h5_file["X"][real_index]

        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        signal = torch.from_numpy(
            signal.T.copy()
        )

        # ----------------------------------------------------
        # Read one-hot modulation label.
        #
        # Y:
        #     (24,)
        #
        # Convert to:
        #     class index 0-23
        # ----------------------------------------------------

        label_one_hot = self._h5_file["Y"][real_index]

        label = int(
            np.argmax(label_one_hot)
        )

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Read SNR.
        # ----------------------------------------------------

        snr = int(
            self._h5_file["Z"][real_index][0]
        )

        snr = torch.tensor(
            snr,
            dtype=torch.int16,
        )

        return signal, label, snr

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def close(self):
        """Close the HDF5 file."""

        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()


# ============================================================
# DataLoader creation
# ============================================================

def create_dataloaders(
    hdf5_path: str = DEFAULT_DATASET_PATH,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
):
    """
    Create train, validation and test DataLoaders.

    Returns:
        train_loader
        val_loader
        test_loader
    """

    processed_path = Path(processed_dir)

    train_dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=str(
            processed_path / "train_indices.npy"
        ),
    )

    val_dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=str(
            processed_path / "val_indices.npy"
        ),
    )

    test_dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=str(
            processed_path / "test_indices.npy"
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RadioML 2018.01A DataLoader Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = RadioMLDataset(
        indices_path=(
            "data/processed/train_indices.npy"
        )
    )

    val_dataset = RadioMLDataset(
        indices_path=(
            "data/processed/val_indices.npy"
        )
    )

    test_dataset = RadioMLDataset(
        indices_path=(
            "data/processed/test_indices.npy"
        )
    )

    print()
    print("Dataset sizes:")
    print(
        f"Train      : {len(train_dataset):,}"
    )
    print(
        f"Validation : {len(val_dataset):,}"
    )
    print(
        f"Test       : {len(test_dataset):,}"
    )

    # --------------------------------------------------------
    # Read one sample
    # --------------------------------------------------------

    print()
    print("Reading one training sample...")

    signal, label, snr = train_dataset[0]

    print(
        f"Signal shape : {tuple(signal.shape)}"
    )
    print(
        f"Signal dtype : {signal.dtype}"
    )
    print(
        f"Label        : {label.item()}"
    )
    print(
        f"SNR          : {snr.item()} dB"
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    print()
    print("Creating DataLoaders...")

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=4,
            num_workers=0,
        )
    )

    # --------------------------------------------------------
    # Read one batch
    # --------------------------------------------------------

    print()
    print("Reading one training batch...")

    signals, labels, snrs = next(
        iter(train_loader)
    )

    print(
        f"Batch signal shape : {tuple(signals.shape)}"
    )
    print(
        f"Batch label shape  : {tuple(labels.shape)}"
    )
    print(
        f"Batch SNR shape    : {tuple(snrs.shape)}"
    )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    assert signals.shape == (
        4,
        2,
        1024,
    )

    assert labels.shape == (4,)

    assert snrs.shape == (4,)

    assert signals.dtype == torch.float32

    assert labels.dtype == torch.long

    print()
    print("=" * 70)
    print("DATALOADER TEST PASSED")
    print("=" * 70)