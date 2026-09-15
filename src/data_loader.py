"""
data_loader.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 1

Responsibility:
    Load RadioML 2018.01A from its HDF5 file into a PyTorch-compatible
    dataset without loading the complete 21+ GB file into RAM.

Dataset:
    RadioML 2018.01A

Raw HDF5 structure:
    X : (N, 1024, 2) float32
        I/Q signal samples

    Y : (N, 24) int64
        One-hot modulation labels

    Z : (N, 1) int64
        SNR value in dB

Returned PyTorch sample:
    signal : Tensor, shape (2, 1024)
    label  : Tensor, scalar class index [0, 23]
    snr    : int, SNR value in dB

IMPORTANT:
    SNR is never dropped. It remains aligned with every signal and label.
    Downstream accuracy-vs-SNR and attack generalization evaluation depend
    on this alignment.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Constants

NUM_CLASSES = 24
SIGNAL_LENGTH = 1024
NUM_CHANNELS = 2

# RadioML 2018.01A Dataset

class RadioML2018Dataset(Dataset):
    """
    Lazy HDF5-backed PyTorch Dataset.
    The complete HDF5 file is NOT loaded into memory.
    Only the requested sample is read when __getitem__() is called.
    """
    def __init__(
        self,
        hdf5_path: str,
        indices: Optional[Sequence[int]] = None,
    ):
        self.hdf5_path = Path(hdf5_path)

        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"RadioML dataset not found:\n{self.hdf5_path}"
            )

        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else None
        )

        # Do not keep an HDF5 file open while constructing the dataset.
        # The file is opened lazily when a sample is requested.
        self._h5_file = None

        # Read metadata only.
        with h5py.File(self.hdf5_path, "r") as h5_file:
            required_keys = {"X", "Y", "Z"}
            if not required_keys.issubset(h5_file.keys()):
                raise ValueError(
                    "Invalid RadioML HDF5 file.\n"
                    f"Expected datasets: {required_keys}\n"
                    f"Found: {list(h5_file.keys())}"
                )

            x_shape = h5_file["X"].shape
            y_shape = h5_file["Y"].shape
            z_shape = h5_file["Z"].shape

            self.length = x_shape[0]

            # Validate X

            if x_shape[1:] != (SIGNAL_LENGTH, NUM_CHANNELS):
                raise ValueError(
                    "Unexpected X shape.\n"
                    f"Expected: (N, {SIGNAL_LENGTH}, {NUM_CHANNELS})\n"
                    f"Found: {x_shape}"
                )

            # Validate Y

            if y_shape != (self.length, NUM_CLASSES):
                raise ValueError(
                    "Unexpected Y shape.\n"
                    f"Expected: (N, {NUM_CLASSES})\n"
                    f"Found: {y_shape}"
                )

            # Validate Z

            if z_shape != (self.length, 1):
                raise ValueError(
                    "Unexpected Z shape.\n"
                    f"Expected: (N, 1)\n"
                    f"Found: {z_shape}"
                )

    # HDF5 handling

    def _open_file(self):
        """
        Open the HDF5 file lazily.
        This prevents the complete dataset from being loaded
        into RAM.
        """

        if self._h5_file is None:
            self._h5_file = h5py.File(
                self.hdf5_path,
                "r",
            )

    def close(self):
        """Close the HDF5 file if it is open."""

        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        """Attempt to close the HDF5 file when the object is destroyed."""

        try:
            self.close()
        except Exception:
            pass

    # PyTorch Dataset interface
    
    def __len__(self) -> int:
        """Return number of available samples."""

        if self.indices is not None:
            return len(self.indices)

        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Read one sample.
        Returns:
            signal:
                torch.FloatTensor of shape (2, 1024)
            label:
                torch.LongTensor containing class index 0-23
            snr:
                Integer SNR value in dB
        """

        self._open_file()

        # Resolve actual HDF5 index

        if self.indices is not None:
            real_index = int(self.indices[index])
        else:
            real_index = index
            
        # Read ONE sample from HDF5

        signal = self._h5_file["X"][real_index]
        label_one_hot = self._h5_file["Y"][real_index]
        snr_value = self._h5_file["Z"][real_index]

        # ----------------------------------------------------
        # Signal
        #
        # HDF5:
        #     (1024, 2)
        #
        # PyTorch:
        #     (2, 1024)
        # ----------------------------------------------------

        signal = torch.from_numpy(
            np.asarray(signal, dtype=np.float32)
        )

        signal = signal.transpose(0, 1).contiguous()

        # ----------------------------------------------------
        # Label
        #
        # Y is one-hot:
        #
        # [0, 0, 1, 0, ...]
        #
        # Convert to:
        #
        # 2
        # ----------------------------------------------------

        label = torch.tensor(
            int(np.argmax(label_one_hot)),
            dtype=torch.long,
        )

        # SNR

        snr = int(snr_value[0])
        return signal, label, snr


# DataLoader helper

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.
    Windows note:
        num_workers=0 is intentionally used initially because
        HDF5 + multiprocessing requires additional care.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

# Dataset inspection

def inspect_dataset(hdf5_path: str) -> None:
    """
    Print HDF5 metadata.
    This function only reads dataset metadata and does not load
    the complete dataset into RAM.
    """

    path = Path(hdf5_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    with h5py.File(path, "r") as h5_file:
        print("=" * 70)
        print("RadioML 2018.01A Dataset")
        print("=" * 70)
        print(f"File: {path}")
        print()

        for key in h5_file.keys():
            dataset = h5_file[key]

            print(
                f"{key}: "
                f"shape={dataset.shape}, "
                f"dtype={dataset.dtype}"
            )
        print("=" * 70)

# Sample test

def test_loader(hdf5_path: str) -> None:
    """
    Perform a small sanity test.
    Only one sample and one small batch are read.
    """

    print()
    print("Creating lazy dataset...")

    dataset = RadioML2018Dataset(hdf5_path)

    print(f"Dataset length: {len(dataset):,}")

    # Test one sample

    print()
    print("Reading one sample...")

    signal, label, snr = dataset[0]

    print(f"Signal shape : {tuple(signal.shape)}")
    print(f"Signal dtype : {signal.dtype}")
    print(f"Label        : {label.item()}")
    print(f"SNR          : {snr} dB")

    # Test DataLoader
    
    print()
    print("Creating DataLoader...")

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch_signal, batch_label, batch_snr = next(iter(loader))

    print(f"Batch signal shape : {tuple(batch_signal.shape)}")
    print(f"Batch label shape  : {tuple(batch_label.shape)}")
    print(f"Batch SNR shape    : {tuple(batch_snr.shape)}")

    print()
    print("Loader test completed successfully.")

    dataset.close()

# Main

if __name__ == "__main__":

    DATASET_PATH = (
        "data/raw/GOLD_XYZ_OSC.0001_1024.hdf5"
    )
    inspect_dataset(DATASET_PATH)
    test_loader(DATASET_PATH)