import ast
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class DatasetService:
    CHUNK_SIZE = 100_000

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        self.hdf5_path = self.raw_dir / "GOLD_XYZ_OSC.0001_1024.hdf5"
        self.classes_path = self.raw_dir / "classes.txt"

        self.split_paths = {
            "train": self.processed_dir / "train_indices.npy",
            "validation": self.processed_dir / "val_indices.npy",
            "test": self.processed_dir / "test_indices.npy",
        }

        self._distribution_cache = None
        self._distribution_signature = None

    def _load_classes(self) -> list[str]:
        if not self.classes_path.exists():
            raise FileNotFoundError("Original classes.txt was not found.")

        tree = ast.parse(
            self.classes_path.read_text(encoding="utf-8")
        )

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "classes":
                        values = ast.literal_eval(node.value)
                        if not isinstance(values, list):
                            break
                        return [str(value) for value in values]

        raise ValueError("Could not parse the original classes.txt file.")

    def _require_dataset(self):
        if not self.hdf5_path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {self.hdf5_path}"
            )

    def _load_splits(self) -> dict[str, np.ndarray]:
        splits = {}

        for name, path in self.split_paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {name} split: {path}"
                )

            splits[name] = np.load(path, mmap_mode="r")

        return splits

    def _get_distributions(self):
        self._require_dataset()

        stat = self.hdf5_path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)

        if (
            self._distribution_cache is not None
            and self._distribution_signature == signature
        ):
            return self._distribution_cache

        classes = self._load_classes()
        class_counts = np.zeros(len(classes), dtype=np.int64)
        snr_counts: dict[int, int] = {}

        with h5py.File(self.hdf5_path, "r") as h5:
            total = h5["Y"].shape[0]

            for start in range(0, total, self.CHUNK_SIZE):
                end = min(start + self.CHUNK_SIZE, total)

                labels = np.argmax(h5["Y"][start:end], axis=1)
                snr_values = h5["Z"][start:end].reshape(-1)

                class_counts += np.bincount(
                    labels, minlength=len(classes)
                )

                unique_snr, counts = np.unique(
                    snr_values, return_counts=True
                )

                for value, count in zip(unique_snr, counts):
                    key = int(value)
                    snr_counts[key] = (
                        snr_counts.get(key, 0) + int(count)
                    )

        result = {
            "classes": classes,
            "class_counts": class_counts.tolist(),
            "snr_counts": dict(sorted(snr_counts.items())),
        }

        self._distribution_cache = result
        self._distribution_signature = signature

        return result

    def overview(self) -> dict[str, Any]:
        self._require_dataset()

        classes = self._load_classes()
        splits = self._load_splits()

        with h5py.File(self.hdf5_path, "r") as h5:
            x_shape = h5["X"].shape
            y_shape = h5["Y"].shape
            z_shape = h5["Z"].shape

            total = int(x_shape[0])

            if len(x_shape) != 3 or x_shape[2] != 2:
                raise ValueError("Unexpected HDF5 signal dimensions.")

            if y_shape[0] != total or z_shape[0] != total:
                raise ValueError("HDF5 dataset arrays have inconsistent lengths.")

        snr_counts = self._get_distributions()["snr_counts"]
        snr_values = list(snr_counts.keys())

        train_size = len(splits["train"])
        val_size = len(splits["validation"])
        test_size = len(splits["test"])

        return {
            "exists": True,
            "dataset": {
                "name": "RadioML2018.01A",
                "source": "GOLD_XYZ_OSC.0001_1024.hdf5",
                "path": "data/raw/GOLD_XYZ_OSC.0001_1024.hdf5",
                "format": "HDF5",
                "total_samples": total,
                "num_classes": len(classes),
                "classes_path": "data/raw/classes.txt",
                "signal_length": int(x_shape[1]),
                "input_shape": list(x_shape),
                "input_channels": int(x_shape[2]),
                "channel_names": ["I", "Q"],
                "snr_levels": len(snr_values),
                "snr_values": snr_values,
                "snr_min": min(snr_values) if snr_values else None,
                "snr_max": max(snr_values) if snr_values else None,
                "snr_step": (
                    snr_values[1] - snr_values[0]
                    if len(snr_values) > 1
                    else None
                ),
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "test_samples": test_size,
            },
        }

    def class_distribution(self) -> dict[str, Any]:
        data = self._get_distributions()

        return {
            "classes": [
                {"name": name, "samples": count}
                for name, count in zip(
                    data["classes"], data["class_counts"]
                )
            ]
        }

    def snr_distribution(self) -> dict[str, Any]:
        data = self._get_distributions()

        return {
            "snr": [
                {"snr": level, "samples": count}
                for level, count in data["snr_counts"].items()
            ]
        }

    def splits(self) -> dict[str, Any]:
        split_arrays = self._load_splits()
        total = sum(len(arr) for arr in split_arrays.values())

        return {
            "splits": {
                "train": {
                    "count": len(split_arrays["train"]),
                    "percentage": len(split_arrays["train"]) / total * 100,
                },
                "validation": {
                    "count": len(split_arrays["validation"]),
                    "percentage": len(split_arrays["validation"]) / total * 100,
                },
                "test": {
                    "count": len(split_arrays["test"]),
                    "percentage": len(split_arrays["test"]) / total * 100,
                },
            },
            "total_samples": total,
        }

    def quality(self) -> dict[str, Any]:
        checks = []

        def add(name: str, passed: bool, detail: str):
            checks.append({
                "name": name,
                "ok": bool(passed),
                "status": "passed" if passed else "failed",
                "detail": detail,
            })

        add(
            "HDF5 dataset",
            self.hdf5_path.is_file(),
            "Original HDF5 dataset file is available."
            if self.hdf5_path.is_file()
            else "HDF5 dataset file is missing.",
        )

        add(
            "Original class metadata",
            self.classes_path.is_file(),
            "Original raw class ordering is available."
            if self.classes_path.is_file()
            else "classes.txt is missing.",
        )

        try:
            self._require_dataset()

            with h5py.File(self.hdf5_path, "r") as h5:
                x = h5["X"]
                y = h5["Y"]
                z = h5["Z"]

                valid_shapes = (
                    len(x.shape) == 3
                    and x.shape[1:] == (1024, 2)
                    and y.shape == (x.shape[0], 24)
                    and z.shape == (x.shape[0], 1)
                )

                valid_dtypes = (
                    x.dtype == np.dtype("float32")
                    and y.dtype == np.dtype("int64")
                    and z.dtype == np.dtype("int64")
                )

                total = x.shape[0]

            add(
                "HDF5 dimensions",
                valid_shapes,
                f"X={x.shape}, Y={y.shape}, Z={z.shape}",
            )

            add(
                "HDF5 data types",
                valid_dtypes,
                f"X={x.dtype}, Y={y.dtype}, Z={z.dtype}",
            )

            splits = self._load_splits()

            split_arrays = [
                np.asarray(splits[name])
                for name in ("train", "validation", "test")
            ]

            combined = np.concatenate(split_arrays)
            unique_count = np.unique(combined).size

            in_range = (
                combined.size > 0
                and int(combined.min()) >= 0
                and int(combined.max()) < total
            )

            add(
                "Split index range",
                in_range,
                "All split indices are within HDF5 sample boundaries."
                if in_range
                else "One or more split indices are out of range.",
            )

            no_overlap = unique_count == combined.size

            add(
                "Split independence",
                no_overlap,
                "Train, validation and test contain no overlapping indices."
                if no_overlap
                else f"Found {combined.size - unique_count:,} overlapping indices.",
            )

            complete = unique_count == total and combined.size == total

            add(
                "Split coverage",
                complete,
                f"{unique_count:,} unique indices for {total:,} samples."
                if complete
                else f"Found {unique_count:,} unique indices for {total:,} samples.",
            )

        except Exception as exc:
            add("Dataset integrity", False, str(exc))

        return {
            "checks": checks,
            "passed": all(item["ok"] for item in checks),
            "total_checks": len(checks),
        }

    def sample(self, class_name: str, snr: int) -> dict[str, Any]:
        data = self._get_distributions()

        if class_name not in data["classes"]:
            raise ValueError(f"Unknown modulation class: {class_name}")

        if int(snr) not in data["snr_counts"]:
            raise ValueError(f"Unknown SNR level: {snr}")

        class_index = data["classes"].index(class_name)
        target_snr = int(snr)

        with h5py.File(self.hdf5_path, "r") as h5:
            total = h5["Y"].shape[0]
            selected_index = None

            for start in range(0, total, self.CHUNK_SIZE):
                end = min(start + self.CHUNK_SIZE, total)

                labels = np.argmax(h5["Y"][start:end], axis=1)
                snr_values = h5["Z"][start:end].reshape(-1)

                matches = np.flatnonzero(
                    (labels == class_index)
                    & (snr_values == target_snr)
                )

                if matches.size:
                    selected_index = start + int(matches[0])
                    break

            if selected_index is None:
                raise LookupError(
                    "No sample found for the selected class and SNR."
                )

            signal = np.asarray(
                h5["X"][selected_index],
                dtype=np.float32,
            )

        energy = float(np.sqrt(np.sum(signal ** 2)))
        signal = signal / max(energy, 1e-12)

        return {
            "class_name": class_name,
            "snr": target_snr,
            "index": selected_index,
            "samples": signal.tolist(),
            "sample_count": int(signal.shape[0]),
            "channels": ["I", "Q"],
        }
