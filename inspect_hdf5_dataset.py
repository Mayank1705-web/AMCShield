
from pathlib import Path
import h5py
import json

DATASET_PATH = Path(
    r"D:\MediCaps University\7 SEM\Project 1\AMCShield\data\raw\GOLD_XYZ_OSC.0001_1024.hdf5"
)

print("\n========== AMCShield DATASET INSPECTION ==========")

if not DATASET_PATH.exists():
    raise FileNotFoundError(DATASET_PATH)

print(f"\nFile: {DATASET_PATH.name}")
print(f"Size: {DATASET_PATH.stat().st_size / (1024**3):.3f} GiB")

with h5py.File(DATASET_PATH, "r") as f:

    print("\n========== ROOT KEYS ==========")
    print(list(f.keys()))

    print("\n========== ROOT ATTRIBUTES ==========")
    for key, value in f.attrs.items():
        print(f"{key}: {value}")

    print("\n========== INTERNAL STRUCTURE ==========")

    count = 0

    def inspect(name, obj):
        global count
        count += 1

        print(f"\nObject: {name}")

        if isinstance(obj, h5py.Group):
            print("Type: GROUP")
            print("Children:", list(obj.keys()))

        elif isinstance(obj, h5py.Dataset):
            print("Type: DATASET")
            print("Shape:", obj.shape)
            print("Dtype:", obj.dtype)
            print("Chunks:", obj.chunks)
            print("Compression:", obj.compression)

            for key, value in obj.attrs.items():
                print(f"Attribute {key}: {value}")

    f.visititems(inspect)

    print("\n========== INSPECTION COMPLETE ==========")
    print("Total HDF5 objects:", count)

print("\nOriginal dataset was opened in read-only mode.")
