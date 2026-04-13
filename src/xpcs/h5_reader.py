"""
h5_reader.py — Inspect the contents of an HDF5 file.

Recursively prints every group, dataset (with shape, dtype, and a
small data preview), and attribute found in the file.

Usage:
    python h5_reader.py                       # uses the default path below
    python h5_reader.py path/to/file.h5       # uses a custom path
"""

import sys
from pathlib import Path

import h5py
import numpy as np

# ── Default file path (edit as needed) ──────────────────────────────────────
DEFAULT_PATH = Path(
    r"C:\Users\Andy Wan\Desktop\APSU_1004574_shpyrko_202411"
    r"\A344_NaBH_att000020_329K_001\A344_NaBH_att000020_329K_001_001.h5"
)


def print_attrs(obj, indent: int = 0) -> None:
    """Print all attributes attached to an HDF5 object."""
    prefix = "  " * indent
    for attr_name, attr_val in obj.attrs.items():
        print(f"{prefix}  [attr] {attr_name}: {attr_val}")


def print_dataset_info(name: str, dataset: h5py.Dataset, indent: int = 0) -> None:
    """Print shape, dtype, and a small preview of a dataset."""
    prefix = "  " * indent
    print(f"{prefix}📊 Dataset: {name}")
    print(f"{prefix}    shape: {dataset.shape}  dtype: {dataset.dtype}")

    # Show a compact preview of the data
    try:
        data = dataset[()]
        if isinstance(data, bytes):
            print(f"{prefix}    value: {data.decode('utf-8', errors='replace')}")
        elif isinstance(data, np.ndarray):
            if data.size <= 10:
                print(f"{prefix}    value: {data}")
            else:
                flat = data.flatten()
                preview = ", ".join(str(v) for v in flat[:5])
                print(f"{prefix}    value: [{preview}, ...] (total {data.size} elements)")
                print(f"{prefix}    min: {np.nanmin(data)}  max: {np.nanmax(data)}")
        else:
            print(f"{prefix}    value: {data}")
    except Exception as exc:
        print(f"{prefix}    ⚠ Could not read data: {exc}")

    print_attrs(dataset, indent)


def explore_group(group: h5py.Group, indent: int = 0) -> None:
    """Recursively walk groups and datasets."""
    prefix = "  " * indent
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{prefix}📂 Group: {item.name}")
            print_attrs(item, indent)
            explore_group(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print_dataset_info(item.name, item, indent)
        else:
            print(f"{prefix}❓ Unknown item: {key} ({type(item)})")


def read_h5(filepath: Path) -> None:
    """Open an HDF5 file and print its full structure."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    print(f"╔══ Reading HDF5 file ══╗")
    print(f"  {filepath}\n")

    with h5py.File(filepath, "r") as f:
        # Top-level attributes
        if f.attrs:
            print("── Root Attributes ──")
            print_attrs(f)
            print()

        # Walk the full tree
        print("── File Structure ──")
        explore_group(f)

    print(f"\n╚══ Done ══╝")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    read_h5(path)