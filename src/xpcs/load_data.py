"""
load_data.py — Load detector frame data from an HDF5 file.

Uses the same reading pattern as HdfDataset in hdf_handler.py:
  • Opens the file with h5py (+ hdf5plugin for blosc compression)
  • Reads the 3-D image stack at the configured data path
  • Applies hdf_handler-style dtype conversion (uint8→int16, uint16→int16/int32, etc.)
  • Provides helper functions for batch access and average scattering

Usage:
    python load_data.py                            # uses the default path
    python load_data.py path/to/file.h5            # custom file
    python load_data.py path/to/file.h5 /my/path   # custom file + dataset path
"""

import sys
from pathlib import Path

import h5py
import hdf5plugin          # registers blosc / bitshuffle filters
import numpy as np

# ── Default file path (edit as needed) ──────────────────────────────────────
DEFAULT_H5_PATH = Path(
    r"C:\Users\Andy Wan\Desktop\APSU_1004574_shpyrko_202411"
    r"\A344_NaBH_att000020_329K_001\A344_NaBH_att000020_329K_001_001.h5"
)


# ── dtype conversion (mirrors HdfDataset.__init__) ─────────────────────────
LAMBDA2M_PIXELS = 1813 * 1558


def _convert_dtype(original_dtype: np.dtype, frame_pixels: int) -> np.dtype:
    """
    Reproduce the dtype promotion logic from hdf_handler.py so that pixel
    values never overflow during downstream summation/averaging.
    """
    if original_dtype == np.uint8:
        return np.dtype(np.int16)
    if original_dtype == np.uint16:
        if frame_pixels == LAMBDA2M_PIXELS:
            return np.dtype(np.int16)     # lambda2m 12-bit data
        return np.dtype(np.int32)         # likely Eiger
    if original_dtype == np.uint32:
        return np.dtype(np.int32)
    return original_dtype


# ── dataset path auto-detection ─────────────────────────────────────────────
# hdf_handler uses "/entry/data/data" by default, but the local cropped file
# stores data at "/data".  We try common paths in order.
CANDIDATE_PATHS = ["/entry/data/data", "/data", "/entry/data"]


def _find_dataset_path(f: h5py.File) -> str:
    """Return the first matching dataset path in the file."""
    for p in CANDIDATE_PATHS:
        if p in f and isinstance(f[p], h5py.Dataset):
            return p
    raise KeyError(
        f"Could not find a dataset at any of {CANDIDATE_PATHS}. "
        "Pass the correct path as the `data_path` argument."
    )


# ── public API ──────────────────────────────────────────────────────────────
def load_frames(
    filepath: Path | str,
    data_path: str | None = None,
    begin_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    chunk_size: int = 32,
) -> np.ndarray:
    """
    Load a 3-D stack of detector frames from an HDF5 file.

    Reads in chunks to avoid loading the entire dataset into memory at once
    (mirrors the batch-wise reading in HdfDataset.__getbatch__).

    Parameters
    ----------
    filepath : path to the .h5 file
    data_path : HDF5 internal path to the dataset (auto-detected if None)
    begin_frame : first frame index to load (inclusive)
    end_frame : last frame index to load (exclusive, -1 = all)
    stride : step between frames
    chunk_size : number of frames to read per I/O operation

    Returns
    -------
    frames : np.ndarray, shape (N, H, W)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        if data_path is None:
            data_path = _find_dataset_path(f)

        ds = f[data_path]
        n_total, *frame_shape = ds.shape
        frame_pixels = int(np.prod(frame_shape))

        # Resolve frame range
        if end_frame <= 0:
            end_frame = n_total
        end_frame = min(end_frame, n_total)

        target_dtype = _convert_dtype(ds.dtype, frame_pixels)
        indices = np.arange(begin_frame, end_frame, stride)

        # Read in chunks to keep peak memory bounded
        parts = []
        for i in range(0, len(indices), chunk_size):
            idx_chunk = indices[i : i + chunk_size]
            chunk = ds[idx_chunk[0] : idx_chunk[-1] + 1]
            # If stride > 1, we grabbed a contiguous block; subsample it
            if stride > 1:
                local_sel = idx_chunk - idx_chunk[0]
                chunk = chunk[local_sel]
            if chunk.dtype != target_dtype:
                chunk = chunk.astype(target_dtype)
            parts.append(chunk)

        frames = np.concatenate(parts, axis=0)

    return frames


def compute_mean_scattering(
    filepath: Path | str,
    data_path: str | None = None,
    begin_frame: int = 0,
    end_frame: int = -1,
    chunk_size: int = 32,
) -> np.ndarray:
    """
    Compute the mean scattering pattern (average over all frames).

    Reads the file in chunks to keep memory usage bounded—mirrors the
    batch-wise summation in HdfDataset.get_scattering().

    Parameters
    ----------
    filepath : path to the .h5 file
    data_path : HDF5 internal path
    begin_frame, end_frame : frame range
    chunk_size : number of frames to read per chunk

    Returns
    -------
    mean_pattern : np.ndarray, shape (H, W)
    """
    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        if data_path is None:
            data_path = _find_dataset_path(f)

        ds = f[data_path]
        n_total = ds.shape[0]
        frame_shape = ds.shape[1:]
        frame_pixels = int(np.prod(frame_shape))
        target_dtype = _convert_dtype(ds.dtype, frame_pixels)

        if end_frame <= 0:
            end_frame = n_total
        end_frame = min(end_frame, n_total)

        accumulator = np.zeros(frame_shape, dtype=np.float64)
        n_frames = 0

        for start in range(begin_frame, end_frame, chunk_size):
            stop = min(start + chunk_size, end_frame)
            chunk = ds[start:stop].astype(target_dtype)
            accumulator += chunk.sum(axis=0)
            n_frames += chunk.shape[0]

    mean_pattern = (accumulator / n_frames).astype(np.float32)
    return mean_pattern


def compute_frame_totals(
    filepath: Path | str,
    data_path: str | None = None,
    begin_frame: int = 0,
    end_frame: int = -1,
    chunk_size: int = 32,
) -> np.ndarray:
    """
    Compute the total intensity per frame without loading all frames at once.

    Returns
    -------
    totals : np.ndarray, shape (N_frames,)
    """
    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        if data_path is None:
            data_path = _find_dataset_path(f)

        ds = f[data_path]
        n_total = ds.shape[0]

        if end_frame <= 0:
            end_frame = n_total
        end_frame = min(end_frame, n_total)

        totals = []
        for start in range(begin_frame, end_frame, chunk_size):
            stop = min(start + chunk_size, end_frame)
            chunk = ds[start:stop].astype(np.float64)
            totals.append(chunk.sum(axis=(1, 2)))

    return np.concatenate(totals)


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_H5_PATH
    data_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading data from: {filepath}")

    # Just load a small sample for the summary stats
    print("Reading first 10 frames for quick summary ...")
    sample = load_frames(filepath, data_path=data_path, end_frame=10)
    print(f"  Sample frames : shape={sample.shape}  dtype={sample.dtype}")
    print(f"  Min value     : {sample.min()}")
    print(f"  Max value     : {sample.max()}")
    print(f"  Mean value    : {sample.mean():.4f}")

    print("\nComputing mean scattering pattern (all frames) ...")
    mean_img = compute_mean_scattering(filepath, data_path=data_path)
    print(f"  Mean pattern  : shape={mean_img.shape}  dtype={mean_img.dtype}")
    print(f"  Min pixel     : {mean_img.min():.4f}")
    print(f"  Max pixel     : {mean_img.max():.4f}")

    print("\nComputing per-frame intensity totals ...")
    totals = compute_frame_totals(filepath, data_path=data_path)
    print(f"  Total frames  : {len(totals)}")
    print(f"  Avg intensity : {totals.mean():.2f}")
    print(f"  Std intensity : {totals.std():.2f}")


if __name__ == "__main__":
    main()
