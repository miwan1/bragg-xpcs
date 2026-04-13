"""
plot_data.py — Visualize detector data from an HDF5 file.

Produces four figures:
  1. Mean scattering pattern (log-scale)
  2. Single-frame snapshot
  3. Frame-by-frame total intensity time series
  4. Intensity distribution histogram

Uses chunked I/O so that large files (e.g. 3000 × 1813 × 1558) do not
exhaust memory.

Usage:
    python plot_data.py                              # uses the default path
    python plot_data.py path/to/file.h5              # custom file
    python plot_data.py path/to/file.h5 /my/path     # custom file + dataset path
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from load_data import (
    load_frames,
    compute_mean_scattering,
    compute_frame_totals,
)

# ── Default file path (edit as needed) ──────────────────────────────────────
DEFAULT_H5_PATH = Path(
    r"C:\Users\Andy Wan\Desktop\APSU_1004574_shpyrko_202411"
    r"\A344_NaBH_att000020_329K_001\A344_NaBH_att000020_329K_001_001.h5"
)


def plot_mean_scattering(ax, mean_img: np.ndarray) -> None:
    """Log-scale image of the average scattering pattern."""
    floor = max(mean_img[mean_img > 0].min() * 0.1, 1e-3) if np.any(mean_img > 0) else 1e-3
    norm = mcolors.LogNorm(vmin=floor, vmax=mean_img.max())
    im = ax.imshow(mean_img, cmap="inferno", norm=norm, aspect="equal")
    ax.set_title("Mean Scattering Pattern (log scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.colorbar(im, ax=ax, label="Intensity", shrink=0.85)


def plot_single_frame(ax, frame: np.ndarray, frame_idx: int = 0) -> None:
    """Display a single detector frame."""
    frame = frame.astype(np.float32)
    floor = max(frame[frame > 0].min() * 0.1, 1e-3) if np.any(frame > 0) else 1e-3
    norm = mcolors.LogNorm(vmin=floor, vmax=max(frame.max(), floor * 10))
    im = ax.imshow(frame, cmap="viridis", norm=norm, aspect="equal")
    ax.set_title(f"Single Frame #{frame_idx} (log scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.colorbar(im, ax=ax, label="Intensity", shrink=0.85)


def plot_intensity_timeseries(ax, totals: np.ndarray) -> None:
    """Total intensity per frame over time."""
    frame_indices = np.arange(len(totals))

    ax.plot(frame_indices, totals, color="#4fc3f7", linewidth=0.8, alpha=0.9)
    ax.fill_between(frame_indices, totals, alpha=0.15, color="#4fc3f7")
    ax.axhline(totals.mean(), color="#ff7043", linestyle="--", linewidth=1,
               label=f"Mean = {totals.mean():.0f}")
    ax.set_title("Total Intensity per Frame", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Total Counts")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)


def plot_intensity_histogram(ax, frame: np.ndarray) -> None:
    """Histogram of nonzero pixel intensities from a sample frame (log-log)."""
    nonzero = frame[frame > 0].flatten()
    if nonzero.size == 0:
        ax.text(0.5, 0.5, "No nonzero pixels", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        return

    max_val = int(nonzero.max())
    bins = np.logspace(0, np.log10(max_val + 1), min(100, max_val + 1))
    ax.hist(nonzero, bins=bins, color="#ab47bc", edgecolor="none", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Intensity Distribution (sample frame)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25, which="both")


def main():
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_H5_PATH
    data_path = sys.argv[2] if len(sys.argv) > 2 else None

    # ── 1. Mean scattering (chunked, memory-safe) ──────────────────────────
    print(f"Loading data from: {filepath}")
    print("Computing mean scattering pattern (chunked) …")
    mean_img = compute_mean_scattering(filepath, data_path=data_path)
    print(f"  Mean pattern: shape={mean_img.shape}")

    # ── 2. Load a single frame for the snapshot + histogram ────────────────
    mid_idx = 1500  # middle of the 3000-frame dataset
    print(f"Loading single frame #{mid_idx} …")
    sample = load_frames(filepath, data_path=data_path,
                         begin_frame=mid_idx, end_frame=mid_idx + 1)
    single_frame = sample[0]
    print(f"  Frame shape: {single_frame.shape}")

    # ── 3. Per-frame totals (chunked, no full load) ────────────────────────
    print("Computing per-frame intensity totals (chunked) …")
    totals = compute_frame_totals(filepath, data_path=data_path)
    print(f"  {len(totals)} frames processed")

    # ── Build the 2×2 figure ───────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"{filepath.stem}",
        fontsize=16,
        fontweight="bold",
        color="#e0e0e0",
        y=0.98,
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_mean_scattering(ax1, mean_img)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_single_frame(ax2, single_frame, frame_idx=mid_idx)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_intensity_timeseries(ax3, totals)

    ax4 = fig.add_subplot(gs[1, 1])
    plot_intensity_histogram(ax4, single_frame)

    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.95)

    # Save and show
    out_dir = filepath.parent
    out_path = out_dir / f"{filepath.stem}_plots.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
