"""
main.py — XPCS analysis entry point.

Set your filepath below, then comment/uncomment the steps you want to run.
"""

from pathlib import Path

from load_data import main as load_data_summary
from plot_data import (
    show_mean_scattering,
    show_single_frame,
    show_intensity_timeseries,
    show_intensity_histogram,
    show_all,
)

# ── Set your file path here ────────────────────────────────────────────────
FILEPATH = Path(
    "/Volumes/shpyrkogroup/2024_11_APSU_XPCS_NBH10"
    "/APSU_1004574_shpyrko_202411"
    "/A344_NaBH_att000020_329K_001"
    "/A344_NaBH_att000020_329K_001_001.h5"
)
DATA_PATH = None  # HDF5 internal path (None = auto-detect)

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Print a quick data summary
    load_data_summary(FILEPATH, DATA_PATH)

    # Choose which plots to generate (comment out any you don't need):
    show_mean_scattering(FILEPATH, DATA_PATH)
    show_single_frame(FILEPATH, DATA_PATH, frame_idx=1500)
    # show_intensity_timeseries(FILEPATH, DATA_PATH)
    # show_intensity_histogram(FILEPATH, DATA_PATH, frame_idx=1500)

    # Or plot everything in one combined 2×2 figure:
    # show_all(FILEPATH, DATA_PATH, frame_idx=1500)
