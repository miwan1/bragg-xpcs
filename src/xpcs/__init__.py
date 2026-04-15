"""
xpcs — Bragg-XPCS data loading, analysis, and plotting utilities.

Import directly from the package for convenience::

    from xpcs import load_frames, compute_mean_scattering, DEFAULT_H5_PATH
    from xpcs import plot_mean_scattering, plot_single_frame, ...
"""

from xpcs.load_data import (
    DEFAULT_H5_PATH,
    load_frames,
    compute_mean_scattering,
    compute_frame_totals,
    main as load_data_main,
)

from xpcs.plot_data import (
    plot_mean_scattering,
    plot_single_frame,
    plot_intensity_timeseries,
    plot_intensity_histogram,
    main as plot_data_main,
)

__all__ = [
    # data loading
    "DEFAULT_H5_PATH",
    "load_frames",
    "compute_mean_scattering",
    "compute_frame_totals",
    "load_data_main",
    # plotting
    "plot_mean_scattering",
    "plot_single_frame",
    "plot_intensity_timeseries",
    "plot_intensity_histogram",
    "plot_data_main",
]
