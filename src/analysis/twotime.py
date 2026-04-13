import torch
from boost_corr import MultitauCorrelator, TwotimeCorrelator
import boost_corr
from boost_corr.xpcs_aps_8idi.gpu_corr_multitau import solve_multitau
from boost_corr.xpcs_aps_8idi.gpu_corr_twotime import solve_twotime
import os
import h5py
import subprocess

if __name__ == '__main__':
    print(f"boost-corr version: {boost_corr.__version__}")

    # Update data paths
    data_dir = '/Users/mingzewanmba/Desktop/A343_NaBH_att000020_320K_001'
    raw_file = os.path.join(data_dir, 'A343_NaBH_att000020_320K_001_001.h5')
    qmap_file = os.path.join(data_dir, 'qmaps/A343_sq360dq36sp1dp1linearq.hdf')
    output_dir = os.path.join(data_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Select GPU or CPU (Mac doesn't have CUDA, so try -1 for CPU or 0 if MPS is supported via PyTorch device parsing)
    gpu_id = -1 # Usually -1 indicates CPU in these scripts. We can print to see if it works.

    meta_file = os.path.join(data_dir, 'A343_NaBH_att000020_320K_001_001.hdf')

    # Workaround: boost_corr refuses to save results if the .hdf file lacks a schema version
    with h5py.File(meta_file, 'a') as f:
        if '/entry/schema_version' not in f:
            f.create_dataset('/entry/schema_version', data='nexus')

    print("\n--- Running Multitau ---")
    result_mt = solve_multitau(
        raw=raw_file,
        qmap=qmap_file,
        output=output_dir,
        gpu_id=gpu_id,
        begin_frame=0,
        end_frame=-1,
        stride_frame=1,
        avg_frame=1,
        num_loaders=4,
        meta_fname=meta_file,
        save_results=True,
        verbose=True
    )

    print("\n--- Running Twotime ---")
    result_tt = solve_twotime(
        raw=raw_file,
        qmap=qmap_file,
        output=output_dir,
        gpu_id=gpu_id,
        begin_frame=0,
        end_frame=-1,
        stride_frame=1,
        avg_frame=1,
        num_loaders=4,
        meta_fname=meta_file,
        save_results=True,
        verbose=True
    )
    print("\nDone! Results saved to directory.")
    print("\nLaunching pyxpcsviewer...")
    subprocess.Popen(['pyxpcsviewer'])