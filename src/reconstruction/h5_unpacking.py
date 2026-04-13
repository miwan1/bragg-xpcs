import h5py
from pathlib import Path


data_path = Path("data")
filename = Path("A073_rocking_curve_cropped.h5")

h5_path = data_path / filename

print(f"Reading from: {h5_path}")
try:
    with h5py.File(h5_path, 'r') as f:
        print("Keys at root:", list(f.keys()))
        if '/data' in f:
            data_node = f['/data']
            print("\nContents of '/data':")
            
            if isinstance(data_node, h5py.Group):
                print("'/data' is a Group containing:")
                for key in data_node.keys():
                    item = data_node[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"  - {key}: Dataset, shape={item.shape}, dtype={item.dtype}")
                    else:
                        print(f"  - {key}: {type(item)}")
            elif isinstance(data_node, h5py.Dataset):
                print(f"'/data' is a Dataset.")
                print(f"  Shape: {data_node.shape}")
                print(f"  Dtype: {data_node.dtype}")
        else:
            print("'/data' not found in file.")
except Exception as e:
    print(f"Error accessing file: {e}")
