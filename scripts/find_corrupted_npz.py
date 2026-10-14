from pathlib import Path

import numpy as np

landmarks_dir = Path("data/landmarks")
corrupted = []

for idx, file in enumerate(landmarks_dir.glob("*.npz")):
    try:
        with np.load(str(file)) as loaded:
            _ = loaded["landmarks"]
    except Exception as e:
        print(f"Error reading {file.name}: {e}")
        corrupted.append(file)

print(f"Found {len(corrupted)} corrupted files out of {idx+1} total files.")
for f in corrupted:
    print(f"Corrupted: {f}")
