import kagglehub
import os
import shutil
from zipfile import ZipFile

# Download latest version
path = kagglehub.dataset_download(
    "jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet"
)

print("Path to dataset files:", path)

# Determine the dataset directory (kagglehub returns a path to the dataset root or a file)
dataset_dir = path if os.path.isdir(path) else os.path.dirname(path)

# Create local directory in the project
script_dir = os.path.dirname(os.path.abspath(__file__))
local_dir = os.path.join(script_dir, "..", "data", "grid")
os.makedirs(local_dir, exist_ok=True)

# Copy the dataset to local dir
if os.path.exists(dataset_dir):
    if os.path.isdir(dataset_dir):
        shutil.copytree(dataset_dir, local_dir, dirs_exist_ok=True)
        print(f"Copied dataset directory to {local_dir}")
    else:
        shutil.copy2(dataset_dir, local_dir)
        print(f"Copied dataset file to {local_dir}")

# Explicitly extract GRID.zip if present (specific to this dataset)
zip_path = os.path.join(local_dir, "GRID.zip")
if os.path.exists(zip_path):
    extract_path = os.path.join(local_dir, "GRID")
    os.makedirs(extract_path, exist_ok=True)
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(zip_path)
    print(f"Extracted {zip_path} to {extract_path}")
else:
    print("No GRID.zip found; dataset may already be extracted.")

print(f"Dataset ready in {local_dir}")
print("Contents:")
for item in os.listdir(local_dir):
    print(f"  - {item}")
