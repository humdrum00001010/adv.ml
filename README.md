# Advanced ML Project

## Overview
This repository contains code, scripts, and resources for an advanced machine learning project focused on facial manipulation detection and analysis. The core of the project is a variational auto‑encoder (VAE) that operates on lip landmark sequences.

## Repository Structure
- `adv/` – Core ML modules and utilities.
- `demo/` & `demo_median/` – Example runs and outputs.
- `external/` – Third‑party tools (e.g., FaceForensics++ – **not** included due to size).
- `scripts/` – Training, evaluation, data preparation, and demo generation scripts.
- `models/` – Saved model checkpoints (generated after training).
- `logs/` – Training logs (generated after training).
- `env/` – Virtual‑environment configuration (ignored in Git).

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/adv-ml.git
cd adv-ml
```

### 2. Create a Python virtual environment
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install the required packages
The repository now ships a **requirements.txt** that captures the full development environment.

```bash
pip install -r requirements.txt
```

### 4. Data fetching (the tricky part)

The project relies on the **FaceForensics++** dataset and a small synthetic lip‑landmark corpus.

#### FaceForensics++
1. Register and download the dataset from the official source:
   https://github.com/ondyari/FaceForensics
2. After download, extract the archive and place the top‑level folder inside a new `data/ffpp/` directory:
   ```
   adv-ml/
   └─ data/
      └─ ffpp/
         ├─ original/
         ├─ manipulated/
         └─ ...
   ```

#### Synthetic Lip‑Landmark Data
Run the provided script to generate a synthetic dataset:
```bash
python scripts/generate_synthetic.py --output data/synthetic/
```
The script creates NumPy files (`*.npy`) containing `(T, K, 2)` landmark tensors.

#### Kaggle Grid Corpus (optional)
If you need the Grid corpus for lip‑reading experiments, you can download it via `kagglehub`:
```bash
python scripts/download_grid_kagglehub.py
```
The script will place the data under `data/grid/`.

> **Note:** All data directories (`data/ffpp/`, `data/synthetic/`, `data/grid/`) are ignored by Git. Ensure the paths match the structure above; otherwise the data loader will not find the files.

## Reproducibility Checklist
1. **Environment** – Use the provided `requirements.txt` and the virtual environment described above.
2. **Data** – Follow the data‑fetching steps exactly; verify that the folder structure matches the examples.
3. **Random seeds** – The training script (`scripts/train.py`) sets deterministic seeds for NumPy and PyTorch.
4. **Model checkpoint** – After training, the final model is saved as `models/vae_final.pth`. You can re‑run evaluation with:
   ```bash
   python scripts/evaluate.py --model_path models/vae_final.pth
   ```
5. **Logging** – Training losses are written to `training_losses.csv` and plotted automatically.

## Running a Quick Demo
```bash
# Generate demo samples (requires the model checkpoint)
python scripts/make_demo_samples.py --model_path models/vae_final.pth --output demo/
```
The script will produce manipulated video samples and a JSON report in `demo/`.

## Contact
For questions or contributions, please reach out to:

**humdrum00001010@gmail.com**

---

*Happy hacking!*
