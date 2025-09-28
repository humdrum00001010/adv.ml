# Advanced ML Project

## Overview
This repository contains code, scripts, and resources for an advanced machine learning project focused on facial manipulation detection and analysis. The core of the project is a variational auto‑encoder (VAE) that operates on lip landmark sequences.

## Repository Structure
- `adv/` – Core ML modules and utilities.
- `demo/` & `demo_median/` – Example runs and outputs.
- `external/` – Third‑party tools (e.g., FaceForensics++ – **not** included due to size).
- `scripts/` – Training, evaluation, data preparation, and demo generation scripts.
- `models/` – Saved model checkpoints (generated automatically).
- `logs/` – Training logs (generated automatically).
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
1. Register and download the dataset from the official source: https://github.com/ondyari/FaceForensics  
2. After download, extract the archive and place the top‑level folder inside a new `data/ffpp/` directory:
```
adv-ml/
└─ data/
   └─ ffpp/
      ├─ original/
      ├─ manipulated/
      └─ …
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

## Run FFPP Pipeline
The `run_ffpp_pipeline.py` script orchestrates the full workflow for the FaceForensics++ (FFPP) dataset: data preparation, model training, and evaluation.

```bash
python scripts/run_ffpp_pipeline.py \
    --data_dir data/ffpp \
    --model_dir models \
    --output_dir results \
    --epochs 50 \
    --batch_size 32
```

**Arguments**
- `--data_dir` Path to the FFPP data root.  
- `--model_dir` Directory where model checkpoints will be saved or loaded from.  
- `--output_dir` Directory for logs, visualisations, and final reports.  
- `--epochs` Number of training epochs (default 50).  
- `--batch_size` Batch size for training (default 32).  
- `--device` (Optional) `cpu`, `cuda`, or `mps`. If omitted, the script auto‑detects the best device.

**What the script does**
1. **Preprocess** raw videos into lip‑landmark sequences.  
2. **Train** the VAE model (or load an existing checkpoint).  
3. **Evaluate** on a held‑out split, generating ROC curves and quantitative metrics.  
4. **Save** all artefacts (model checkpoint, logs, plots) under `--output_dir`.

For a complete list of options, run:
```bash
python scripts/run_ffpp_pipeline.py --help
```

Make sure the required data is in place as described in the **Data fetching** section before running the pipeline.

## Contact
For questions or contributions, please reach out to:

**humdrum00001010@gmail.com**

---

## About Git

This project uses **Git** for version control. Below are some conventions and commands that help keep the repository clean and collaborative:

* **Branching model** – The `main` branch always contains the latest stable code. Feature work should be done on short‑lived branches named `feature/<description>` or `bugfix/<description>`.  
* **Pull requests** – Before merging a feature branch, open a PR, request a review, and ensure CI (if any) passes. Squash commits to keep history tidy.  
* **Commit messages** – Follow the conventional format:
  ```
  <type>(<scope>): <short summary>

  <optional longer description>
  ```
  Common `<type>` values are `feat`, `fix`, `docs`, `refactor`, `test`, and `chore`.  
* **Syncing** – Keep your local `main` up‑to‑date:
  ```bash
  git checkout main
  git pull origin main
  ```
* **Resolving conflicts** – If a merge conflict occurs, edit the conflicted files, stage the resolved versions with `git add <file>`, then continue the merge (`git commit` or `git merge --continue`).  
* **Tagging releases** – When a stable version is ready, create an annotated tag:
  ```bash
  git tag -a v1.0.0 -m "First stable release"
  git push origin v1.0.0
  ```

By following these practices, contributors can work efficiently and the project history remains clear and maintainable.

*Happy hacking!*