#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ffpp_pipeline.py

Orchestrates the full FaceForensics++ pipeline:
1) Extract lip landmarks for originals and manipulations (DF, F2F, FS, NT, DFD)
2) Train the LF/HF VAE on real-only landmarks
3) Evaluate per manipulation (and aggregated) with real-only calibrated threshold

Assumptions:
- This script lives in adv ml/scripts/
- The repository already contains:
  - scripts/prepare_ffpp.py
  - scripts/train.py
  - scripts/evaluate.py
- FaceForensics++ videos are downloaded locally (per their instructions), and you provide the root path.

Example:
  python3 adv\ ml/scripts/run_ffpp_pipeline.py \
      --ffpp_root /data/FaceForensics++ \
      --output_dir "data/landmarks/ffpp" \
      --compressions c23 c40 \
      --methods DF F2F FS NT DFD \
      --max_videos_per_cat 100 \
      --epochs 50 \
      --batch_size 64 \
      --device auto \
      --tau_quantile 0.80 \
      --aggregate

Notes:
- "max_videos_per_cat" controls how many videos per category (per method and compression) are processed to keep runs manageable.
- "device auto" picks CUDA if available, else MPS, else CPU.
"""

import argparse
import os
import sys
import shlex
import subprocess
from pathlib import Path
from datetime import datetime


DEFAULT_METHODS = ["DF", "F2F", "FS", "FSH", "NT", "DFD"]
DEFAULT_COMPRESSIONS = ["c23", "c40"]


def which_device(auto_choice: str) -> str:
    """
    Resolve device choice.
    - auto: cuda if available else mps if available else cpu
    - explicit: return as-is if in [cuda, mps, cpu]
    """
    if auto_choice != "auto":
        return auto_choice

    # Try CUDA
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Try MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run(cmd: str, cwd: Path = None, env: dict = None, log_path: Path = None) -> int:
    """
    Run a shell command with streaming output. Returns the exit code.
    """
    print(f"\n[RUN] {cmd}\n")
    process = subprocess.Popen(
        shlex.split(cmd),
        cwd=str(cwd) if cwd else None,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log_file = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_file:
                log_file.write(line)
    finally:
        ret = process.wait()
        if log_file:
            log_file.close()
    print(f"\n[EXIT CODE] {ret}\n")
    return ret


def ensure_scripts_exist(repo_root: Path) -> None:
    for rel in ["scripts/prepare_ffpp.py", "scripts/train.py", "scripts/evaluate.py"]:
        p = repo_root / rel
        if not p.exists():
            raise FileNotFoundError(f"Required script not found: {p}")


def build_subdirs(compressions, methods):
    """
    Build real_subdirs and (per-method and aggregate) fake_subdirs lists to feed train/eval scripts.
    Returns:
      real_subdirs: list of "original/<comp>/real"
      per_method_fake: dict method -> list of "<method>/<comp>/fake"
      agg_fake: list of all fake subdirs across methods
    """
    real_subdirs = [f"original/{c}/real" for c in compressions]
    per_method_fake = {m: [f"{m}/{c}/fake" for c in compressions] for m in methods}
    agg_fake = [sub for ms in per_method_fake.values() for sub in ms]
    return real_subdirs, per_method_fake, agg_fake


def main():
    parser = argparse.ArgumentParser(description="Run FaceForensics++ VAE pipeline")
    parser.add_argument(
        "--ffpp_root",
        type=str,
        required=True,
        help="Path to FaceForensics++ root (contains original_sequences/ and manipulated_sequences/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/landmarks/ffpp",
        help="Where to write extracted landmark .npy and metadata.json",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=f"Manipulations to include (default: {DEFAULT_METHODS})",
    )
    parser.add_argument(
        "--compressions",
        nargs="+",
        default=DEFAULT_COMPRESSIONS,
        help=f"Compression levels (default: {DEFAULT_COMPRESSIONS})",
    )
    parser.add_argument(
        "--max_videos_per_cat",
        type=int,
        default=100,
        help="Max videos per category (per method & compression) to process",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size (default: 64). Increase if you have headroom.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    # Model hyperparameters for deeper/wider models
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Model hidden dimension (TCN/GRU width).",
    )
    parser.add_argument(
        "--lf_latent_dim", type=int, default=32, help="LF branch latent dimension."
    )
    parser.add_argument(
        "--hf_latent_dim", type=int, default=64, help="HF branch latent dimension."
    )
    parser.add_argument(
        "--num_tcn_layers", type=int, default=4, help="Number of TCN layers in encoder."
    )
    parser.add_argument(
        "--tcn_kernel_size", type=int, default=3, help="TCN kernel size."
    )
    parser.add_argument("--tcn_dropout", type=float, default=0.1, help="TCN dropout.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for train/eval (default: auto)",
    )
    parser.add_argument(
        "--tau_quantile",
        type=float,
        default=0.80,
        help="Quantile for real-only threshold calibration (default: 0.80)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Also run an aggregated evaluation across all manipulations",
    )
    parser.add_argument(
        "--dry_run_extract",
        action="store_true",
        help="Skip actual extraction (useful if you already extracted landmarks)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # adv ml/
    ensure_scripts_exist(repo_root)

    # Paths
    prepare_script = repo_root / "scripts" / "prepare_ffpp.py"
    train_script = repo_root / "scripts" / "train.py"
    eval_script = repo_root / "scripts" / "evaluate.py"

    ffpp_root = Path(args.ffpp_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device resolution
    device = which_device(args.device)
    print(f"[DEVICE] Using device: {device}")

    # Build subdirs for train/eval
    real_subdirs, per_method_fake, agg_fake = build_subdirs(
        args.compressions, args.methods
    )
    print(f"[SUBDIRS] Real: {real_subdirs}")
    for m, s in per_method_fake.items():
        print(f"[SUBDIRS] Fake ({m}): {s}")
    if args.aggregate:
        print(f"[SUBDIRS] Fake (aggregate): {agg_fake}")

    # Logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = repo_root / "logs" / f"ffpp_{timestamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extract landmarks from FF++
    if not args.dry_run_extract:
        extract_cmd = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(prepare_script))} "
            f"--ffpp_root {shlex.quote(str(ffpp_root))} "
            f"--output_dir {shlex.quote(str(output_dir))} "
            f"--methods {' '.join(map(shlex.quote, args.methods))} "
            f"--compressions {' '.join(map(shlex.quote, args.compressions))} "
            f"--target_fps 25 --seq_len 64 --min_frames 48 "
            f"--smoothing_alpha 0.3 "
            f"--max_videos_per_cat {args.max_videos_per_cat}"
        )
        code = run(
            extract_cmd,
            cwd=repo_root,
            log_path=logs_dir / "01_extract_ffpp.log",
        )
        if code != 0:
            print("[ERROR] Extraction failed. Aborting.")
            sys.exit(code)
    else:
        print("[INFO] Skipping extraction (dry_run_extract).")

    # 2) Train on originals (real-only)
    #    data_dir is the output_dir; subdirs are real_subdirs we built
    train_cmd = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(train_script))} "
        f"--data_dir {shlex.quote(str(output_dir))} "
        f"--real_subdirs {' '.join(map(shlex.quote, real_subdirs))} "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--lr {args.lr} "
        f"--hidden_dim {args.hidden_dim} "
        f"--lf_latent_dim {args.lf_latent_dim} "
        f"--hf_latent_dim {args.hf_latent_dim} "
        f"--num_tcn_layers {args.num_tcn_layers} "
        f"--tcn_kernel_size {args.tcn_kernel_size} "
        f"--tcn_dropout {args.tcn_dropout} "
        f"--device {shlex.quote(device)}"
    )
    code = run(train_cmd, cwd=repo_root, log_path=logs_dir / "02_train.log")
    if code != 0:
        print("[ERROR] Training failed. Aborting.")
        sys.exit(code)

    # Resolve model path (train script writes models/vae_final.pth)
    model_path = repo_root / "models" / "vae_final.pth"
    if not model_path.exists():
        print(f"[WARN] Trained model not found at {model_path}. Continuing anyway...")

    # 3) Evaluate per manipulation (and aggregate if requested)
    #    For each method: real_subdirs vs per_method_fake[method]
    for m in args.methods:
        fake_subdirs = per_method_fake[m]
        eval_cmd = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(eval_script))} "
            f"--model_path {shlex.quote(str(model_path))} "
            f"--data_dir {shlex.quote(str(output_dir))} "
            f"--real_subdirs {' '.join(map(shlex.quote, real_subdirs))} "
            f"--fake_subdirs {' '.join(map(shlex.quote, fake_subdirs))} "
            f"--device {shlex.quote('mps' if device == 'mps' else ('cuda' if device == 'cuda' else 'cpu'))} "
            f"--batch_size 4 "
            f"--hidden_dim {args.hidden_dim} "
            f"--lf_latent_dim {args.lf_latent_dim} "
            f"--hf_latent_dim {args.hf_latent_dim} "
            f"--num_tcn_layers {args.num_tcn_layers} "
            f"--tcn_kernel_size {args.tcn_kernel_size} "
            f"--tcn_dropout {args.tcn_dropout} "
            f"--tau_quantile {args.tau_quantile}"
        )
        code = run(eval_cmd, cwd=repo_root, log_path=logs_dir / f"03_eval_{m}.log")
        if code != 0:
            print(f"[WARN] Evaluation failed for {m} (exit {code}). Continuing...")

    if args.aggregate:
        # Aggregate across all methods
        eval_cmd = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(eval_script))} "
            f"--model_path {shlex.quote(str(model_path))} "
            f"--data_dir {shlex.quote(str(output_dir))} "
            f"--real_subdirs {' '.join(map(shlex.quote, real_subdirs))} "
            f"--fake_subdirs {' '.join(map(shlex.quote, agg_fake))} "
            f"--device {shlex.quote('mps' if device == 'mps' else ('cuda' if device == 'cuda' else 'cpu'))} "
            f"--batch_size 4 "
            f"--hidden_dim {args.hidden_dim} "
            f"--lf_latent_dim {args.lf_latent_dim} "
            f"--hf_latent_dim {args.hf_latent_dim} "
            f"--num_tcn_layers {args.num_tcn_layers} "
            f"--tcn_kernel_size {args.tcn_kernel_size} "
            f"--tcn_dropout {args.tcn_dropout} "
            f"--tau_quantile {args.tau_quantile}"
        )
        code = run(eval_cmd, cwd=repo_root, log_path=logs_dir / "04_eval_agg.log")
        if code != 0:
            print(f"[WARN] Aggregate evaluation failed (exit {code}).")

    print("\n[DONE] FF++ pipeline completed.")
    print(f"[LOGS] {logs_dir}")


if __name__ == "__main__":
    main()
