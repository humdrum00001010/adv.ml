#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_demo_samples.py

Generate visual demo samples showing:
- original video (3s clip) and thumbnail
- fake video (3s clip) and thumbnail
- model scores and predictions for the paired landmark sequences

Assumptions:
- Landmarks are stored under: <landmarks_dir>/
    original/<comp>/real/*.npy
    DF|F2F|FS|FSH|NT|DFD/<comp>/fake/*.npy
- Raw videos are stored under: <videos_root>/
    original_sequences/youtube/<comp>/videos/<stem>.mp4
    manipulated_sequences/<method_folder>/<comp>/videos/<stem>.mp4
- A trained model exists at: <model_path> (default: models/vae_final.pth)
- This script resides in adv ml/scripts/ and will import evaluate/model/data_loader from the same repo.

Usage example:
  python3 adv\ ml/scripts/make_demo_samples.py \
    --landmarks_dir data/landmarks/ffpp \
    --videos_root data/ffpp_raw \
    --output_dir demo \
    --methods DF F2F FS FSH NT DFD \
    --compressions c23 \
    --num_samples 3 \
    --device auto \
    --tau_quantile 0.80 \
    --calib_max_reals 64
"""

import argparse
import os
import sys
import json
import glob
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add repo root (two parents up from this file: adv ml/scripts/ -> adv ml/)
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

# Make local project modules importable
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Import evaluation utilities and model
from evaluate import evaluate_sample  # type: ignore
from model import LFHFVAE  # type: ignore

# For low-pass filtering in calibration and sample scoring (match evaluate.py)
import scipy.signal as signal


METHOD_MAP: Dict[str, str] = {
    "DF": "Deepfakes",
    "F2F": "Face2Face",
    "FS": "FaceSwap",
    "FSH": "FaceShifter",
    "NT": "NeuralTextures",
    "DFD": "DeepFakeDetection",
}

DEFAULT_METHODS = ["DF", "F2F", "FS", "FSH", "NT", "DFD"]
DEFAULT_COMPRESSIONS = ["c23"]


def which_device(auto_choice: str) -> torch.device:
    """
    Resolve device choice:
    - auto: cuda if available else mps if available else cpu
    - explicit: return requested device if available, else fallback to cpu
    """
    if auto_choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if auto_choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if (
        auto_choice == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def compute_lpf(x: np.ndarray, fs: int = 25, cutoff: float = 5.0) -> np.ndarray:
    """Low-pass filter like evaluate.py (Butterworth, filtfilt)."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(4, normal_cutoff, btype="low", analog=False)
    # Pad-aware filtfilt (scipy handles padlen internally)
    return signal.filtfilt(b, a, x, axis=0)


def infer_input_dim(landmarks_dir: Path, real_subdirs: List[str]) -> int:
    """Infer input dimension (K*2) from any available real sample."""
    for sub in real_subdirs:
        p = landmarks_dir / sub
        if p.exists():
            files = sorted([f for f in p.glob("*.npy")])
            if files:
                sample = np.load(str(files[0]))
                if sample.ndim == 3:
                    return int(sample.shape[1] * sample.shape[2])
                return int(sample.shape[-1])
    # Default to 50 (20 lip points -> 40, but here some setups use 50)
    return 50


def load_model(model_path: Path, input_dim: int, device: torch.device) -> LFHFVAE:
    """Load LFHFVAE from disk."""
    model = LFHFVAE(
        input_dim=input_dim, hidden_dim=128, lf_latent_dim=32, hf_latent_dim=64
    ).to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def calibrate_tau(
    model: LFHFVAE,
    device: torch.device,
    real_npy_files: List[Path],
    tau_quantile: float,
    calib_max_reals: int = 64,
) -> float:
    """
    Compute scores for up to calib_max_reals real sequences and set tau as quantile.
    Uses evaluate.evaluate_sample to remain consistent with evaluation definitions.
    """
    if not real_npy_files:
        return 0.5  # fallback
    sel = real_npy_files[: min(calib_max_reals, len(real_npy_files))]
    real_scores: List[float] = []
    for npy_path in sel:
        x = np.load(str(npy_path))
        # Accept (T, K, 2) or (T, D); convert to torch.float32
        x_t = torch.from_numpy(x.astype(np.float32, copy=False))
        s, _, _, _, _, _, _ = evaluate_sample(
            model, x_t, device, tau=0.5
        )  # tau ignored for scoring
        real_scores.append(float(s))
    if not real_scores:
        return 0.5
    return float(np.quantile(np.array(real_scores, dtype=np.float32), tau_quantile))


def find_fake_to_real_match(
    method: str,
    comp: str,
    fake_stem: str,
    landmarks_dir: Path,
) -> Optional[Path]:
    """
    For methods with t_s style stems, map fake 't_s' -> real 't'.
    Return the original landmark path if found.
    """
    # Attempt mapping for non-DFD methods
    if method in {"DF", "F2F", "FS", "FSH", "NT"} and "_" in fake_stem:
        tgt = fake_stem.split("_", 1)[0]
        candidate = landmarks_dir / "original" / comp / "real" / f"{tgt}_landmarks.npy"
        if candidate.exists():
            return candidate
    return None


def video_path_for_real(videos_root: Path, comp: str, stem: str) -> Optional[Path]:
    cands = [
        videos_root
        / "original_sequences"
        / "youtube"
        / comp
        / "videos"
        / f"{stem}.mp4",
        videos_root
        / "original_sequences"
        / "youtube"
        / "raw"
        / "videos"
        / f"{stem}.mp4",
        videos_root / "original_sequences" / "actors" / comp / "videos" / f"{stem}.mp4",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def video_path_for_fake(
    videos_root: Path, method_folder: str, comp: str, stem: str
) -> Optional[Path]:
    cands = [
        videos_root
        / "manipulated_sequences"
        / method_folder
        / comp
        / "videos"
        / f"{stem}.mp4",
        videos_root
        / "manipulated_sequences"
        / method_folder
        / "raw"
        / "videos"
        / f"{stem}.mp4",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def clip_video(src: Path, dst: Path, seconds: int = 3) -> bool:
    """
    Use ffmpeg to clip the first `seconds` seconds from src into dst.
    Returns True on success or if src doesn't exist (no-op), False on failure.
    """
    if not src or not src.exists():
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        str(seconds),
        "-i",
        str(src),
        "-c",
        "copy",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def extract_thumbnail(src: Path, dst: Path) -> bool:
    """
    Use ffmpeg to extract the first frame as thumbnail.
    Returns True on success, False otherwise.
    """
    if not src or not src.exists():
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        "select=eq(n\\,0)",
        "-frames:v",
        "1",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def gather_candidates(
    landmarks_dir: Path, methods: List[str], compressions: List[str]
) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    """
    Build a dictionary of available real and fake landmark npy files per method and compression.
    Structure:
      data[method][comp]['fake'] -> list of Paths
      data['ORIG'][comp]['real'] -> list of Paths (shared originals)
    """
    data: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}
    # Originals
    for comp in compressions:
        orig_dir = landmarks_dir / "original" / comp / "real"
        if orig_dir.exists():
            data.setdefault("ORIG", {}).setdefault(comp, {})["real"] = sorted(
                orig_dir.glob("*.npy")
            )
        else:
            data.setdefault("ORIG", {}).setdefault(comp, {})["real"] = []

    # Fakes
    for m in methods:
        m_dir_name = m  # same short code as directory name in landmarks_dir
        for comp in compressions:
            fdir = landmarks_dir / m_dir_name / comp / "fake"
            files = sorted(fdir.glob("*.npy")) if fdir.exists() else []
            data.setdefault(m, {}).setdefault(comp, {})["fake"] = files
    return data


def select_pairs(
    data: Dict[str, Dict[str, Dict[str, List[Path]]]],
    methods: List[str],
    compressions: List[str],
    landmarks_dir: Path,
    num_samples: int,
) -> List[Tuple[str, str, Path, Path]]:
    """
    Select up to num_samples pairs per method per compression.
    Pair selection tries to match fake 't_s' -> real 't' if available, else random real.
    Returns list of tuples: (method, comp, real_npy, fake_npy)
    """
    pairs: List[Tuple[str, str, Path, Path]] = []
    for m in methods:
        if m not in data:
            continue
        for comp in compressions:
            fakes = list(data[m].get(comp, {}).get("fake", []))
            reals = list(data.get("ORIG", {}).get(comp, {}).get("real", []))
            if not fakes or not reals:
                continue
            random.shuffle(fakes)
            count = 0
            for fake_npy in fakes:
                fake_stem = fake_npy.stem.replace("_landmarks", "")
                real_match = find_fake_to_real_match(m, comp, fake_stem, landmarks_dir)
                if real_match is None or not real_match.exists():
                    # fallback to random real
                    real_match = random.choice(reals)
                pairs.append((m, comp, real_match, fake_npy))
                count += 1
                if count >= num_samples:
                    break
    return pairs


def score_sequence(model: LFHFVAE, device: torch.device, npy_path: Path) -> float:
    """Load a single npy sequence and compute score with evaluate_sample."""
    x = np.load(str(npy_path))
    x_t = torch.from_numpy(x.astype(np.float32, copy=False))
    s, _, _, _, _, _, _ = evaluate_sample(
        model, x_t, device, tau=0.5
    )  # tau ignored for scoring
    return float(s)


def main():
    parser = argparse.ArgumentParser(
        description="Create demo samples: videos, thumbnails, and model predictions."
    )
    parser.add_argument(
        "--landmarks_dir",
        type=str,
        default=str(_REPO_ROOT / "data" / "landmarks" / "ffpp"),
    )
    parser.add_argument(
        "--videos_root", type=str, default=str(_REPO_ROOT / "data" / "ffpp_raw")
    )
    parser.add_argument("--output_dir", type=str, default=str(_REPO_ROOT / "demo"))
    parser.add_argument(
        "--model_path", type=str, default=str(_REPO_ROOT / "models" / "vae_final.pth")
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=f"Methods (default: {DEFAULT_METHODS})",
    )
    parser.add_argument(
        "--compressions",
        nargs="+",
        default=DEFAULT_COMPRESSIONS,
        help=f"Compressions (default: {DEFAULT_COMPRESSIONS})",
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Pairs per method per compression"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--tau_quantile",
        type=float,
        default=0.80,
        help="Quantile for tau calibration from real-only scores",
    )
    parser.add_argument(
        "--calib_max_reals",
        type=int,
        default=64,
        help="Max real samples to use for tau calibration",
    )
    parser.add_argument(
        "--clip_seconds", type=int, default=3, help="Video clip length for demo"
    )
    args = parser.parse_args()

    landmarks_dir = Path(args.landmarks_dir).resolve()
    videos_root = Path(args.videos_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = which_device(args.device)
    print(f"[DEVICE] Using {device.type}")

    # Gather candidates and select pairs
    data = gather_candidates(landmarks_dir, args.methods, args.compressions)
    pairs = select_pairs(
        data, args.methods, args.compressions, landmarks_dir, args.num_samples
    )
    if not pairs:
        print("[WARN] No pairs found. Ensure landmarks and videos are prepared.")
        return

    # Calibrate tau from real-only pool
    # Use all available reals across requested compressions (youtube originals)
    real_pool: List[Path] = []
    for comp in args.compressions:
        real_pool.extend(data.get("ORIG", {}).get(comp, {}).get("real", []))
    if not real_pool:
        print("[WARN] No real sequences found for tau calibration. Using tau=0.5.")
    else:
        print(
            f"[INFO] Found {len(real_pool)} real sequences for tau calibration (will cap at {args.calib_max_reals})."
        )

    # Infer input_dim and load model
    real_subdirs = [f"original/{c}/real" for c in args.compressions]
    input_dim = infer_input_dim(landmarks_dir, real_subdirs)
    model = load_model(model_path, input_dim=input_dim, device=device)

    tau = calibrate_tau(
        model,
        device,
        real_pool,
        tau_quantile=args.tau_quantile,
        calib_max_reals=args.calib_max_reals,
    )
    print(f"[TAU] Calibrated tau at q={args.tau_quantile:.2f}: {tau:.6f}")

    # Process each selected pair
    for idx, (method, comp, real_npy, fake_npy) in enumerate(pairs, start=1):
        base = f"sample_{idx:02d}_{method}_{comp}"
        sample_dir = output_dir / base
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Scores and predictions
        s_real = score_sequence(model, device, real_npy)
        pred_real = int(s_real > tau)
        s_fake = score_sequence(model, device, fake_npy)
        pred_fake = int(s_fake > tau)

        # Locate videos by stems
        real_stem = real_npy.stem.replace("_landmarks", "")
        fake_stem = fake_npy.stem.replace("_landmarks", "")

        real_video = video_path_for_real(videos_root, comp, real_stem)
        fake_video = video_path_for_fake(
            videos_root, METHOD_MAP.get(method, method), comp, fake_stem
        )

        # Clip videos and export thumbnails (if available)
        real_clip = sample_dir / f"{base}_orig.mp4"
        fake_clip = sample_dir / f"{base}_fake.mp4"
        real_thumb = sample_dir / f"{base}_orig.jpg"
        fake_thumb = sample_dir / f"{base}_fake.jpg"

        if real_video and real_video.exists():
            ok = clip_video(real_video, real_clip, seconds=args.clip_seconds)
            if ok:
                extract_thumbnail(real_clip, real_thumb)
        if fake_video and fake_video.exists():
            ok = clip_video(fake_video, fake_clip, seconds=args.clip_seconds)
            if ok:
                extract_thumbnail(fake_clip, fake_thumb)

        # Save JSON metadata
        meta = {
            "method": method,
            "method_folder": METHOD_MAP.get(method, method),
            "compression": comp,
            "tau_quantile": args.tau_quantile,
            "tau": tau,
            "device": device.type,
            "model_path": str(model_path),
            "real": {
                "landmarks": str(real_npy),
                "video": str(real_video) if real_video else None,
                "clip": str(real_clip) if real_clip.exists() else None,
                "thumbnail": str(real_thumb) if real_thumb.exists() else None,
                "score": s_real,
                "pred": int(pred_real),  # 1=live, 0=spoof
            },
            "fake": {
                "landmarks": str(fake_npy),
                "video": str(fake_video) if fake_video else None,
                "clip": str(fake_clip) if fake_clip.exists() else None,
                "thumbnail": str(fake_thumb) if fake_thumb.exists() else None,
                "score": s_fake,
                "pred": int(pred_fake),  # 1=live, 0=spoof
            },
        }
        with open(sample_dir / f"{base}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[SAMPLE] {base} | real(score={s_real:.3f}, pred={pred_real}) "
            f"| fake(score={s_fake:.3f}, pred={pred_fake})"
        )

    print(f"[DONE] Wrote {len(pairs)} demo sample(s) under: {output_dir}")


if __name__ == "__main__":
    main()
