#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_ffpp.py

FaceForensics++ integration script:
- Scans an FF++ root directory for original and manipulated videos
- Supports manipulations: Deepfakes (DF), Face2Face (F2F), FaceSwap (FS), NeuralTextures (NT), DeepFakeDetection (DFD)
- Supports compression levels: c0 (lossless/raw), c23, c40 (default: c23, c40)
- Extracts lip landmarks as (T, K, 2) arrays (K=20) using MediaPipe FaceMesh
- Saves .npy and metadata.json under:
    <output_dir>/
      original/<comp>/real/*.npy
      DF|F2F|FS|NT|DFD/<comp>/fake/*.npy

Typical FF++ layout (as downloaded by their script):
- original_sequences/youtube/&lt;comp&gt;/videos/*.mp4
- original_sequences/actors/&lt;comp&gt;/videos/*.mp4  (DeepFakeDetection originals)
- manipulated_sequences/Deepfakes|Face2Face|FaceSwap|NeuralTextures/&lt;comp&gt;/videos/*.mp4
- DeepFakeDetection is also hosted; path may vary, e.g. manipulated_sequences/DeepFakeDetection/&lt;comp&gt;/videos/*.mp4

Usage:
    python3 adv\ ml/scripts/prepare_ffpp.py \
        --ffpp_root /path/to/FaceForensics++ \
        --output_dir data/landmarks/ffpp \
        --methods DF F2F FS NT DFD \
        --compressions c23 c40 \
        --target_fps 25 --seq_len 64 --min_frames 48 \
        --max_videos_per_cat 100

Notes:
- This script depends on OpenCV and MediaPipe. Ensure they are installed in your environment.
- If some categories or compressions are missing on disk, they are skipped with a warning.
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

try:
    import mediapipe as mp
except Exception as e:
    print(
        "ERROR: mediapipe is not available. Please install it (pip install mediapipe)."
    )
    sys.exit(1)

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


# -----------------------------
# Configuration and constants
# -----------------------------

# Map method short names to FF++ folder names under 'manipulated_sequences'
METHOD_MAP = {
    "DF": "Deepfakes",
    "F2F": "Face2Face",
    "FS": "FaceSwap",
    "FSH": "FaceShifter",
    "NT": "NeuralTextures",
    "DFD": "DeepFakeDetection",
}

# Default supported video extensions
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm")


# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh


# 20 ordered lip landmark indices for MediaPipe FaceMesh
# A curated subset for stable external/internal lip contour coverage
LIP_INDICES = [
    61,
    291,
    0,
    37,
    39,
    40,
    185,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    78,
    95,
    88,
    178,
    87,
    317,
][:20]  # Ensure exactly 20 points


# -----------------------------
# Utility functions
# -----------------------------


def discover_original_videos(ffpp_root: Path, compression: str) -> List[Path]:
    """
    Discover original sequences (YouTube and DeepFakeDetection actors) in FF++ for a given compression.
    Typical paths:
      - original_sequences/youtube/<comp>/videos/*.mp4 (or raw)
      - original_sequences/actors/<comp>/videos/*.mp4 (or raw)
    """
    patterns = [
        # YouTube originals
        ffpp_root / "original_sequences" / "youtube" / compression / "videos" / "*",
        ffpp_root / "original_sequences" / "youtube" / "raw" / "videos" / "*",
        # DFD actor originals
        ffpp_root / "original_sequences" / "actors" / compression / "videos" / "*",
        ffpp_root / "original_sequences" / "actors" / "raw" / "videos" / "*",
    ]
    files: List[Path] = []
    for p in patterns:
        for f in glob.glob(str(p)):
            fp = Path(f)
            if fp.suffix.lower() in VIDEO_EXTS and fp.is_file():
                files.append(fp)
    # Deduplicate and sort
    files = sorted(list({str(p): Path(p) for p in files}.values()))
    return files


def discover_manip_videos(
    ffpp_root: Path, method_folder: str, compression: str
) -> List[Path]:
    """
    Discover manipulated sequences in FF++ for a given method and compression.
    Typical path: manipulated_sequences/<method>/<comp>/videos/*.mp4
    """
    patterns = [
        ffpp_root
        / "manipulated_sequences"
        / method_folder
        / compression
        / "videos"
        / "*",
        # Some variants may use a different compression folder naming:
        ffpp_root / "manipulated_sequences" / method_folder / "raw" / "videos" / "*",
    ]
    files: List[Path] = []
    for p in patterns:
        for f in glob.glob(str(p)):
            fp = Path(f)
            if fp.suffix.lower() in VIDEO_EXTS and fp.is_file():
                files.append(fp)
    return sorted(files)


def ema_smooth(seq: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Apply EMA smoothing along time dim for (T, K, 2).
    """
    if alpha <= 0.0:
        return seq
    out = np.zeros_like(seq)
    out[0] = seq[0]
    for t in range(1, seq.shape[0]):
        out[t] = alpha * seq[t] + (1.0 - alpha) * out[t - 1]
    return out


def extract_lip_landmarks_from_video(
    video_path: Path,
    target_fps: int = 25,
    min_frames: int = 48,
    seq_len: int = 64,
    smoothing_alpha: float = 0.3,
    normalize_lip_relative: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Extract lip landmarks from a single video using MediaPipe FaceMesh.

    Returns:
        seq: np.ndarray with shape (T, 20, 2) or None on failure.
        meta: dict with metadata or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frames_total <= 0:
        cap.release()
        return None, None

    # Downsample to target_fps
    step = max(1, int(round(fps / float(target_fps)))) if fps > 0 else 1
    frame_indices = list(range(0, frames_total, step))
    if len(frame_indices) < min_frames:
        cap.release()
        return None, None

    # Trim or pad to seq_len
    if len(frame_indices) > seq_len:
        frame_indices = frame_indices[:seq_len]
    T = len(frame_indices)

    lip_seq = []
    face_detected = 0
    # MediaPipe FaceMesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as face_mesh:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                # Repeat last if read fails
                if lip_seq:
                    lip_seq.append(lip_seq[-1])
                else:
                    lip_seq.append(np.zeros((20, 2), dtype=np.float32))
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0]
                pts = []
                for li in LIP_INDICES:
                    lmk = landmarks.landmark[li]
                    # MediaPipe coordinates are already in [0,1]
                    pts.append([lmk.x, lmk.y])
                pts = np.asarray(pts, dtype=np.float32)

                if normalize_lip_relative:
                    # Normalize to per-frame lip bounding box to [0,1]
                    mn = pts.min(axis=0)
                    mx = pts.max(axis=0)
                    rng = np.maximum(mx - mn, 1e-6)
                    pts = (pts - mn) / rng

                lip_seq.append(pts)
                face_detected += 1
            else:
                if lip_seq:
                    lip_seq.append(lip_seq[-1])
                else:
                    lip_seq.append(np.zeros((20, 2), dtype=np.float32))

    cap.release()

    if len(lip_seq) == 0:
        return None, None

    seq = np.asarray(lip_seq, dtype=np.float32)  # (T, 20, 2)
    # Pad if shorter than seq_len
    if seq.shape[0] < seq_len:
        pad_len = seq_len - seq.shape[0]
        pad = np.tile(seq[-1:], (pad_len, 1, 1))
        seq = np.concatenate([seq, pad], axis=0)

    # Smoothing
    if smoothing_alpha > 0.0:
        seq = ema_smooth(seq, alpha=smoothing_alpha)

    meta = {
        "video_path": str(video_path),
        "original_fps": float(fps),
        "frames_total": frames_total,
        "extracted_frames": int(T),
        "face_detected_frames": int(face_detected),
        "face_detected_rate": float(face_detected / max(1, T)),
        "target_fps": target_fps,
        "seq_len": seq_len,
        "normalized_to_lip_bbox": bool(normalize_lip_relative),
        "timestamp": int(time.time()),
    }
    return seq, meta


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_sequence(out_dir: Path, base_stem: str, arr: np.ndarray, meta: Dict) -> None:
    npy_path = out_dir / f"{base_stem}_landmarks.npy"
    meta_path = out_dir / f"{base_stem}_metadata.json"
    np.save(npy_path, arr)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# -----------------------------
# Main processing
# -----------------------------


def process_category(
    name: str,
    videos: List[Path],
    out_dir: Path,
    is_real: bool,
    args: argparse.Namespace,
) -> Tuple[int, int]:
    """
    Process a list of videos for a given category (original or manipulation).
    Saves outputs in out_dir/real or out_dir/fake accordingly.
    """
    label_dir = out_dir / ("real" if is_real else "fake")
    ensure_dir(label_dir)

    total = len(videos)
    processed = 0

    iterator = tqdm(videos, desc=f"{name}", unit="vid") if TQDM_AVAILABLE else videos
    for vpath in iterator:
        base_stem = vpath.stem
        npy_path = label_dir / f"{base_stem}_landmarks.npy"
        meta_path = label_dir / f"{base_stem}_metadata.json"
        if npy_path.exists() and meta_path.exists():
            # Skip existing
            processed += 1
            continue

        seq, meta = extract_lip_landmarks_from_video(
            vpath,
            target_fps=args.target_fps,
            min_frames=args.min_frames,
            seq_len=args.seq_len,
            smoothing_alpha=args.smoothing_alpha,
            normalize_lip_relative=args.normalize_lip_relative,
        )
        if seq is None:
            continue

        try:
            save_sequence(label_dir, base_stem, seq, meta)
            processed += 1
        except Exception as e:
            print(f"Failed to save {vpath}: {e}")

        if args.max_videos_per_cat and processed >= args.max_videos_per_cat:
            break

    return processed, total


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FaceForensics++ lip landmark dataset"
    )
    parser.add_argument(
        "--ffpp_root",
        type=str,
        required=True,
        help="Path to FaceForensics++ root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/landmarks/ffpp",
        help="Output directory for landmarks",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["DF", "F2F", "FS", "FSH", "NT", "DFD"],
        help="Manipulation methods to include: DF F2F FS FSH NT DFD",
    )
    parser.add_argument(
        "--compressions",
        nargs="+",
        default=["c23", "c40"],
        help="Compression levels to include (e.g., c0 c23 c40). Default: c23 c40",
    )

    parser.add_argument(
        "--target_fps", type=int, default=25, help="Target FPS for downsampling"
    )
    parser.add_argument(
        "--seq_len", type=int, default=64, help="Sequence length (pad/truncate)"
    )
    parser.add_argument(
        "--min_frames", type=int, default=48, help="Minimum frames after downsampling"
    )
    parser.add_argument(
        "--smoothing_alpha",
        type=float,
        default=0.3,
        help="EMA smoothing factor in [0,1]",
    )
    parser.add_argument(
        "--normalize_lip_relative",
        action="store_true",
        help="Normalize coordinates per-frame by lip bbox to [0,1]",
    )
    parser.add_argument(
        "--max_videos_per_cat",
        type=int,
        default=None,
        help="Max videos per category (debugging)",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Only list counts, do not process videos"
    )

    args = parser.parse_args()
    ffpp_root = Path(args.ffpp_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Validate methods
    methods = []
    for m in args.methods:
        if m not in METHOD_MAP:
            print(f"WARNING: Unknown method '{m}' - skipping")
            continue
        methods.append(m)

    if not methods:
        print("No valid methods specified. Exiting.")
        sys.exit(0)

    # Process originals and each manipulation per compression
    grand_total = 0
    grand_done = 0

    for comp in args.compressions:
        # Originals
        orig_videos = discover_original_videos(ffpp_root, comp)
        if len(orig_videos) == 0:
            print(f"[{comp}] No original videos found - skipping originals.")
        else:
            print(f"[{comp}] Originals: found {len(orig_videos)} videos")
            if not args.dry_run:
                out_dir = output_dir / "original" / comp
                done, total = process_category(
                    name=f"original/{comp}",
                    videos=orig_videos,
                    out_dir=out_dir,
                    is_real=True,
                    args=args,
                )
                grand_done += done
                grand_total += total

        # Manipulations
        for m in methods:
            method_folder = METHOD_MAP[m]
            m_videos = discover_manip_videos(ffpp_root, method_folder, comp)
            if len(m_videos) == 0:
                print(f"[{comp}] {m}({method_folder}): no videos found - skipping.")
                continue
            print(f"[{comp}] {m}({method_folder}): found {len(m_videos)} videos")
            if args.dry_run:
                continue

            out_dir = output_dir / m / comp
            done, total = process_category(
                name=f"{m}/{comp}",
                videos=m_videos,
                out_dir=out_dir,
                is_real=False,
                args=args,
            )
            grand_done += done
            grand_total += total

    if not args.dry_run:
        print(
            f"\nCompleted: processed {grand_done}/{grand_total} videos across categories."
        )
    else:
        print("\nDry run completed.")


if __name__ == "__main__":
    main()
