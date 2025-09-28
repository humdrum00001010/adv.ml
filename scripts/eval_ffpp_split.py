#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ffpp_split.py

Split-aware evaluation for FaceForensics++ using the official splits to report TEST accuracy.

What this script does:
- Loads the official FF++ split files (train/val/test) to select sequences.
- Calibrates the decision threshold (tau) using REAL-ONLY sequences from the VAL split (by default).
- Evaluates the trained model on the TEST split and reports accuracy:
  - Overall (real + fake)
  - Per manipulation method (DF, F2F, FS, FSH, NT, DFD)
  - Per compression (e.g., c23, c40) if multiple are provided

Assumptions:
- Landmarks were extracted with scripts/prepare_ffpp.py into:
    <landmarks_dir>/
      original/<comp>/real/<stem>_landmarks.npy
      <METHOD>/<comp>/fake/<t_s>_landmarks.npy
  where METHOD in [DF, F2F, FS, FSH, NT, DFD] and comp in [c23, c40, ...].

- The official split files live in:
    <splits_dir>/train.json
    <splits_dir>/val.json
    <splits_dir>/test.json
  Each JSON is a list of [target, source] pairs e.g. ["469","481"].
  For REALs, we use both target and source stems (e.g., "469", "481").
  For FAKEs, we use the pair as "469_481" under each manipulated method folder.
  (DeepFakeDetection also follows a similar "target_source" naming in our landmark output.)

- A trained model checkpoint is available (default: models/vae_final.pth).

Usage example:
  python3 adv\ ml/scripts/eval_ffpp_split.py \
    --model_path models/vae_final.pth \
    --landmarks_dir data/landmarks/ffpp \
    --splits_dir external/FaceForensics/dataset/splits \
    --methods DF F2F FS FSH NT DFD \
    --compressions c23 \
    --device auto \
    --tau_source val --tau_quantile 0.80

Notes:
- This script uses the evaluate.evaluate_sample routine for scoring to stay consistent with the main evaluation code.
- TEST accuracy reported here is based on the official test split indices discovered on disk (missing files are skipped).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ----- Repo-local imports (evaluate/model) -----
# Add <repo_root>/scripts to sys.path so we can import evaluate and model.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]  # adv ml/
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from evaluate import evaluate_sample  # type: ignore
from model import LFHFVAE  # type: ignore


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


def which_device(name: str) -> torch.device:
    """Resolve device by preference."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if (
        name == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def load_split_pairs(split_path: Path) -> List[Tuple[str, str]]:
    """
    Load a split JSON file. Each entry is a list/tuple [target, source] of strings.
    Returns a list of (target, source).
    """
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs: List[Tuple[str, str]] = []
    for item in data:
        if isinstance(item, list) and len(item) == 2:
            t, s = item
            pairs.append((str(t), str(s)))
    return pairs


def infer_input_dim(landmarks_dir: Path, compressions: List[str]) -> int:
    """
    Infer input feature dimension from any available real landmark file.
    """
    for comp in compressions:
        real_dir = landmarks_dir / "original" / comp / "real"
        if real_dir.exists():
            for f in sorted(real_dir.glob("*.npy")):
                arr = np.load(str(f))
                if arr.ndim == 3:  # (T, K, 2)
                    return int(arr.shape[1] * arr.shape[2])
                return int(arr.shape[-1])
    # Reasonable default (20 lip points -> 40 dims, but some variants use 25 pts -> 50 dims)
    return 50


def load_model(
    model_path: Path,
    input_dim: int,
    device: torch.device,
    hidden_dim: int = 128,
    lf_latent_dim: int = 32,
    hf_latent_dim: int = 64,
    num_tcn_layers: int = 4,
    tcn_kernel_size: int = 3,
    tcn_dropout: float = 0.1,
) -> LFHFVAE:
    """Load LFHFVAE with saved weights."""
    model = LFHFVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        lf_latent_dim=lf_latent_dim,
        hf_latent_dim=hf_latent_dim,
        num_tcn_layers=num_tcn_layers,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
    ).to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def find_real_npy(landmarks_dir: Path, comp: str, stem: str) -> Optional[Path]:
    """
    Given a youtube ID stem (e.g., "469"), return the path to its real npy, if it exists.
    """
    p = landmarks_dir / "original" / comp / "real" / f"{stem}_landmarks.npy"
    return p if p.exists() else None


def find_fake_npy(
    landmarks_dir: Path, method: str, comp: str, t: str, s: str
) -> Optional[Path]:
    """
    Given a pair (target, source), return path to fake npy "<t>_<s>_landmarks.npy" for a method, if it exists.
    """
    p = landmarks_dir / method / comp / "fake" / f"{t}_{s}_landmarks.npy"
    return p if p.exists() else None


def calibrate_tau_from_val(
    model: LFHFVAE,
    device: torch.device,
    landmarks_dir: Path,
    compressions: List[str],
    val_pairs: List[Tuple[str, str]],
    tau_quantile: float,
    max_reals: int = 512,
) -> float:
    """
    Calibrate tau using REAL-ONLY sequences from the VAL split across given compressions.
    """
    real_paths: List[Path] = []
    # Collect unique stems from pairs (both t and s)
    stems = []
    for t, s in val_pairs:
        stems.append(t)
        stems.append(s)
    # unique while preserving order
    seen = set()
    uniq_stems = [x for x in stems if not (x in seen or seen.add(x))]
    # Gather paths
    for comp in compressions:
        for stem in uniq_stems:
            p = find_real_npy(landmarks_dir, comp, stem)
            if p is not None:
                real_paths.append(p)
            if len(real_paths) >= max_reals:
                break
        if len(real_paths) >= max_reals:
            break

    if not real_paths:
        # Fallback: use a default tau
        return 0.5

    scores: List[float] = []
    for p in real_paths:
        arr = np.load(str(p))
        x = torch.from_numpy(arr.astype(np.float32, copy=False))
        s, _, _, _, _, _, _ = evaluate_sample(model, x, device, tau=0.0)
        scores.append(float(s))
    tau = float(np.quantile(np.array(scores, dtype=np.float32), tau_quantile))
    return tau


def evaluate_split(
    model: LFHFVAE,
    device: torch.device,
    landmarks_dir: Path,
    methods: List[str],
    compressions: List[str],
    test_pairs: List[Tuple[str, str]],
    tau: float,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate on TEST split. Returns a metrics dict:
    {
      'overall': {'acc': float, 'n_real': int, 'n_fake': int, 'n_total': int},
      '<METHOD>': {'acc': float, 'n_real': int, 'n_fake': int, 'n_total': int, 'per_comp': {'c23': float, 'c40': float}}
      ...
    }
    """
    results: Dict[str, Dict[str, float]] = {}
    # Overall counters
    all_labels: List[int] = []
    all_preds: List[int] = []

    # Evaluate per method
    for method in methods:
        method_labels: List[int] = []
        method_preds: List[int] = []
        per_comp: Dict[str, Tuple[int, int]] = {}  # comp -> (correct, total)

        for comp in compressions:
            # REALS for this comp: include both target and source stems from test pairs
            real_paths: List[Path] = []
            seen_real = set()
            for t, s in test_pairs:
                for stem in (t, s):
                    if (comp, stem) in seen_real:
                        continue
                    p = find_real_npy(landmarks_dir, comp, stem)
                    if p is not None:
                        real_paths.append(p)
                        seen_real.add((comp, stem))

            # FAKES for this comp: for the given method, build t_s
            fake_paths: List[Path] = []
            for t, s in test_pairs:
                p = find_fake_npy(landmarks_dir, method, comp, t, s)
                if p is not None:
                    fake_paths.append(p)

            # Score REALS
            for p in real_paths:
                arr = np.load(str(p))
                x = torch.from_numpy(arr.astype(np.float32, copy=False))
                s, _, _, _, _, _, _ = evaluate_sample(model, x, device, tau=0.0)
                pred = int(s > tau)  # 1 = live
                method_labels.append(1)
                method_preds.append(pred)

            # Score FAKES
            for p in fake_paths:
                arr = np.load(str(p))
                x = torch.from_numpy(arr.astype(np.float32, copy=False))
                s, _, _, _, _, _, _ = evaluate_sample(model, x, device, tau=0.0)
                pred = int(s > tau)  # 1 = live
                method_labels.append(0)
                method_preds.append(pred)

            # Per-comp accuracy (if any samples)
            if method_labels and method_preds:
                # Filter method samples for this comp only for an accurate per-comp metric:
                # Recompute quickly: we just measured sequentially, so we need comp-specific counters.
                # For simplicity, compute comp-acc locally now:
                comp_labels: List[int] = []
                comp_preds: List[int] = []
                # re-score per comp only (to avoid indexing complexity)
                comp_labels.extend([1] * len(real_paths))
                comp_preds.extend(
                    [
                        int(
                            evaluate_sample(
                                model,
                                torch.from_numpy(
                                    np.load(str(p)).astype(np.float32, copy=False)
                                ),
                                device,
                                tau=0.0,
                            )[0]
                            > tau
                        )
                        for p in real_paths
                    ]
                )
                comp_labels.extend([0] * len(fake_paths))
                comp_preds.extend(
                    [
                        int(
                            evaluate_sample(
                                model,
                                torch.from_numpy(
                                    np.load(str(p)).astype(np.float32, copy=False)
                                ),
                                device,
                                tau=0.0,
                            )[0]
                            > tau
                        )
                        for p in fake_paths
                    ]
                )
                if comp_labels:
                    comp_correct = int(
                        np.sum(
                            (np.array(comp_labels) == np.array(comp_preds)).astype(
                                np.int32
                            )
                        )
                    )
                    per_comp[comp] = (comp_correct, len(comp_labels))

        # Aggregate method accuracy
        if method_labels:
            correct = int(
                np.sum(
                    (np.array(method_labels) == np.array(method_preds)).astype(np.int32)
                )
            )
            acc = correct / max(1, len(method_labels))
            results[method] = {
                "acc": float(acc),
                "n_real": int(np.sum(np.array(method_labels) == 1)),
                "n_fake": int(np.sum(np.array(method_labels) == 0)),
                "n_total": int(len(method_labels)),
            }
            # Add per-comp breakdown
            if per_comp:
                results[method]["per_comp"] = {
                    c: (float(correct) / max(1, total))
                    for c, (correct, total) in per_comp.items()
                }

            # Update overall
            all_labels.extend(method_labels)
            all_preds.extend(method_preds)
        else:
            results[method] = {
                "acc": float("nan"),
                "n_real": 0,
                "n_fake": 0,
                "n_total": 0,
            }

    # Overall
    if all_labels:
        correct = int(
            np.sum((np.array(all_labels) == np.array(all_preds)).astype(np.int32))
        )
        results["overall"] = {
            "acc": float(correct / max(1, len(all_labels))),
            "n_real": int(np.sum(np.array(all_labels) == 1)),
            "n_fake": int(np.sum(np.array(all_labels) == 0)),
            "n_total": int(len(all_labels)),
        }
    else:
        results["overall"] = {
            "acc": float("nan"),
            "n_real": 0,
            "n_fake": 0,
            "n_total": 0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FF++ TEST accuracy using official splits."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(_REPO_ROOT / "models" / "vae_final.pth"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--landmarks_dir",
        type=str,
        default=str(_REPO_ROOT / "data" / "landmarks" / "ffpp"),
        help="Directory containing extracted landmark .npy files",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=str(_REPO_ROOT / "external" / "FaceForensics" / "dataset" / "splits"),
        help="Directory with FF++ split JSON files (train/val/test)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=f"Manipulations to evaluate (default: {DEFAULT_METHODS})",
    )
    parser.add_argument(
        "--compressions",
        nargs="+",
        default=DEFAULT_COMPRESSIONS,
        help=f"Compressions to include (default: {DEFAULT_COMPRESSIONS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use",
    )
    # Model hyperparameters (to evaluate deeper/wider models)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Model hidden dimension (TCN/GRU width).",
    )
    parser.add_argument(
        "--lf_latent_dim",
        type=int,
        default=32,
        help="LF branch latent dimension.",
    )
    parser.add_argument(
        "--hf_latent_dim",
        type=int,
        default=64,
        help="HF branch latent dimension.",
    )
    parser.add_argument(
        "--num_tcn_layers",
        type=int,
        default=4,
        help="Number of TCN layers in encoder.",
    )
    parser.add_argument(
        "--tcn_kernel_size",
        type=int,
        default=3,
        help="TCN kernel size.",
    )
    parser.add_argument(
        "--tcn_dropout",
        type=float,
        default=0.1,
        help="TCN dropout.",
    )
    parser.add_argument(
        "--tau_source",
        type=str,
        default="val",
        choices=["val", "test_real", "fixed"],
        help="Where to calibrate tau from: val (recommended), test_real (not recommended), or fixed.",
    )
    parser.add_argument(
        "--tau_quantile",
        type=float,
        default=0.80,
        help="Quantile on real-only scores to set tau (when tau_source is val or test_real).",
    )
    parser.add_argument(
        "--tau_fixed",
        type=float,
        default=0.5,
        help="Fixed tau to use if --tau_source fixed",
    )
    parser.add_argument(
        "--max_val_reals",
        type=int,
        default=512,
        help="Max real sequences from VAL to use for tau calibration.",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default=None,
        help="Optional path to save a JSON report with per-method and overall metrics.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    landmarks_dir = Path(args.landmarks_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    device = which_device(args.device)
    print(f"[DEVICE] Using: {device.type}")

    # Load splits
    test_pairs = load_split_pairs(splits_dir / "test.json")
    val_pairs = load_split_pairs(splits_dir / "val.json")

    # Build/Load model
    input_dim = infer_input_dim(landmarks_dir, args.compressions)
    model = load_model(
        model_path,
        input_dim=input_dim,
        device=device,
        hidden_dim=args.hidden_dim,
        lf_latent_dim=args.lf_latent_dim,
        hf_latent_dim=args.hf_latent_dim,
        num_tcn_layers=args.num_tcn_layers,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_dropout=args.tcn_dropout,
    )

    # Calibrate tau
    if args.tau_source == "val":
        tau = calibrate_tau_from_val(
            model,
            device,
            landmarks_dir,
            compressions=args.compressions,
            val_pairs=val_pairs,
            tau_quantile=args.tau_quantile,
            max_reals=args.max_val_reals,
        )
        print(
            f"[TAU] Using VAL split (real-only) q={args.tau_quantile:.2f} => tau={tau:.6f}"
        )
    elif args.tau_source == "test_real":
        tau = calibrate_tau_from_val(
            model,
            device,
            landmarks_dir,
            compressions=args.compressions,
            val_pairs=test_pairs,  # use test pairs' reals (not recommended)
            tau_quantile=args.tau_quantile,
            max_reals=args.max_val_reals,
        )
        print(
            f"[TAU] Using TEST split reals (q={args.tau_quantile:.2f}) => tau={tau:.6f}"
        )
    else:
        tau = float(args.tau_fixed)
        print(f"[TAU] Using fixed tau={tau:.6f}")

    # Evaluate
    results = evaluate_split(
        model=model,
        device=device,
        landmarks_dir=landmarks_dir,
        methods=args.methods,
        compressions=args.compressions,
        test_pairs=test_pairs,
        tau=tau,
    )

    # Print summary
    print("\n=== FF++ TEST Accuracy (split-aware) ===")
    if "overall" in results:
        o = results["overall"]
        print(
            f"OVERALL: acc={o['acc']:.4f} (n_total={int(o['n_total'])}, n_real={int(o['n_real'])}, n_fake={int(o['n_fake'])})"
        )
    for m in args.methods:
        if m in results:
            r = results[m]
            acc = r.get("acc", float("nan"))
            n_total = int(r.get("n_total", 0))
            n_real = int(r.get("n_real", 0))
            n_fake = int(r.get("n_fake", 0))
            print(
                f"{m:4s}: acc={acc:.4f} (n_total={n_total}, n_real={n_real}, n_fake={n_fake})"
            )
            if "per_comp" in r:
                per_comp = r["per_comp"]
                pcs = ", ".join(
                    [f"{c}:{per_comp[c]:.4f}" for c in sorted(per_comp.keys())]
                )
                print(f"      per_comp: {pcs}")

    # Save report if requested
    if args.report_json:
        out_path = Path(args.report_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_path": str(model_path),
                    "landmarks_dir": str(landmarks_dir),
                    "methods": args.methods,
                    "compressions": args.compressions,
                    "tau_source": args.tau_source,
                    "tau_quantile": args.tau_quantile
                    if args.tau_source != "fixed"
                    else None,
                    "tau": tau,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\n[REPORT] Saved to {out_path}")


if __name__ == "__main__":
    main()
