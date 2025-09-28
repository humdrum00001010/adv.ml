import os
import glob
import argparse
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import json

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Lip landmark indices (20 ordered points from FACEMESH_LIPS for proper contours)
LIP_INDICES = [
    # Outer upper lip (clockwise from left)
    61,
    291,
    0,
    37,
    39,
    40,
    185,
    # Outer lower lip (continuing clockwise)
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    # Inner upper/lower (key points for detail)
    78,
    95,
    88,
    178,
    87,
    317,
    402,
    318,
]  # Total 20; ordered for polyline drawing
LIP_INDICES = LIP_INDICES[:20]  # Ensure 20


def extract_lip_landmarks_from_video(
    video_path,
    target_fps=25,
    min_frames=48,
    smoothing_alpha=0.3,
    seq_len=64,
    normalize_lip_relative=False,
):
    """
    Extract lip landmarks from a single video.

    Args:
        video_path (str): Path to video file.
        target_fps (int): Target FPS for downsampling.
        min_frames (int): Minimum frames to process.
        smoothing_alpha (float): EMA smoothing factor [0,1].
        seq_len (int): Pad/truncate to this length.

    Returns:
        np.array: (T, 25, 2) landmarks or None if invalid.
        dict: Metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None, None

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps

    if total_frames < min_frames:
        print(
            f"Skipping {video_path}: too short ({total_frames} < {min_frames} frames)"
        )
        cap.release()
        return None, None

    # Compute downsampled frame indices (uniform sampling to target_fps)
    step = max(1, int(original_fps / target_fps))
    frame_indices = list(range(0, total_frames, step))
    T_raw = len(frame_indices)
    if T_raw < min_frames:
        print(f"Skipping {video_path}: too short after downsampling ({T_raw} frames)")
        cap.release()
        return None, None

    # Limit to seq_len if longer
    if T_raw > seq_len:
        frame_indices = frame_indices[:seq_len]
        T_raw = seq_len

    landmarks_seq = []
    face_detected = 0
    lip_conf_avg = 0.0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as face_mesh:
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip frame horizontally for selfie view (optional)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                lip_points = []

                # Extract lip landmarks
                for idx in LIP_INDICES:
                    landmark = face_landmarks.landmark[idx]
                    # Normalize to [0,1] using image dimensions (simple; bbox better but approx)
                    x = landmark.x  # Already normalized by MediaPipe
                    y = landmark.y
                    lip_points.append([x, y])

                lip_points = np.array(lip_points)

                # Optional: Normalize per frame using lip bbox (more stable)
                if normalize_lip_relative and len(lip_points) > 0:
                    min_coords = lip_points.min(axis=0)
                    max_coords = lip_points.max(axis=0)
                    range_coords = max_coords - min_coords + 1e-6
                    lip_points = (lip_points - min_coords) / range_coords

                landmarks_seq.append(lip_points)
                face_detected += 1
                lip_conf_avg += np.mean(
                    [
                        face_landmarks.landmark[idx].presence
                        for idx in LIP_INDICES
                        if hasattr(face_landmarks.landmark[idx], "presence")
                    ]
                )

            else:
                # No face: pad with last or zeros
                if landmarks_seq:
                    landmarks_seq.append(landmarks_seq[-1])
                else:
                    landmarks_seq.append(np.zeros((25, 2)))

    cap.release()

    if len(landmarks_seq) == 0:
        return None, None

    seq = np.array(landmarks_seq)  # (T_raw, 25, 2)

    # EMA smoothing along time axis
    if smoothing_alpha > 0:
        smoothed = np.zeros_like(seq)
        smoothed[0] = seq[0]
        for t in range(1, T_raw):
            smoothed[t] = (
                smoothing_alpha * seq[t] + (1 - smoothing_alpha) * smoothed[t - 1]
            )
        seq = smoothed

    # Pad or truncate to seq_len
    if T_raw < seq_len:
        pad_len = seq_len - T_raw
        pad = np.tile(seq[-1:], (pad_len, 1, 1))
        seq = np.concatenate([seq, pad], axis=0)
    else:
        seq = seq[:seq_len]

    # Metadata
    metadata = {
        "video_path": video_path,
        "original_fps": original_fps,
        "total_frames": total_frames,
        "duration_s": duration,
        "extracted_frames": T_raw,
        "face_detected_rate": face_detected / len(frame_indices),
        "lip_conf_avg": lip_conf_avg / face_detected if face_detected > 0 else 0.0,
    }

    return seq, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract lip landmarks from videos using MediaPipe."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory with videos"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for .npy"
    )
    parser.add_argument(
        "--label_real",
        type=bool,
        default=True,
        help="If true, save to 'real' subdir; else 'fake'",
    )
    parser.add_argument(
        "--smoothing_alpha", type=float, default=0.3, help="EMA smoothing factor [0,1]"
    )
    parser.add_argument(
        "--target_fps", type=int, default=25, help="Target FPS for downsampling"
    )
    parser.add_argument(
        "--min_frames", type=int, default=48, help="Minimum frames to process"
    )
    parser.add_argument(
        "--seq_len", type=int, default=64, help="Sequence length (pad/truncate)"
    )
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument(
        "--max_videos", type=int, default=None, help="Max videos to process"
    )
    parser.add_argument(
        "--single_video", type=str, default=None, help="Process single video file"
    )
    parser.add_argument(
        "--normalize_lip_relative",
        action="store_true",
        help="Normalize to lip bbox [0,1] (for model invariance; disables easy overlays)",
    )

    args = parser.parse_args()

    # Create output dir
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    label_dir = output_base / ("real" if args.label_real else "fake")
    label_dir.mkdir(exist_ok=True)

    # Find videos
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.mpg"]
    video_paths = []
    if args.single_video:
        video_paths = [args.single_video]
    else:
        for ext in video_extensions:
            if args.recursive:
                video_paths.extend(
                    glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True)
                )
            else:
                video_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    video_paths = [p for p in video_paths if Path(p).is_file()]
    if args.max_videos:
        video_paths = video_paths[: args.max_videos]

    print(f"Found {len(video_paths)} videos to process.")

    processed = 0
    for video_path in video_paths:
        print(f"Processing {video_path}...")
        seq, metadata = extract_lip_landmarks_from_video(
            video_path,
            args.target_fps,
            args.min_frames,
            args.smoothing_alpha,
            args.seq_len,
            args.normalize_lip_relative,
        )

        if seq is not None:
            base_name = Path(video_path).stem
            npy_path = label_dir / f"{base_name}_landmarks.npy"
            np.save(npy_path, seq)

            # Save metadata
            meta_path = label_dir / f"{base_name}_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved: {npy_path} (shape: {seq.shape})")
            processed += 1
        else:
            print(f"Skipped {video_path}")

    print(f"Processed {processed}/{len(video_paths)} videos.")


if __name__ == "__main__":
    main()
