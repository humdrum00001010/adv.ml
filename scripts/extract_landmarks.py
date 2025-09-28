import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import glob
import os

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Lip landmark indices (subset of 20 keypoints from MediaPipe FaceMesh for lips: outer/inner contours)
# Upper lip: 13, 14, 15, 61, 78, 81, 82, 185
# Lower lip: 291, 308, 324, 375, 321, 405
# Inner: 78, 95, 88, 178, 87, 14 (approx 20 total, focused on lip region)
LIP_INDICES = [
    13,
    14,
    15,
    61,
    78,
    81,
    82,
    185,
    291,
    308,
    324,
    375,
    321,
    405,
    78,
    95,
    88,
    178,
    87,
    14,
]


def extract_lip_landmarks(
    video_path: str, target_frames: int = 64, alpha_ema: float = 0.3
) -> np.ndarray:
    """
    Extract lip landmarks from video.
    - Sample target_frames at ~25fps.
    - Normalize xy to [0,1] relative to frame size.
    - Apply light EMA smoothing.
    Returns: np.array of shape (T, K, 2)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Subsample to target_frames (uniform sampling)
    frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

    landmarks_seq = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # For better lip detection
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape

                lip_points = []
                for idx in LIP_INDICES:
                    lm = face_landmarks.landmark[idx]
                    # Normalize to [0,1]
                    x = lm.x  # Already normalized by MediaPipe, but confirm
                    y = lm.y
                    lip_points.append([x, y])

                lip_points = np.array(lip_points)  # (K, 2)
                landmarks_seq.append(lip_points)
            else:
                # If no face, pad with zeros or previous (for demo, pad zeros)
                print(f"No face detected in frame {i}")
                lip_points = np.zeros((len(LIP_INDICES), 2))
                landmarks_seq.append(lip_points)

    cap.release()

    if len(landmarks_seq) < target_frames:
        # Pad if video too short
        padding = np.zeros((target_frames - len(landmarks_seq), len(LIP_INDICES), 2))
        landmarks_seq = np.vstack([landmarks_seq, padding])
    else:
        landmarks_seq = landmarks_seq[:target_frames]

    landmarks_array = np.array(landmarks_seq)  # (T, K, 2)

    # Light EMA smoothing along time axis (per coordinate)
    for k in range(landmarks_array.shape[1]):
        for c in range(2):  # x and y
            smoothed = np.zeros_like(landmarks_array[:, k, c])
            smoothed[0] = landmarks_array[0, k, c]
            for t in range(1, landmarks_array.shape[0]):
                smoothed[t] = (
                    alpha_ema * landmarks_array[t, k, c]
                    + (1 - alpha_ema) * smoothed[t - 1]
                )
            landmarks_array[:, k, c] = smoothed

    return landmarks_array


def plot_landmark_trajectories(landmarks: np.ndarray, save_path: str = None):
    """Plot trajectories of selected lip keypoints over time."""
    T, K, _ = landmarks.shape
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    # Plot first 4 keypoints as example (e.g., lip corners, center)
    keypoint_ids = [0, 1, 2, 3]  # Indices in LIP_INDICES
    colors = ["r", "g", "b", "orange"]

    for i, kid in enumerate(keypoint_ids):
        ax = axes[i]
        x_traj = landmarks[:, kid, 0]
        y_traj = landmarks[:, kid, 1]

        ax.plot(x_traj, label=f"X (kp {kid})", color=colors[i])
        ax.plot(y_traj, label=f"Y (kp {kid})", color=colors[i], linestyle="--")
        ax.set_title(f"Keypoint {LIP_INDICES[kid]} Trajectory")
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Normalized Position")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Batch processing for first 5 videos in GRID dataset
    video_dir = "../data/grid/data/"
    output_dir = "../data/landmarks/"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = "../data/batch_landmark_trajectories.png"

    # Find all MP4 files recursively, take first 5
    mp4_files = sorted(
        glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
    )[:5]
    print(f"Found {len(mp4_files)} videos to process: {mp4_files}")

    all_landmarks = []
    batch_stats = {"mean_std_x": [], "mean_std_y": []}

    for i, video_path in enumerate(mp4_files):
        print(f"\nProcessing video {i + 1}: {os.path.basename(video_path)}")
        landmarks = extract_lip_landmarks(video_path)
        print(f"Extracted shape: {landmarks.shape}")

        # Save per-file npy
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_npy = os.path.join(output_dir, f"landmarks_{base_name}.npy")
        np.save(output_npy, landmarks)
        print(f"Saved to {output_npy}")

        all_landmarks.append(landmarks)

        # Compute per-sequence stats (e.g., std of positions for movement)
        std_x = np.std(landmarks[:, :, 0])
        std_y = np.std(landmarks[:, :, 1])
        batch_stats["mean_std_x"].append(std_x)
        batch_stats["mean_std_y"].append(std_y)

    # Batch summary stats
    if all_landmarks:
        avg_shape = np.mean([lm.shape for lm in all_landmarks], axis=0)
        print(f"\nBatch summary:")
        print(f"Average sequence length: {int(avg_shape[0])} frames")
        print(f"Average movement (std x): {np.mean(batch_stats['mean_std_x']):.4f}")
        print(f"Average movement (std y): {np.mean(batch_stats['mean_std_y']):.4f}")

        # Plot example from first video
        example_landmarks = all_landmarks[0]
        plot_landmark_trajectories(example_landmarks, save_path=plot_path)
        print(f"Example plot saved to {plot_path}")
    else:
        print("No videos processed.")
