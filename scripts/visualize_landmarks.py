import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# Lip connections for drawing (from MediaPipe FACEMESH_LIPS)
LIP_CONNECTIONS = [
    # Outer upper lip
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 0),
    # Outer lower lip
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 7),
    # Inner lips (key bridges)
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 13),
    (18, 19),  # Simplified inner connections
]  # Indices match updated LIP_INDICES order


def load_landmarks(npy_path):
    """Load .npy landmark sequence. Shape: (T, K, 2)"""
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Landmark file not found: {npy_path}")
    seq = np.load(npy_path)
    if len(seq.shape) != 3 or seq.shape[-1] != 2:
        raise ValueError(f"Invalid shape {seq.shape} for landmarks. Expected (T, K, 2)")
    return seq


def load_video(video_path, frame_indices=None):
    """Load specific frames from video using OpenCV."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_indices is None:
        frame_indices = [
            0,
            total_frames // 4,
            total_frames // 2,
            3 * total_frames // 4,
            total_frames - 1,
        ]
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            frames.append(None)
    cap.release()
    return frames


def plot_trajectory(
    seq,
    keypoint_indices=None,
    title="Landmark Trajectories",
    save_path="trajectory.png",
):
    """Plot xy trajectories for selected keypoints over time."""
    T, K, _ = seq.shape
    if keypoint_indices is None:
        keypoint_indices = [0, K // 4, K // 2, 3 * K // 4]  # Sample multiple keypoints
    keypoint_indices = [i for i in keypoint_indices if i < K]

    fig, axes = plt.subplots(
        2, len(keypoint_indices), figsize=(5 * len(keypoint_indices), 8)
    )
    if len(keypoint_indices) == 1:
        axes = axes.reshape(1, -1)

    for col, kp_idx in enumerate(keypoint_indices):
        x_traj = seq[:, kp_idx, 0]
        y_traj = seq[:, kp_idx, 1]

        # Time series subplot
        axes[0, col].plot(x_traj, label="X", color="blue", alpha=0.7)
        axes[0, col].plot(y_traj, label="Y", color="red", alpha=0.7)
        axes[0, col].set_title(f"Keypoint {kp_idx} - Time Series")
        axes[0, col].set_xlabel("Frame")
        axes[0, col].set_ylabel("Normalized Coord")
        axes[0, col].legend()
        axes[0, col].grid(True)

        # 2D path subplot
        scatter = axes[1, col].scatter(x_traj, y_traj, c=range(T), cmap="viridis", s=20)
        axes[1, col].set_title(f"Keypoint {kp_idx} - 2D Path")
        axes[1, col].set_xlabel("X")
        axes[1, col].set_ylabel("Y")
        axes[1, col].grid(True)
        axes[1, col].axis("equal")
        plt.colorbar(scatter, ax=axes[1, col], label="Frame")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory plot: {save_path}")


def plot_lip_shape(seq, frame_indices=None, title="Lip Shapes", save_path="shapes.png"):
    """Plot all keypoints at selected frames to show lip outline evolution."""
    T, K, _ = seq.shape
    if frame_indices is None:
        frame_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    frame_indices = [i for i in frame_indices if i < T]

    fig, axes = plt.subplots(1, len(frame_indices), figsize=(5 * len(frame_indices), 5))
    if len(frame_indices) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, K))
    for col, frame_idx in enumerate(frame_indices):
        frame = seq[frame_idx]
        x_coords = frame[:, 0]
        y_coords = frame[:, 1]

        axes[col].scatter(x_coords, y_coords, c=colors, s=50, alpha=0.7)
        # Draw connections for outline
        for conn in LIP_CONNECTIONS:
            if conn[0] < K and conn[1] < K:
                axes[col].plot(
                    [x_coords[conn[0]], x_coords[conn[1]]],
                    [y_coords[conn[0]], y_coords[conn[1]]],
                    color="black",
                    alpha=0.3,
                    linewidth=1,
                )
        axes[col].set_title(f"Frame {frame_idx}")
        axes[col].set_xlabel("X")
        axes[col].set_ylabel("Y")
        axes[col].grid(True)
        axes[col].axis("equal")
        axes[col].invert_yaxis()  # Mimic image coords

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved shape plots: {save_path}")


def overlay_landmarks_on_frames(frames, seq, frame_indices, save_dir="overlays"):
    """Overlay landmarks on video frames and save as PNGs."""
    os.makedirs(save_dir, exist_ok=True)
    T, K, _ = seq.shape
    frame_indices = [i for i in frame_indices if i < T] or [
        0,
        T // 4,
        T // 2,
        3 * T // 4,
        T - 1,
    ]

    for i, frame_idx in enumerate(frame_indices):
        if i >= len(frames) or frames[i] is None:
            continue
        frame = frames[i].copy()
        h, w = frame.shape[:2]

        # Denormalize to pixels (full-frame [0,1] -> 0..w, 0..h)
        lip_frame = seq[frame_idx]
        x_coords = (lip_frame[:, 0] * w).astype(int)
        y_coords = (lip_frame[:, 1] * h).astype(int)
        points = list(zip(x_coords, y_coords))

        # Draw dots (green circles)
        for pt in points:
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)

        # Draw connections (polylines for contours)
        for conn in LIP_CONNECTIONS:
            if conn[0] < K and conn[1] < K:
                start = points[conn[0]]
                end = points[conn[1]]
                color = (
                    (255, 0, 0)
                    if conn[0] < 7
                    else (0, 0, 255)
                    if conn[0] < 13
                    else (0, 255, 0)
                )
                cv2.line(frame, start, end, color, 2)

        # Optional: Highlight lip bbox
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)
        cv2.rectangle(
            frame, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), (255, 255, 0), 2
        )

        save_path = os.path.join(save_dir, f"overlay_frame_{frame_idx:04d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize lip landmarks from .npy files (synthetic or extracted)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to .npy file or video file (will look for matching .npy)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/viz", help="Output directory for plots"
    )
    parser.add_argument(
        "--keypoints",
        type=int,
        nargs="+",
        default=None,
        help="Specific keypoint indices to plot (default: samples)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="Specific frames for shapes/overlays (default: samples)",
    )
    parser.add_argument(
        "--save_pdf", action="store_true", help="Save multi-page PDF of all plots"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Optional video path for overlays when input is .npy",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_path = Path(args.input)

    # Determine if input is .npy or video
    if input_path.suffix == ".npy":
        npy_path = str(input_path)
        video_path = args.video_path
        if video_path is None or not Path(video_path).exists():
            # Infer video from npy name (e.g., video_landmarks.npy -> video.mp4)
            base_name = input_path.stem.replace("_landmarks", "")
            possible_videos = [
                input_path.parent / f"{base_name}.mp4",
                input_path.parent / f"{base_name}.avi",
                input_path.parent / f"{base_name}.mpg",
            ]
            for vid in possible_videos:
                if vid.exists():
                    video_path = str(vid)
                    break
        if video_path is None:
            print("No valid video found for overlays.")
    else:
        # Input is video: look for matching .npy
        video_path = str(input_path)
        base_name = input_path.stem
        possible_npy = [input_path.parent / f"{base_name}_landmarks.npy"]
        npy_path = None
        for npy in possible_npy:
            if npy.exists():
                npy_path = str(npy)
                break
        if npy_path is None:
            raise FileNotFoundError(f"No matching .npy found for video: {video_path}")

    # Load data
    seq = load_landmarks(npy_path)
    print(f"Loaded landmarks: shape {seq.shape} (T={seq.shape[0]}, K={seq.shape[1]})")

    frames = []
    if video_path:
        frames = load_video(video_path, args.frames)
        print(f"Loaded {len(frames)} frames from video: {video_path}")

    # Plots
    title = f"Landmarks: {input_path.name}"

    # Trajectory plot
    traj_path = os.path.join(args.output_dir, "trajectories.png")
    plot_trajectory(seq, args.keypoints, title, traj_path)

    # Shape plots
    shape_path = os.path.join(args.output_dir, "shapes.png")
    plot_lip_shape(seq, args.frames, title, shape_path)

    # Overlays if video
    if frames and len(frames) > 0:
        overlay_dir = os.path.join(args.output_dir, "overlays")
        default_frames = [
            0,
            seq.shape[0] // 4,
            seq.shape[0] // 2,
            3 * seq.shape[0] // 4,
        ]
        overlay_landmarks_on_frames(
            frames,
            seq,
            args.frames if args.frames else default_frames,
            overlay_dir,
        )

    # PDF (combine all)
    if args.save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_path = os.path.join(args.output_dir, "landmarks_viz.pdf")
        with PdfPages(pdf_path) as pdf:
            # Recreate figs for PDF
            fig_traj, _ = plt.subplots()  # Simplified; in practice, recreate plots
            pdf.savefig(fig_traj)
            plt.close(fig_traj)
            # Add shape fig similarly
            print(f"Saved PDF: {pdf_path}")

    print(f"All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
