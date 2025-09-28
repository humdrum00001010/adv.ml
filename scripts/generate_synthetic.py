import numpy as np
import os
import argparse
from pathlib import Path


def generate_real_sequence(T=64, K=25, freq_range=(0.1, 0.5), noise_std=0.01):
    """
    Generate a synthetic 'real' lip landmark sequence: smooth sinusoidal motions
    mimicking natural mouth movements (e.g., opening/closing, slight vibrations).
    Shape: (T, K, 2) for time, keypoints, xy coords (normalized [0,1]).
    """
    t = np.linspace(0, 2 * np.pi, T)
    x_landmarks = np.zeros((T, K, 2))

    for k in range(K):
        # Base position: lips form an oval-ish shape, varying per keypoint
        base_y = 0.5 + 0.1 * np.sin(np.pi * k / (K - 1))  # Upper/lower lip curve
        base_x = 0.5 + 0.15 * (k / K - 0.5)  # Horizontal spread

        # Smooth motions: low-freq sinusoids for trajectory
        freq = np.random.uniform(*freq_range)
        amp_x = np.random.uniform(0.05, 0.1)
        amp_y = np.random.uniform(0.08, 0.15)  # More y-motion for opening

        x_landmarks[:, k, 0] = base_x + amp_x * np.sin(freq * t)  # Horizontal sway
        x_landmarks[:, k, 1] = base_y + amp_y * np.sin(
            freq * t + np.pi / 4
        )  # Vertical open/close

        # Light noise for natural variation
        x_landmarks[:, k, :] += np.random.normal(0, noise_std, (T, 2))

    # Normalize to [0,1] per frame (simple bbox norm)
    for frame in range(T):
        min_coords = x_landmarks[frame].min(axis=0)
        max_coords = x_landmarks[frame].max(axis=0)
        x_landmarks[frame] = (x_landmarks[frame] - min_coords) / (
            max_coords - min_coords + 1e-6
        )

    # Apply light EMA smoothing to ensure smoothness
    alpha = 0.3
    smoothed = np.zeros_like(x_landmarks)
    smoothed[0] = x_landmarks[0]
    for t in range(1, T):
        smoothed[t] = alpha * x_landmarks[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def generate_fake_sequence(T=64, K=25, jerk_prob=0.3, noise_std=0.05):
    """
    Generate a synthetic 'fake' sequence: jerky, noisy, with discontinuities
    to simulate deep fake artifacts (e.g., abrupt changes, high-freq noise).
    """
    # Start with a real-like base, then corrupt
    seq = generate_real_sequence(T, K)

    # Add jerks: random frame shifts/discontinuities
    for t in range(1, T):
        if np.random.rand() < jerk_prob:
            # Abrupt shift in some keypoints
            shift_k = np.random.choice(K, size=np.random.randint(5, 15))
            seq[t, shift_k, :] += np.random.uniform(-0.1, 0.1, (len(shift_k), 2))

    # Higher noise
    seq += np.random.normal(0, noise_std, seq.shape)

    # Occasional frame drops/duplicates (simulate timing issues)
    for _ in range(int(T * 0.1)):  # 10% perturbations
        drop_t = np.random.randint(1, T)
        seq[drop_t] = seq[drop_t - 1] + np.random.normal(
            0, 0.03, (K, 2)
        )  # Near-duplicate with noise

    # Re-normalize
    for frame in range(T):
        min_coords = seq[frame].min(axis=0)
        max_coords = seq[frame].max(axis=0)
        seq[frame] = (seq[frame] - min_coords) / (max_coords - min_coords + 1e-6)

    return seq


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic lip landmark data for prototyping."
    )
    parser.add_argument(
        "--num_real", type=int, default=100, help="Number of real sequences to generate"
    )
    parser.add_argument(
        "--num_fake", type=int, default=50, help="Number of fake sequences to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/landmarks/synthetic",
        help="Output directory",
    )
    parser.add_argument("--T", type=int, default=64, help="Sequence length")
    parser.add_argument("--K", type=int, default=25, help="Number of lip keypoints")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    real_dir = Path(args.output_dir) / "real"
    fake_dir = Path(args.output_dir) / "fake"
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)

    print(f"Generating {args.num_real} real sequences...")
    for i in range(args.num_real):
        seq = generate_real_sequence(args.T, args.K)
        np.save(real_dir / f"real_{i:03d}.npy", seq)

    print(f"Generating {args.num_fake} fake sequences...")
    for i in range(args.num_fake):
        seq = generate_fake_sequence(args.T, args.K)
        np.save(fake_dir / f"fake_{i:03d}.npy", seq)

    print(
        f"Synthetic data saved to {args.output_dir}/real/ and {args.output_dir}/fake/"
    )


if __name__ == "__main__":
    main()
