import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
import warnings

warnings.filterwarnings("ignore")


class LipLandmarkDataset(Dataset):
    def __init__(
        self,
        data_dir,
        max_length=64,
        smoothing_alpha=0.3,
        lpf_cutoff=5.0,
        fps=25.0,
        is_real=True,
    ):
        """
        PyTorch Dataset for lip landmark sequences.

        Args:
            data_dir (str): Path to directory containing .npy files (e.g., 'data/landmarks/grid').
            max_length (int): Fixed sequence length (T=64).
            smoothing_alpha (float): EMA smoothing factor (0.0 = no smoothing).
            lpf_cutoff (float): Low-pass filter cutoff frequency in Hz.
            fps (float): Assumed frames per second for filter design.
            is_real (bool): If True, treat all as real (label=1); else fake (label=0). For mixed dirs, override.
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.smoothing_alpha = smoothing_alpha
        self.lpf_cutoff = lpf_cutoff
        self.fps = fps
        self.is_real = is_real

        # Find all .npy files
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.file_paths:
            raise ValueError(f"No .npy files found in {data_dir}")

        # Design LPF (Butterworth, 4th order)
        nyquist = fps / 2.0
        normal_cutoff = lpf_cutoff / nyquist
        self.b, self.a = butter(4, normal_cutoff, btype="low", analog=False)

    def _apply_ema_smoothing(self, seq):
        """Apply EMA smoothing along time axis."""
        if self.smoothing_alpha == 0.0:
            return seq
        smoothed = np.copy(seq)
        for k in range(seq.shape[1]):
            for d in range(seq.shape[2]):
                smoothed[:, k, d] = self._ema_filter(seq[:, k, d], self.smoothing_alpha)
        return smoothed

    def _ema_filter(self, data, alpha):
        """1D EMA filter."""
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * data[i]
        return smoothed

    def _apply_lpf(self, seq):
        """Apply low-pass filter to get LF target."""
        lpf_seq = np.copy(seq)
        for k in range(seq.shape[1]):
            for d in range(seq.shape[2]):
                lpf_seq[:, k, d] = filtfilt(self.b, self.a, seq[:, k, d])
        return lpf_seq

    def _pad_or_truncate(self, seq):
        """Pad to max_length by repeating last frame or truncate."""
        T = seq.shape[0]
        if T < self.max_length:
            pad_len = self.max_length - T
            pad = np.tile(seq[-1:], (pad_len, 1, 1))
            seq = np.concatenate([seq, pad], axis=0)
        elif T > self.max_length:
            seq = seq[: self.max_length]
        return seq

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load .npy
        file_path = self.file_paths[idx]
        seq = np.load(file_path)  # Shape: (T, K, 2)

        # Assume seq is already normalized [0,1]; if not, normalize here if needed

        # Apply smoothing
        seq = self._apply_ema_smoothing(seq)

        # Pad/truncate
        seq = self._pad_or_truncate(seq)

        # Compute LF target
        x_lf = self._apply_lpf(seq)

        # HF residual target
        r_hf = seq - x_lf

        # To tensors
        x = torch.from_numpy(seq).float()  # (T, K, 2)
        x_lf = torch.from_numpy(x_lf).float()
        r_hf = torch.from_numpy(r_hf).float()

        # Label: 1 for real, 0 for fake (override if per-file metadata, but simple here)
        label = torch.tensor(1.0 if self.is_real else 0.0)

        # For training: return x, x_lf, r_hf, label
        # For inference: return x, label
        return {
            "input": x,
            "lf_target": x_lf,
            "hf_target": r_hf,
            "label": label,
            "file": os.path.basename(file_path),
        }


# Example usage (for testing)
if __name__ == "__main__":
    # Assume data exists
    dataset = LipLandmarkDataset(
        "adv ml/data/landmarks/synthetic", max_length=64, is_real=True
    )
    sample = dataset[0]
    print(
        f"Sample shapes: input {sample['input'].shape}, lf {sample['lf_target'].shape}, hf {sample['hf_target'].shape}"
    )
    print(f"Label: {sample['label']}")
