import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.signal import stft, butter, filtfilt
from scipy.stats import entropy
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import glob
import argparse

# Assume model.py exists with LFHFVAE class
from model import LFHFVAE


class LipLandmarkDataset(Dataset):
    def __init__(self, data_dir, subdirs, seq_len=64, min_frames=48):
        self.seq_len = seq_len
        self.min_frames = min_frames
        self.files = []
        for subdir in subdirs:
            sub_path = os.path.join(data_dir, subdir)
            if os.path.exists(sub_path):
                self.files.extend(glob.glob(os.path.join(sub_path, "*.npy")))
        self.files = [f for f in self.files if np.load(f).shape[0] >= min_frames]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        seq = np.load(file_path)  # (T, K, 2)
        if seq.shape[0] > self.seq_len:
            start = np.random.randint(0, seq.shape[0] - self.seq_len)
            seq = seq[start : start + self.seq_len]
        else:
            pad_len = self.seq_len - seq.shape[0]
            seq = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
        seq = seq.reshape(self.seq_len, -1).astype(np.float32)  # (T, 50)
        return torch.from_numpy(seq)


def lowpass_filter(x, fs=25, cutoff=5.0, order=4):
    """
    Robust low-pass filter that handles short sequences by adapting padlen and
    falling back to EMA smoothing if needed.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    T = x.shape[0]
    # Default padlen used by filtfilt is 3 * max(len(a), len(b))
    default_pad = 3 * max(len(a), len(b))
    # Ensure padlen < T and non-negative
    padlen = max(0, min(default_pad, T - 1))

    if T <= 2 or padlen < 1:
        # Fallback: simple EMA smoothing along time axis
        alpha = 0.3
        y = np.copy(x)
        for t in range(1, T):
            y[t] = alpha * y[t - 1] + (1.0 - alpha) * x[t]
        return y

    # Use Gustafsson method for better edge behavior; adapt padlen for short T
    return filtfilt(b, a, x, axis=0, padlen=padlen, method="gust")


def kl_divergence(mu, logvar):
    # Flexible KL for q(z|x)=N(mu, sigma^2) vs p(z)=N(0,I).
    # Works with shapes (B, L) or (B, T, L).
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    reduce_dims = list(range(1, kl.dim()))
    return kl.sum(dim=reduce_dims)


def stft_loss(hat_r, r, fs=25, nperseg=32):
    """
    Non-differentiable STFT-based auxiliary loss computed with NumPy on CPU.
    Returns a Python float to avoid device/dtype issues (e.g., MPS float64).
    """
    total = 0.0
    B = hat_r.shape[0]
    for b in range(B):
        f_hat, t_hat, Zxx_hat = stft(
            hat_r[b].detach().cpu().numpy(), fs=fs, nperseg=nperseg
        )
        f, t, Zxx = stft(r[b].detach().cpu().numpy(), fs=fs, nperseg=nperseg)
        abs_hat = np.abs(Zxx_hat)
        abs_ref = np.abs(Zxx)
        mse = float(np.mean((abs_hat - abs_ref) ** 2))
        # High freq weight (upper 1/3)
        high_start = len(f) // 3
        high_mse = float(
            2.0 * np.sum((abs_hat[high_start:] - abs_ref[high_start:]) ** 2)
        )
        total += mse + high_mse
    return float(total / max(1, B))


def gradient_loss(hat_x, x):
    grad1_hat = torch.diff(hat_x, dim=1)
    grad1_x = torch.diff(x, dim=1)
    l1 = F.l1_loss(grad1_hat, grad1_x)
    grad2_hat = torch.diff(grad1_hat, dim=1)
    grad2_x = torch.diff(grad1_x, dim=1)
    l2 = F.l1_loss(grad2_hat, grad2_x)
    grad3_hat = torch.diff(grad2_hat, dim=1)
    grad3_x = torch.diff(grad2_x, dim=1)
    l3 = F.l1_loss(grad3_hat, grad3_x)
    return l1 + 0.5 * l2 + 0.3 * l3


def decorrelation_loss(z_lf, z_hf, eps: float = 1e-5):
    """
    Encourage LF and HF latents to be decorrelated by penalizing the
    cross-correlation Frobenius norm between LF and HF features.

    Accepts flexible shapes:
      - z_lf: (B, Lf) or (B, T, Lf) or (Lf,)
      - z_hf: (B, T, Lh) or (B, Lh)

    We align LF to HF by repeating LF across time if needed, then flatten
    (B, T, ·) -> (N, ·) with N = B*T, standardize each feature dim, and
    compute C = (Z_lf^T @ Z_hf) / N. Loss = ||C||_F^2.
    """
    # Normalize shapes
    if z_hf.dim() == 2:
        # (B, Lh) -> add time dim T=1
        z_hf = z_hf.unsqueeze(1)  # (B, 1, Lh)
    elif z_hf.dim() != 3:
        raise ValueError(f"z_hf must be (B, Lh) or (B, T, Lh), got {z_hf.shape}")

    B, T, Lh = z_hf.shape

    if z_lf.dim() == 1:
        # (Lf,) -> (B, T, Lf) by repeat
        Lf = z_lf.shape[0]
        z_lf = z_lf.view(1, 1, Lf).repeat(B, T, 1)
    elif z_lf.dim() == 2:
        # (B, Lf) -> (B, T, Lf)
        B_lf, Lf = z_lf.shape
        if B_lf != B:
            raise ValueError(f"Batch mismatch: z_lf {z_lf.shape}, z_hf {z_hf.shape}")
        z_lf = z_lf.unsqueeze(1).repeat(1, T, 1)  # (B, T, Lf)
    elif z_lf.dim() == 3:
        # (B, T, Lf)
        if z_lf.shape[0] != B or z_lf.shape[1] != T:
            raise ValueError(f"BT mismatch: z_lf {z_lf.shape}, z_hf {z_hf.shape}")
        Lf = z_lf.shape[2]
    else:
        raise ValueError(f"Unsupported z_lf shape: {z_lf.shape}")

    # Flatten batch and time
    Zlf = z_lf.reshape(B * T, Lf)  # (N, Lf)
    Zhf = z_hf.reshape(B * T, Lh)  # (N, Lh)

    # Standardize columns (zero-mean, unit-variance)
    Zlf = Zlf - Zlf.mean(dim=0, keepdim=True)
    Zhf = Zhf - Zhf.mean(dim=0, keepdim=True)
    Zlf = Zlf / (Zlf.std(dim=0, unbiased=False, keepdim=True) + eps)
    Zhf = Zhf / (Zhf.std(dim=0, unbiased=False, keepdim=True) + eps)

    # Cross-correlation matrix
    N = Zlf.shape[0]
    C = (Zlf.T @ Zhf) / max(1, N)  # (Lf, Lh)

    # Frobenius norm squared
    loss = (C**2).sum()
    return loss


def jitter_augment(x, max_shift=1, drop_prob=0.05):
    B, T, D = x.shape
    x_jit = x.clone()
    shift = torch.randint(-max_shift, max_shift + 1, (B,))
    for b in range(B):
        s = int(shift[b].item())
        if s != 0:
            x_jit[b] = torch.roll(x_jit[b], shifts=s, dims=0)
    drop_mask = torch.rand(B, T, device=x.device) > drop_prob
    x_jit = x_jit * drop_mask.unsqueeze(-1).float()
    for b in range(B):
        for t in range(1, T):
            if not drop_mask[b, t]:
                x_jit[b, t] = x_jit[b, t - 1]
    return x_jit


def train_vae(
    data_dir,
    real_subdirs,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    seq_len=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
    hidden_dim=128,
    lf_latent_dim=32,
    hf_latent_dim=64,
    num_tcn_layers=4,
    tcn_kernel_size=3,
    tcn_dropout=0.1,
):
    dataset = LipLandmarkDataset(data_dir, real_subdirs, seq_len, min_frames=48)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Infer input_dim (K*2) from first landmark file
    first = np.load(dataset.files[0])
    input_dim = int(first.shape[1] * first.shape[2])
    model = LFHFVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        lf_latent_dim=lf_latent_dim,
        hf_latent_dim=hf_latent_dim,
        num_tcn_layers=num_tcn_layers,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    beta_lf = 3.0
    beta_hf = 0.4
    kl_anneal_epochs = 20
    kl_cap = 0.3

    model.train()
    losses = []
    kl_step = 0  # placeholder (KL annealing uses per-step factor below)
    log_history = {
        "rec_lf": [],
        "rec_hf": [],
        "stft": [],
        "grad": [],
        "full_rec": [],
        "decor": [],
        "jitter": [],
        "kl_lf": [],
        "kl_hf": [],
        "total": [],
    }

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        sum_rec_lf = 0.0
        sum_rec_hf = 0.0
        sum_stft = 0.0
        sum_grad = 0.0
        sum_full = 0.0
        sum_decor = 0.0
        sum_jitter = 0.0
        sum_kl_lf = 0.0
        sum_kl_hf = 0.0
        sum_total = 0.0
        num_batches = 0

        for batch_idx, x in enumerate(pbar):
            x = x.to(device)  # (B, T, 50)
            B, T, D = x.shape
            # Per-step KL annealing factor (linear warm-up capped by kl_cap)
            total_anneal_steps = max(1, kl_anneal_epochs * len(dataloader))
            global_step = epoch * len(dataloader) + batch_idx
            kl_factor = kl_cap * min(
                1.0, float(global_step) / float(total_anneal_steps)
            )

            out = model(x)
            hat_x_lf = out["x_lf"]
            hat_r_hf = out["r_hf"]
            hat_x = out["x_hat"]
            mu_lf, logvar_lf = out["lf_mu"], out["lf_logvar"]
            mu_hf, logvar_hf = out["hf_mu"], out["hf_logvar"]

            # LF target
            x_np = x.detach().cpu().numpy().copy()
            x_lp_np = np.ascontiguousarray(lowpass_filter(x_np, fs=25))
            x_lp = torch.from_numpy(x_lp_np.astype(np.float32, copy=False)).to(device)
            r_hf = x - x_lp

            # Losses
            rec_lf = F.mse_loss(hat_x_lf, x_lp)
            kl_lf = kl_divergence(mu_lf, logvar_lf).mean()
            kl_lf_loss = beta_lf * kl_lf * (kl_factor)
            # KL warm-up step accounted in kl_factor

            # HF and auxiliary losses
            rec_hf = F.mse_loss(hat_r_hf, r_hf)
            stft_l = stft_loss(hat_r_hf, r_hf)
            grad_l = gradient_loss(hat_x, x)
            full_rec = F.l1_loss(hat_x, x)
            decor_l = decorrelation_loss(mu_lf.mean(0), mu_hf)
            # KL for HF branch (per-timestep), annealed
            kl_hf = kl_divergence(mu_hf, logvar_hf).mean()
            kl_hf_loss = beta_hf * kl_hf * (kl_factor)
            # Jitter consistency
            x_jit = jitter_augment(x)
            out_jit = model(x_jit)
            hat_r_hf_jit = out_jit["r_hf"]
            jitter_l = F.l1_loss(hat_r_hf, hat_r_hf_jit)

            total_loss = (
                1.0 * rec_lf
                + kl_lf_loss
                + 0.5 * rec_hf
                + 0.5 * stft_l
                + 0.3 * grad_l
                + 0.2 * full_rec
                + 0.1 * decor_l
                + 0.1 * jitter_l
                + kl_hf_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            sum_total += total_loss.item()
            sum_rec_lf += rec_lf.item()
            sum_rec_hf += rec_hf.item()
            sum_stft += (
                stft_l.item() if isinstance(stft_l, torch.Tensor) else float(stft_l)
            )
            sum_grad += grad_l.item()
            sum_full += full_rec.item()
            sum_decor += decor_l.item()
            sum_jitter += jitter_l.item()
            sum_kl_lf += kl_lf_loss.item()
            sum_kl_hf += kl_hf_loss.item()
            num_batches += 1
            pbar.set_postfix(
                loss=total_loss.item(),
                rec_lf=rec_lf.item(),
                rec_hf=rec_hf.item(),
                kl_lf=kl_lf_loss.item(),
                kl_hf=kl_hf_loss.item(),
            )

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(dataloader))
        losses.append(avg_loss)
        # Store per-loss epoch averages
        if num_batches > 0:
            log_history["rec_lf"].append(sum_rec_lf / num_batches)
            log_history["rec_hf"].append(sum_rec_hf / num_batches)
            log_history["stft"].append(sum_stft / num_batches)
            log_history["grad"].append(sum_grad / num_batches)
            log_history["full_rec"].append(sum_full / num_batches)
            log_history["decor"].append(sum_decor / num_batches)
            log_history["jitter"].append(sum_jitter / num_batches)
            log_history["kl_lf"].append(sum_kl_lf / num_batches)
            log_history["kl_hf"].append(sum_kl_hf / num_batches)
            log_history["total"].append(sum_total / num_batches)
        print(
            f"Epoch {epoch + 1} | total {avg_loss:.4f} | rec_lf {log_history['rec_lf'][-1]:.4f} | rec_hf {log_history['rec_hf'][-1]:.4f} | kl_lf {log_history['kl_lf'][-1]:.4f} | kl_hf {log_history['kl_hf'][-1]:.4f}"
        )

        if (epoch + 1) % 10 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/vae_epoch_{epoch + 1}.pth")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/vae_final.pth")

    import matplotlib.pyplot as plt

    epochs_axis = list(range(1, len(losses) + 1))
    # Plot multiple losses
    plt.figure()
    plt.plot(epochs_axis, losses, label="total")
    for key in [
        "rec_lf",
        "rec_hf",
        "stft",
        "grad",
        "full_rec",
        "decor",
        "jitter",
        "kl_lf",
        "kl_hf",
    ]:
        if key in log_history and len(log_history[key]) == len(epochs_axis):
            plt.plot(epochs_axis, log_history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_losses.png")
    plt.close()
    # Save CSV of losses
    import csv

    with open("training_losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "epoch",
            "total",
            "rec_lf",
            "rec_hf",
            "stft",
            "grad",
            "full_rec",
            "decor",
            "jitter",
            "kl_lf",
            "kl_hf",
        ]
        writer.writerow(header)
        for i in range(len(epochs_axis)):
            row = [
                epochs_axis[i],
                losses[i],
                *(
                    log_history[k][i]
                    if k in log_history and len(log_history[k]) > i
                    else float("nan")
                    for k in header[2:]
                ),
            ]
            writer.writerow(row)

    print("Training completed. Model saved to models/vae_final.pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LF/HF VAE on lip landmarks")
    parser.add_argument(
        "--data_dir", type=str, default="data/landmarks", help="Data directory"
    )
    parser.add_argument(
        "--real_subdirs", nargs="+", default=["real"], help="Real data subdirs"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Model hidden dimension (TCN/GRU width)",
    )
    parser.add_argument(
        "--lf_latent_dim", type=int, default=32, help="LF branch latent dim"
    )
    parser.add_argument(
        "--hf_latent_dim", type=int, default=64, help="HF branch latent dim"
    )
    parser.add_argument(
        "--num_tcn_layers", type=int, default=4, help="Number of TCN layers in encoder"
    )
    parser.add_argument(
        "--tcn_kernel_size", type=int, default=3, help="TCN kernel size"
    )
    parser.add_argument("--tcn_dropout", type=float, default=0.1, help="TCN dropout")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    args = parser.parse_args()

    train_vae(
        args.data_dir,
        args.real_subdirs,
        args.epochs,
        args.batch_size,
        args.lr,
        device=args.device,
        hidden_dim=args.hidden_dim,
        lf_latent_dim=args.lf_latent_dim,
        hf_latent_dim=args.hf_latent_dim,
        num_tcn_layers=args.num_tcn_layers,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_dropout=args.tcn_dropout,
    )
