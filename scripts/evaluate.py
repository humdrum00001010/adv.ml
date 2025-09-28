import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, ConcatDataset
import scipy.signal as signal
from scipy.stats import entropy
from model import LFHFVAE
from data_loader import LipLandmarkDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def compute_lpf(x, fs=25, cutoff=5.0):
    """Low-pass filter for LF target using Butterworth."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(4, normal_cutoff, btype="low", analog=False)
    x_lf = signal.filtfilt(b, a, x, axis=0)
    return x_lf


def compute_stft_loss(hat_r, r, fs=25):
    """STFT weighted loss, emphasize high freq (not used in eval, but for completeness)."""
    f, t, Zxx_hat = signal.stft(hat_r, fs=fs, nperseg=16)
    _, _, Zxx = signal.stft(r, fs=fs, nperseg=16)
    loss = np.mean(np.abs(Zxx_hat - Zxx) ** 2)
    high_freq_mask = f > np.median(f)
    loss += 2.0 * np.mean(np.abs(Zxx_hat[high_freq_mask] - Zxx[high_freq_mask]) ** 2)
    return loss


def sample_entropy(data, m=2, r=0.2):
    """Approximate sample entropy for complexity C."""
    N = len(data.flatten())
    if N < m + 1:
        return 0.0
    hist, _ = np.histogram(data.flatten(), bins=10)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def jitter_input(x, max_shift=1, drop_prob=0.05):
    """Apply temporal jitter: ±1 frame shift, 5% frame drop."""
    x_jit = x.clone()
    T = x.shape[0]
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        x_jit = torch.roll(x_jit, shifts=shift, dims=0)
        x_jit[:shift] = x_jit[shift]
    elif shift < 0:
        x_jit = torch.roll(x_jit, shifts=shift, dims=0)
        x_jit[T + shift :] = x_jit[T + shift - 1]

    # Frame drops with forward fill
    drop_mask = torch.rand(T) > drop_prob
    for t in range(1, T):
        if not drop_mask[t]:
            x_jit[t] = x_jit[t - 1]
    return x_jit


def evaluate_sample(model, x, device, alpha=1.0, beta=0.5, gamma=0.2, tau=0.5):
    """Evaluate a single sequence. Accepts (T, K, 2) or (T, D) and flattens to (T, D)."""
    model.eval()
    with torch.no_grad():
        # Ensure shape (T, D)
        if x.dim() == 3 and x.shape[-1] == 2:
            T = x.shape[0]
            x_flat = x.reshape(T, -1).float()
        elif x.dim() == 2:
            x_flat = x.float()
        else:
            raise ValueError(
                f"Unsupported input shape for evaluation: {tuple(x.shape)}"
            )

        # Forward
        out = model(x_flat.unsqueeze(0).to(device))
        x_lf_hat = out["x_lf"].squeeze(0)  # (T, D)
        r_hf_hat = out["r_hf"].squeeze(0)  # (T, D)
        x_hat = out["x_hat"].squeeze(0)  # (T, D)

        # Targets computed on flattened (T, D)
        x_np = np.ascontiguousarray(x_flat.cpu().numpy())
        x_lf_np = np.ascontiguousarray(compute_lpf(x_np))
        x_lf_target = torch.from_numpy(x_lf_np).float().to(device)  # (T, D)
        r_hf_target = x_flat.to(device) - x_lf_target

        # Errors
        e_smooth = torch.mean((x_flat.to(device) - x_lf_hat) ** 2).item()
        e_detail = torch.mean((r_hf_target - r_hf_hat) ** 2).item()

        # DRS
        drs = e_smooth / (e_detail + 1e-6)

        # Delta E with jitter on flattened input
        x_jit = jitter_input(x_flat)
        out_jit = model(x_jit.unsqueeze(0).to(device))
        r_hf_hat_jit = out_jit["r_hf"].squeeze(0)
        delta_e = torch.mean((r_hf_hat - r_hf_hat_jit) ** 2).item()

        # Complexity C (use flattened data)
        c = sample_entropy(x_np)

        # Score S
        s = alpha * drs - beta * delta_e + gamma * c

        # Classify
        pred = 1 if s > tau else 0  # 1 = live

        return s, pred, drs, delta_e, c, e_smooth, e_detail


def evaluate_model(
    model_path,
    data_dir,
    real_subdirs,
    fake_subdirs,
    device="cpu",
    batch_size=8,
    alpha=1.0,
    beta=0.5,
    gamma=0.2,
    tau=0.5,
    tau_quantile=None,
    hidden_dim: int = 128,
    lf_latent_dim: int = 32,
    hf_latent_dim: int = 64,
    num_tcn_layers: int = 4,
    tcn_kernel_size: int = 3,
    tcn_dropout: float = 0.1,
):
    """Full evaluation."""
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Infer input_dim (K*2) from first available real sample
    input_dim = 50
    for subdir in real_subdirs:
        path = os.path.join(data_dir, subdir)
        if os.path.exists(path):
            npy_files = sorted([f for f in os.listdir(path) if f.endswith(".npy")])
            if npy_files:
                sample = np.load(os.path.join(path, npy_files[0]))
                if sample.ndim == 3:
                    input_dim = int(sample.shape[1] * sample.shape[2])
                else:
                    input_dim = int(sample.shape[-1])
                break
    model = LFHFVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        lf_latent_dim=lf_latent_dim,
        hf_latent_dim=hf_latent_dim,
        num_tcn_layers=num_tcn_layers,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)

    # Datasets
    real_datasets = []
    for subdir in real_subdirs:
        path = os.path.join(data_dir, subdir)
        if os.path.exists(path):
            ds = LipLandmarkDataset(path, is_real=True)
            real_datasets.append(ds)
    real_ds = ConcatDataset(real_datasets) if real_datasets else None

    fake_datasets = []
    for subdir in fake_subdirs:
        path = os.path.join(data_dir, subdir)
        if os.path.exists(path):
            ds = LipLandmarkDataset(path, is_real=False)
            fake_datasets.append(ds)
    fake_ds = ConcatDataset(fake_datasets) if fake_datasets else None

    real_loader = (
        DataLoader(real_ds, batch_size=batch_size, shuffle=False)
        if real_ds is not None
        else None
    )
    fake_loader = (
        DataLoader(fake_ds, batch_size=batch_size, shuffle=False)
        if fake_ds is not None
        else None
    )

    all_scores = []
    all_labels = []
    all_preds = []

    # Eval real
    if real_loader is not None:
        for batch in real_loader:
            x = batch["input"]  # (B, T, D)
            for i in range(x.shape[0]):
                s, pred, _, _, _, _, _ = evaluate_sample(
                    model, x[i], device, alpha, beta, gamma, tau
                )
                all_scores.append(s)
                all_labels.append(1)
                all_preds.append(pred)

    # Eval fake
    if fake_loader is not None:
        for batch in fake_loader:
            x = batch["input"]
            for i in range(x.shape[0]):
                s, pred, _, _, _, _, _ = evaluate_sample(
                    model, x[i], device, alpha, beta, gamma, tau
                )
                all_scores.append(s)
                all_labels.append(0)
                all_preds.append(pred)

    # Metrics and reporting
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    real_scores = all_scores[all_labels == 1] if all_labels.size else np.array([])
    fake_scores = all_scores[all_labels == 0] if all_labels.size else np.array([])

    # Calibrate tau from real-only scores if requested
    chosen_tau = tau
    if (tau_quantile is not None) and (len(real_scores) > 0):
        chosen_tau = float(np.quantile(real_scores, tau_quantile))
        print(f"Calibrated tau at quantile {tau_quantile:.2f}: {chosen_tau:.6f}")
    else:
        print(f"Using tau: {chosen_tau:.6f}")

    # Recompute predictions with chosen threshold
    all_preds = (all_scores > chosen_tau).astype(int)

    print("Evaluation Results:")
    print(
        f"Total samples: {len(all_scores)} (real: {len(real_scores)}, fake: {len(fake_scores)})"
    )

    acc = None
    if len(real_scores) > 0 and len(fake_scores) > 0:
        acc = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {acc:.2f}")
    elif len(real_scores) > 0 and len(fake_scores) == 0:
        live_recall = (
            float(np.mean(all_preds[all_labels == 1] == 1))
            if np.any(all_labels == 1)
            else float("nan")
        )
        print(f"Real-only live recall at tau: {live_recall:.2f}")

    if len(real_scores) > 0:
        print(
            f"Avg real score: {np.mean(real_scores):.4f} (std: {np.std(real_scores):.4f})"
        )
    if len(fake_scores) > 0:
        print(
            f"Avg fake score: {np.mean(fake_scores):.4f} (std: {np.std(fake_scores):.4f})"
        )

    # Confusion matrix (only if both classes present)
    if len(real_scores) > 0 and len(fake_scores) > 0:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Spoof", "Live"],
            yticklabels=["Spoof", "Live"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("data/eval_confusion_matrix.png")
        plt.close()

    # Score histograms
    plt.figure(figsize=(6, 4))
    if len(real_scores) > 0:
        plt.hist(real_scores, bins=20, alpha=0.6, label="Real")
    if len(fake_scores) > 0:
        plt.hist(fake_scores, bins=20, alpha=0.6, label="Fake")
    # Mark chosen tau
    plt.axvline(chosen_tau, color="red", linestyle="--", label=f"tau={chosen_tau:.3f}")
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/eval_score_hist.png")
    plt.close()

    # Save scores CSV (works even if one class is empty)
    np.savetxt(
        "data/eval_scores.csv",
        np.column_stack((all_labels, all_preds, all_scores))
        if all_labels.size
        else np.empty((0, 3)),
        delimiter=",",
        header="label,pred,score",
        comments="",
    )

    return acc, all_scores, all_labels, all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Lip VAE for deep fake detection"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/vae_final.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/landmarks",
        help="Directory with .npy landmarks",
    )
    parser.add_argument(
        "--real_subdirs", nargs="+", default=["real"], help="Real data subdirs"
    )
    parser.add_argument(
        "--fake_subdirs", nargs="+", default=["fake"], help="Fake data subdirs"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    # Model hyperparameters for evaluation
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lf_latent_dim", type=int, default=32)
    parser.add_argument("--hf_latent_dim", type=int, default=64)
    parser.add_argument("--num_tcn_layers", type=int, default=4)
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument("--tcn_dropout", type=float, default=0.1)
    # Device and scoring params
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument(
        "--tau_quantile",
        type=float,
        default=None,
        help="If provided, calibrate tau as this quantile of real-only scores (e.g., 0.80). Overrides --tau.",
    )
    args = parser.parse_args()

    acc, scores, labels, preds = evaluate_model(
        args.model_path,
        args.data_dir,
        args.real_subdirs,
        args.fake_subdirs,
        device=args.device,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        tau=args.tau,
        tau_quantile=args.tau_quantile,
        hidden_dim=args.hidden_dim,
        lf_latent_dim=args.lf_latent_dim,
        hf_latent_dim=args.hf_latent_dim,
        num_tcn_layers=args.num_tcn_layers,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_dropout=args.tcn_dropout,
    )
