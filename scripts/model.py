import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Temporal Convolution Modules
# ---------------------------


class CausalConv1d(nn.Module):
    """
    1D causal convolution over time with left padding so output length == input length.
    Input shape expected: (B, C, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # we'll pad manually to enforce causality
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.left_pad, 0))  # left, right
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Robust TCN block with:
      - CausalConv1d -> BN -> ReLU -> Dropout
      - CausalConv1d -> BN -> ReLU -> Dropout
      - Residual connection (with 1x1 conv if channel mismatch)
    Input/Output: (B, C_in, T) -> (B, C_out, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.res_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        residual = self.res_proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        return out + residual  # (B, C_out, T)


# ---------------------------
# Encoder
# ---------------------------


class TCNEncoder(nn.Module):
    """
    Multi-layer TCN encoder over time.
    Input:  (B, T, D)  -> transpose to (B, D, T)
    Output: (B, T, H)  (transpose back)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i > 0 else max(64, min(hidden_dim, input_dim * 2))
            # keep final layer at hidden_dim
            if i == num_layers - 1:
                out_ch = hidden_dim
            d = dilations[i % len(dilations)]
            layers.append(
                TCNBlock(
                    in_ch, out_ch, kernel_size=kernel_size, dilation=d, dropout=dropout
                )
            )
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        h = self.net(x)  # (B, H, T)
        h = h.transpose(1, 2)  # (B, T, H)
        return h


# ---------------------------
# LF Head (clip-level latent)
# ---------------------------


class LFVAEHead(nn.Module):
    """
    Low-frequency VAE head:
      - Encode: temporal attention pooling -> z_lf ~ N(mu, sigma)
      - Decode: GRU from z_lf expanded over T -> (B, T, D)
    """

    def __init__(
        self, hidden_dim: int = 128, latent_dim: int = 32, output_dim: int = 50
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Attention to get a clip-level representation
        self.attn_proj = nn.Linear(hidden_dim, 1)  # scalar score per timestep
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> repeated over T -> GRU -> linear
        self.gru_dec = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def encode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, H)
        attn_scores = self.attn_proj(h).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T)
        pooled = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)  # (B, H)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Clamp log-variance to stabilize sampling
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # (B, latent_dim)

    def decode(self, z: torch.Tensor, T: int) -> torch.Tensor:
        # z: (B, latent_dim), expand to (B, T, latent_dim)
        z_rep = z.unsqueeze(1).repeat(1, T, 1)  # (B, T, latent_dim)
        y, _ = self.gru_dec(z_rep)  # (B, T, H)
        out = self.proj_out(y)  # (B, T, D)
        return out


# ---------------------------
# HF Head (frame-level latent)
# ---------------------------


class HFVAEHead(nn.Module):
    """
    High-frequency VAE head:
      - Encode: per-timestep to z_hf ~ N(mu, sigma)
      - Decode: GRU conditioned on z_hf(t) -> (B, T, D)
    """

    def __init__(
        self, hidden_dim: int = 128, latent_dim: int = 64, output_dim: int = 50
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.gru_dec = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def encode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, H)
        mu = self.fc_mu(h)  # (B, T, L)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Clamp log-variance to stabilize sampling
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # (B, T, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, latent_dim)
        y, _ = self.gru_dec(z)  # (B, T, H)
        out = self.proj_out(y)  # (B, T, D)
        return out


# ---------------------------
# Full LF/HF VAE
# ---------------------------


class LFHFVAE(nn.Module):
    """
    LF/HF VAE on lip landmark sequences flattened to D=K*2 (e.g., 50).
    - Encoder: TCN over time -> (B, T, H)
    - LF head: clip-level latent -> GRU decoder -> x_lf
    - HF head: frame-level latent -> GRU decoder -> r_hf
    - Output: x_hat = x_lf + r_hf
    """

    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        lf_latent_dim: int = 32,
        hf_latent_dim: int = 64,
        num_tcn_layers: int = 4,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = TCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_tcn_layers,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
        )

        self.lf_head = LFVAEHead(
            hidden_dim=hidden_dim, latent_dim=lf_latent_dim, output_dim=input_dim
        )
        self.hf_head = HFVAEHead(
            hidden_dim=hidden_dim, latent_dim=hf_latent_dim, output_dim=input_dim
        )

    @staticmethod
    def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute elementwise KL for Gaussian q(z|x)=N(mu, sigma^2) vs p(z)=N(0, I).
        Returns KL summed over latent dim; preserves batch (and time if present).
        Shapes:
          - (B, L) -> (B,)
          - (B, T, L) -> (B, T)
        """
        # Clamp log-variance before KL for numerical stability
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        # 0.5 * sum( mu^2 + exp(logvar) - 1 - logvar )
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        return kl.sum(dim=-1)

    def compute_kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        reduction: str = "mean",
        time_mean: bool = True,
    ) -> torch.Tensor:
        """
        KL loss with flexible reduction.
        - If input has T dimension, optionally average over time before batch reduction.
        - reduction in {"none", "mean", "sum"} (over batch after time handling).
        """
        kl = self._kl_divergence(mu, logvar)  # (B,) or (B, T)
        if kl.dim() == 2 and time_mean:
            kl = kl.mean(dim=1)  # (B,)

        if reduction == "none":
            return kl
        elif reduction == "mean":
            return kl.mean()
        elif reduction == "sum":
            return kl.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, T, D)
        Returns dict with:
          - x_lf: (B, T, D)
          - r_hf: (B, T, D)
          - x_hat: (B, T, D)
          - lf_mu, lf_logvar: (B, Lf)
          - hf_mu, hf_logvar: (B, T, Lh)
          - h: (B, T, H)
        """
        B, T, D = x.shape
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, got {D}"

        # Encoder features
        h = self.encoder(x)  # (B, T, H)

        # LF branch
        lf_mu, lf_logvar = self.lf_head.encode(h)  # (B, Lf)
        lf_z = self.lf_head.reparameterize(lf_mu, lf_logvar)  # (B, Lf)
        x_lf = self.lf_head.decode(lf_z, T=T)  # (B, T, D)

        # HF branch
        hf_mu, hf_logvar = self.hf_head.encode(h)  # (B, T, Lh)
        hf_z = self.hf_head.reparameterize(hf_mu, hf_logvar)  # (B, T, Lh)
        r_hf = self.hf_head.decode(hf_z)  # (B, T, D)

        x_hat = x_lf + r_hf

        return {
            "x_lf": x_lf,
            "r_hf": r_hf,
            "x_hat": x_hat,
            "lf_mu": lf_mu,
            "lf_logvar": lf_logvar,
            "hf_mu": hf_mu,
            "hf_logvar": hf_logvar,
            "h": h,
        }


# ---------------------------
# Utilities
# ---------------------------


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clip_model_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients of a model to stabilize training.
    Returns the total norm of the parameters (viewed as a single vector).
    Call this after loss.backward() and before optimizer.step().
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


# ---------------------------
# Smoke test
# ---------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    B, T, D = 8, 64, 50
    model = LFHFVAE(input_dim=D, hidden_dim=128, lf_latent_dim=32, hf_latent_dim=64).to(
        device
    )
    print(f"Total parameters: {count_parameters(model):,}")

    x = torch.randn(B, T, D, device=device)
    out = model(x)

    print(f"x_lf: {out['x_lf'].shape}")
    print(f"r_hf: {out['r_hf'].shape}")
    print(f"x_hat: {out['x_hat'].shape}")
    print(f"lf_mu: {out['lf_mu'].shape}, lf_logvar: {out['lf_logvar'].shape}")
    print(f"hf_mu: {out['hf_mu'].shape}, hf_logvar: {out['hf_logvar'].shape}")

    # KL loss examples
    kl_lf = model.compute_kl_loss(out["lf_mu"], out["lf_logvar"], reduction="mean")
    kl_hf = model.compute_kl_loss(
        out["hf_mu"], out["hf_logvar"], reduction="mean", time_mean=True
    )
    print(f"KL(LF): {kl_lf.item():.4f}, KL(HF): {kl_hf.item():.4f}")
