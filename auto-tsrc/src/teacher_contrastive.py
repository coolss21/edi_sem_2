"""
teacher_contrastive.py - LightweightContrastiveCNNTeacher

A lightweight CNN-based contrastive encoder inspired by TS2Vec-style
contrastive representation learning. This is NOT a full TS2Vec reproduction.
It uses SimCLR-style NT-Xent loss with two augmented views per sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from .augmentations import apply_augmentation


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LightweightContrastiveCNNTeacher(nn.Module):
    """
    Lightweight contrastive CNN teacher model inspired by TS2Vec-style
    contrastive representation learning.

    Input  : (batch, time_steps, 1)
    Output : (batch, embedding_dim) - L2-normalised embedding
    """

    def __init__(self, time_steps: int, embedding_dim: int = 64,
                 channels: list = None):
        super().__init__()
        if channels is None:
            channels = [32, 64]

        self.time_steps    = time_steps
        self.embedding_dim = embedding_dim

        # Conv blocks: input is (batch, 1, time_steps)
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels[1])

        self.pool  = nn.AdaptiveAvgPool1d(1)  # → (batch, channels[1], 1)
        self.proj  = nn.Linear(channels[1], embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time series to embedding.
        x: (batch, time_steps, 1)  →  returns (batch, embedding_dim)
        """
        # (batch, time_steps, 1) → (batch, 1, time_steps)
        h = x.permute(0, 2, 1)
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool(h).squeeze(-1)         # (batch, channels[1])
        h = self.proj(h)                     # (batch, embedding_dim)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent contrastive loss (SimCLR)
# ─────────────────────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent loss for batch of paired embeddings.
    z1, z2: (N, D) – both already L2-normalised.
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)          # (2N, D)

    sim = torch.mm(z, z.t()) / temperature  # (2N, 2N)

    # mask out self-similarities
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])
    loss = F.cross_entropy(sim, labels)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────────────────

def train_teacher(
    X_train: np.ndarray,
    teacher: LightweightContrastiveCNNTeacher,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    temperature: float = 0.07,
    aug_strength: str = "medium",
    jitter_sigma: float = 0.03,
    mask_ratio: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> list:
    """
    Train teacher with SimCLR-style NT-Xent contrastive loss.
    Returns list of per-epoch losses.
    """
    teacher = teacher.to(device)
    teacher.train()

    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    dataset   = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=g, drop_last=False)

    history = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0
        for (batch_x,) in loader:
            x_np = batch_x.cpu().numpy()

            # Two augmented views
            x1 = apply_augmentation(x_np, aug_strength, jitter_sigma, mask_ratio)
            x2 = apply_augmentation(x_np, aug_strength, jitter_sigma, mask_ratio)

            x1_t = torch.tensor(x1, dtype=torch.float32, device=device)
            x2_t = torch.tensor(x2, dtype=torch.float32, device=device)

            z1 = F.normalize(teacher.encode(x1_t), dim=-1)
            z2 = F.normalize(teacher.encode(x2_t), dim=-1)

            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)
        if verbose:
            print(f"  [Teacher] Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    teacher.eval()
    return history
