"""
student_autoencoder.py - CNN and LSTM autoencoder student models for TSRC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CNN Autoencoder Student
# ─────────────────────────────────────────────────────────────────────────────

class CNNAutoencoderStudent(nn.Module):
    """
    CNN-based autoencoder student.

    Input  : (batch, T, 1)
    Output : reconstruction (batch, T, 1), embedding (batch, embedding_dim)
    """

    def __init__(self, time_steps: int, embedding_dim: int = 64,
                 hidden_dim: int = 64):
        super().__init__()
        self.time_steps    = time_steps
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim

        # Encoder
        self.enc_conv1 = nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        self.enc_pool  = nn.AdaptiveAvgPool1d(1)
        self.enc_fc    = nn.Linear(hidden_dim, embedding_dim)

        # Decoder
        self.dec_fc    = nn.Linear(embedding_dim, hidden_dim * (time_steps // 4 + 1))
        self.dec_conv1 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1)

        self._dec_spatial = time_steps // 4 + 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1) → z: (B, embedding_dim)"""
        h = x.permute(0, 2, 1)                          # (B, 1, T)
        h = F.relu(self.enc_conv1(h))                   # (B, H/2, T)
        h = F.relu(self.enc_conv2(h))                   # (B, H, T)
        h = self.enc_pool(h).squeeze(-1)                # (B, H)
        z = self.enc_fc(h)                              # (B, E)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, E) → recon: (B, T, 1)"""
        B = z.size(0)
        h = F.relu(self.dec_fc(z))                      # (B, H * spatial)
        h = h.view(B, self.hidden_dim, self._dec_spatial)  # (B, H, spatial)
        h = F.interpolate(h, size=self.time_steps, mode='linear', align_corners=False)
        h = F.relu(self.dec_conv1(h))
        h = self.dec_conv2(h)                           # (B, 1, T)
        return h.permute(0, 2, 1)                       # (B, T, 1)

    def forward(self, x: torch.Tensor):
        z     = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder Student
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoderStudent(nn.Module):
    """
    LSTM-based autoencoder student.

    Input  : (batch, T, 1)
    Output : reconstruction (batch, T, 1), embedding (batch, embedding_dim)
    """

    def __init__(self, time_steps: int, embedding_dim: int = 64,
                 hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.time_steps    = time_steps
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers

        # Encoder LSTM  input_size=1
        self.enc_lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True
        )
        self.enc_fc   = nn.Linear(hidden_dim, embedding_dim)

        # Decoder LSTM
        self.dec_fc   = nn.Linear(embedding_dim, hidden_dim)
        self.dec_lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True
        )
        self.dec_out  = nn.Linear(hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1) → z: (B, embedding_dim)"""
        _, (h_n, _) = self.enc_lstm(x)   # h_n: (layers, B, H)
        h = h_n[-1]                       # (B, H)
        z = self.enc_fc(h)                # (B, E)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, E) → recon: (B, T, 1)"""
        B = z.size(0)
        h = F.relu(self.dec_fc(z))                        # (B, H)
        h_rep = h.unsqueeze(1).expand(B, self.time_steps, self.hidden_dim)
        out, _ = self.dec_lstm(h_rep)                     # (B, T, H)
        recon  = self.dec_out(out)                        # (B, T, 1)
        return recon

    def forward(self, x: torch.Tensor):
        z     = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_student(
    student_type: str,
    time_steps: int,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
) -> nn.Module:
    """Return a student model instance."""
    if student_type == "cnn_ae":
        return CNNAutoencoderStudent(time_steps, embedding_dim, hidden_dim)
    elif student_type == "lstm_ae":
        return LSTMAutoencoderStudent(time_steps, embedding_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown student_type: {student_type}. "
                         "Choose 'cnn_ae' or 'lstm_ae'.")
