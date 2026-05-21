"""
trainer.py - Training routines for teacher, student-only, and TSRC student.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .curriculum import CurriculumScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(X, batch_size, shuffle=True, seed=42):
    X_t = torch.tensor(X, dtype=torch.float32)
    ds  = TensorDataset(X_t)
    g   = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=g, drop_last=False)


def _val_split(X: np.ndarray, val_ratio: float = 0.15, seed: int = 42):
    """Split array into train/val."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_ratio))
    return X[idx[n_val:]], X[idx[:n_val]]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Student Autoencoder Only (no teacher, reconstruction only)
# ─────────────────────────────────────────────────────────────────────────────

def train_student_autoencoder_only(
    X_train: np.ndarray,
    student: nn.Module,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    verbose: bool = True,
    history_path: str = None,
) -> dict:
    """
    Train student purely on reconstruction MSE with no teacher guidance.
    Returns training history dict.
    """
    student = student.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr, X_val = _val_split(X_train, seed=seed)
    train_loader = _make_loader(X_tr,  batch_size, shuffle=True,  seed=seed)
    val_loader   = _make_loader(X_val, batch_size, shuffle=False, seed=seed)

    best_val  = float("inf")
    no_improve = 0
    history    = {"epoch": [], "recon_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss = 0.0
        n_batch    = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            recon, _ = student(xb)
            loss = criterion(recon, xb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item(); n_batch += 1

        avg_train = epoch_loss / max(n_batch, 1)

        # Validation
        student.eval()
        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, _ = student(xb)
                val_loss += criterion(recon, xb).item(); n_val += 1
        avg_val = val_loss / max(n_val, 1)

        history["epoch"].append(epoch)
        history["recon_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if verbose:
            print(f"  [StudentOnly] Epoch {epoch:3d}/{epochs}  "
                  f"recon={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val - 1e-6:
            best_val   = avg_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [StudentOnly] Early stopping at epoch {epoch}.")
                break

    student.eval()
    if history_path:
        _save_history_csv(history, history_path)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TSRC Student Training
# ─────────────────────────────────────────────────────────────────────────────

def train_tsrc_student(
    X_train: np.ndarray,
    student: nn.Module,
    teacher: nn.Module,
    device: torch.device,
    scheduler: CurriculumScheduler,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    verbose: bool = True,
    history_path: str = None,
) -> dict:
    """
    Train TSRC student with frozen teacher and curriculum lambda.

    Loss = (1 - lambda) * ReconLoss + lambda * HintLoss
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr, X_val = _val_split(X_train, seed=seed)
    train_loader = _make_loader(X_tr,  batch_size, shuffle=True,  seed=seed)
    val_loader   = _make_loader(X_val, batch_size, shuffle=False, seed=seed)

    best_val   = float("inf")
    no_improve = 0
    history    = {
        "epoch": [], "recon_loss": [], "hint_loss": [],
        "total_loss": [], "lambda": [], "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        lam = scheduler.get_lambda(epoch - 1)  # 0-indexed
        student.train()
        ep_recon = ep_hint = ep_total = 0.0
        n_batch = 0

        for (xb,) in train_loader:
            xb = xb.to(device)

            with torch.no_grad():
                teacher_emb = teacher.encode(xb)   # (B, E)

            recon, student_emb = student(xb)
            recon_loss = criterion(recon, xb)
            hint_loss  = criterion(student_emb, teacher_emb)
            total_loss = (1.0 - lam) * recon_loss + lam * hint_loss

            optimizer.zero_grad(); total_loss.backward(); optimizer.step()

            ep_recon += recon_loss.item()
            ep_hint  += hint_loss.item()
            ep_total += total_loss.item()
            n_batch  += 1

        avg_recon = ep_recon / max(n_batch, 1)
        avg_hint  = ep_hint  / max(n_batch, 1)
        avg_total = ep_total / max(n_batch, 1)

        # Validation
        student.eval()
        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                with torch.no_grad():
                    t_emb = teacher.encode(xb)
                recon, s_emb = student(xb)
                r_l = criterion(recon, xb)
                h_l = criterion(s_emb, t_emb)
                val_loss += ((1.0 - lam) * r_l + lam * h_l).item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)

        history["epoch"].append(epoch)
        history["recon_loss"].append(avg_recon)
        history["hint_loss"].append(avg_hint)
        history["total_loss"].append(avg_total)
        history["lambda"].append(lam)
        history["val_loss"].append(avg_val)

        if verbose:
            print(f"  [TSRC] Epoch {epoch:3d}/{epochs}  "
                  f"lam={lam:.3f}  recon={avg_recon:.4f}  "
                  f"hint={avg_hint:.4f}  total={avg_total:.4f}  val={avg_val:.4f}")

        if avg_val < best_val - 1e-6:
            best_val   = avg_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [TSRC] Early stopping at epoch {epoch}.")
                break

    student.eval()
    if history_path:
        _save_history_csv(history, history_path)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# CSV history saver
# ─────────────────────────────────────────────────────────────────────────────

def _save_history_csv(history: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys   = list(history.keys())
    rows   = list(zip(*[history[k] for k in keys]))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)
    print(f"  [trainer] Training history saved -> {path}")
