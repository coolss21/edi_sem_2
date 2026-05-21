"""
explainability.py - Visualisation outputs for Auto-TSRC.
Uses matplotlib only (no seaborn).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_reconstructions(student: nn.Module, X: np.ndarray,
                          device: torch.device, batch_size: int = 64):
    """Return (original, reconstruction) as numpy arrays."""
    student.eval()
    all_orig  = []
    all_recon = []
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = student(xb)
            all_orig.append(xb.cpu().numpy())
            all_recon.append(recon.cpu().numpy())
    return np.concatenate(all_orig, 0), np.concatenate(all_recon, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reconstruction overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstruction_overlay(
    student: nn.Module,
    X_test: np.ndarray,
    device: torch.device,
    dataset: str,
    plots_dir: str,
    n_samples: int = 3,
    seed: int = 42,
) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    rng    = np.random.default_rng(seed)
    idx    = rng.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    orig, recon = _get_reconstructions(student, X_test[idx], device)

    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples),
                              facecolor="#1a1a2e")
    if n_samples == 1:
        axes = [axes]

    colors_orig  = "#4fc3f7"
    colors_recon = "#f48fb1"

    for i, ax in enumerate(axes):
        t = np.arange(orig.shape[1])
        ax.set_facecolor("#0d1117")
        ax.plot(t, orig[i, :, 0],  color=colors_orig,  lw=1.8,
                label="Original",      alpha=0.9)
        ax.plot(t, recon[i, :, 0], color=colors_recon, lw=1.8,
                label="Reconstructed", alpha=0.9, linestyle="--")
        ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
        ax.set_title(f"Sample {idx[i]}", color="white", fontsize=10)
        ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.suptitle(f"Reconstruction Overlay - {dataset}",
                 color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(plots_dir, f"{dataset}_reconstruction_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"[explainability] Saved -> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reconstruction error heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstruction_error_heatmap(
    student: nn.Module,
    X_test: np.ndarray,
    device: torch.device,
    dataset: str,
    plots_dir: str,
    max_samples: int = 50,
) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    orig, recon = _get_reconstructions(student, X_test[:max_samples], device)
    abs_err = np.abs(orig[:, :, 0] - recon[:, :, 0])  # (N, T)

    fig, ax = plt.subplots(figsize=(12, max(4, abs_err.shape[0] // 3)),
                            facecolor="#1a1a2e")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(abs_err, aspect="auto", cmap="inferno", origin="upper",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.set_ylabel("Abs Error", color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_xlabel("Time Step", color="white", fontsize=11)
    ax.set_ylabel("Sample Index", color="white", fontsize=11)
    ax.set_title(f"Reconstruction Error Heatmap - {dataset}",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    fig.tight_layout()

    path = os.path.join(plots_dir, f"{dataset}_error_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"[explainability] Saved -> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Embedding PCA scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_embedding_pca(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    plots_dir: str,
    title_suffix: str = "",
) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    n_comp = min(2, embeddings.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    emb2d  = pca.fit_transform(embeddings)

    classes = np.unique(y_true)
    cmap    = cm.get_cmap("tab10", max(len(classes), 10))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#0d1117")
    for i, cls in enumerate(classes):
        mask = y_true == cls
        ax.scatter(
            emb2d[mask, 0],
            emb2d[mask, 1] if n_comp > 1 else np.zeros(mask.sum()),
            color=cmap(i), label=f"Class {cls}", s=28, alpha=0.8,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                  color="white", fontsize=11)
    if n_comp > 1:
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                      color="white", fontsize=11)
    else:
        ax.set_ylabel("", color="white")
    ax.set_title(f"Embedding PCA - {dataset} {title_suffix}",
                 color="white", fontsize=13)
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    fig.tight_layout()

    path = os.path.join(plots_dir, f"{dataset}_embedding_pca.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"[explainability] Saved -> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cluster prototypes
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_prototypes(
    student: nn.Module,
    X_test: np.ndarray,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    device: torch.device,
    dataset: str,
    plots_dir: str,
) -> str:
    os.makedirs(plots_dir, exist_ok=True)
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(n_clusters, 1,
                              figsize=(10, 3 * n_clusters),
                              facecolor="#1a1a2e")
    if n_clusters == 1:
        axes = [axes]

    for ax, cid in zip(axes, unique_clusters):
        mask     = cluster_labels == cid
        idx_mask = np.where(mask)[0]
        centroid = embeddings[mask].mean(axis=0)
        dists    = np.linalg.norm(embeddings[mask] - centroid, axis=1)
        proto_i  = idx_mask[np.argmin(dists)]
        ts       = X_test[proto_i, :, 0]
        t        = np.arange(len(ts))

        ax.set_facecolor("#0d1117")
        ax.plot(t, ts, color="#69d2e7", lw=1.8)
        ax.set_title(f"Cluster {cid} prototype (sample {proto_i})",
                     color="white", fontsize=10)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    fig.suptitle(f"Cluster Prototypes - {dataset}",
                 color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(plots_dir, f"{dataset}_cluster_prototypes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"[explainability] Saved -> {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all explainability plots
# ─────────────────────────────────────────────────────────────────────────────

def run_all_explainability(
    student: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    device: torch.device,
    dataset: str,
    plots_dir: str,
):
    """Run all 4 explainability plots."""
    plot_reconstruction_overlay(student, X_test, device, dataset, plots_dir)
    plot_reconstruction_error_heatmap(student, X_test, device, dataset, plots_dir)
    plot_embedding_pca(embeddings, y_test, dataset, plots_dir, title_suffix="(Auto-TSRC)")
    plot_cluster_prototypes(student, X_test, embeddings, cluster_labels,
                            device, dataset, plots_dir)
