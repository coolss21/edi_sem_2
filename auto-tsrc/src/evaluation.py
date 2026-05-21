"""
evaluation.py - Clustering, reconstruction, and SVM probe evaluation utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run model.encode(x) on entire X and return embeddings as numpy.

    X shape: (N, T, 1)
    Returns: (N, embedding_dim)
    """
    model.eval()
    model = model.to(device)
    all_embs = []
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            z  = model.encode(xb)
            all_embs.append(z.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# KMeans clustering
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    """Fit KMeans on embeddings and return cluster labels."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_clustering_metrics(
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    embeddings: np.ndarray,
) -> dict:
    """
    Compute ARI, NMI, CHI, Silhouette.
    Returns dict with all metrics (silhouette=None if not applicable).
    """
    ari = float(adjusted_rand_score(y_true, cluster_labels))
    nmi = float(normalized_mutual_info_score(y_true, cluster_labels,
                                              average_method="arithmetic"))

    n_clusters = len(np.unique(cluster_labels))
    if n_clusters >= 2 and n_clusters < len(embeddings):
        chi = float(calinski_harabasz_score(embeddings, cluster_labels))
    else:
        chi = 0.0

    sil = None
    if n_clusters >= 2 and n_clusters < len(embeddings):
        try:
            sil = float(silhouette_score(embeddings, cluster_labels,
                                          sample_size=min(1000, len(embeddings)),
                                          random_state=42))
        except Exception:
            sil = None

    return {
        "ARI":       ari,
        "NMI":       nmi,
        "CHI":       chi,
        "silhouette": sil,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction MSE
# ─────────────────────────────────────────────────────────────────────────────

def compute_reconstruction_mse(
    student: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Compute mean reconstruction MSE on X."""
    student.eval()
    student = student.to(device)
    total_mse = 0.0
    n_samples  = 0
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = student(xb)
            total_mse += criterion(recon, xb).item()
            n_samples  += xb.shape[0] * xb.shape[1]
    return total_mse / max(n_samples, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SVM Probe (optional)
# ─────────────────────────────────────────────────────────────────────────────

def svm_probe(
    train_emb: np.ndarray,
    y_train: np.ndarray,
    test_emb: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> float:
    """
    Train RBF-SVM on train embeddings and return test accuracy.
    Returns None if SVM fails.
    """
    try:
        scaler    = StandardScaler()
        tr_scaled = scaler.fit_transform(train_emb)
        te_scaled = scaler.transform(test_emb)
        svm = SVC(kernel="rbf", random_state=seed, max_iter=2000)
        svm.fit(tr_scaled, y_train)
        acc = float((svm.predict(te_scaled) == y_test).mean())
        return acc
    except Exception as e:
        print(f"  [evaluation] SVM probe failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline for one model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    model_name: str = "model",
    dataset_name: str = "dataset",
    run_svm: bool = True,
) -> dict:
    """
    Full evaluation: embeddings → KMeans → metrics → SVM probe.
    Returns flat dict of all metrics.
    """
    n_classes = len(np.unique(y_test))

    test_embs  = extract_embeddings(model, X_test,  device)
    train_embs = extract_embeddings(model, X_train, device)

    cluster_labels = run_kmeans(test_embs, n_classes)
    clus_metrics   = compute_clustering_metrics(y_test, cluster_labels, test_embs)

    recon_mse = None
    if hasattr(model, "forward"):
        try:
            recon_mse = compute_reconstruction_mse(model, X_test, device)
        except Exception:
            recon_mse = None

    svm_acc = None
    if run_svm:
        svm_acc = svm_probe(train_embs, y_train, test_embs, y_test)

    return {
        "dataset":           dataset_name,
        "model_name":        model_name,
        "ARI":               clus_metrics["ARI"],
        "NMI":               clus_metrics["NMI"],
        "CHI":               clus_metrics["CHI"],
        "silhouette":        clus_metrics["silhouette"],
        "reconstruction_mse": recon_mse,
        "svm_accuracy":      svm_acc,
    }
