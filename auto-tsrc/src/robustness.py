"""
robustness.py - Corrupt test sets and evaluate model robustness.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn

from .evaluation import (
    extract_embeddings,
    run_kmeans,
    compute_clustering_metrics,
    compute_reconstruction_mse,
)


# ─────────────────────────────────────────────────────────────────────────────
# Corruption functions
# ─────────────────────────────────────────────────────────────────────────────

def corrupt_missing_random(X: np.ndarray, missing_ratio: float = 0.1,
                            seed: int = 0) -> np.ndarray:
    """Randomly set `missing_ratio` of values to NaN."""
    rng  = np.random.default_rng(seed)
    X_c  = X.copy()
    N, T, C = X_c.shape
    n_missing = max(1, int(T * missing_ratio))
    for i in range(N):
        idx = rng.choice(T, n_missing, replace=False)
        X_c[i, idx, :] = np.nan
    return X_c


def corrupt_spike_noise(X: np.ndarray, spike_ratio: float = 0.05,
                         spike_scale: float = 5.0, seed: int = 0) -> np.ndarray:
    """Add large spikes to `spike_ratio` of time steps."""
    rng  = np.random.default_rng(seed)
    X_c  = X.copy()
    N, T, C = X_c.shape
    n_spike = max(1, int(T * spike_ratio))
    global_std = float(np.nanstd(X_c))
    for i in range(N):
        idx = rng.choice(T, n_spike, replace=False)
        signs = rng.choice([-1, 1], size=n_spike)
        X_c[i, idx, 0] += signs * spike_scale * global_std
    return X_c


def corrupt_gaussian_noise(X: np.ndarray, sigma: float = 0.05,
                            seed: int = 0) -> np.ndarray:
    """Add Gaussian noise with std = sigma."""
    rng = np.random.default_rng(seed)
    return (X + rng.normal(0, sigma, X.shape)).astype(np.float32)


def corrupt_outlier_burst(X: np.ndarray, burst_len: int = None,
                           burst_scale: float = 6.0, seed: int = 0) -> np.ndarray:
    """Add a contiguous burst anomaly to each sample."""
    rng  = np.random.default_rng(seed)
    X_c  = X.copy()
    N, T, C = X_c.shape
    if burst_len is None:
        burst_len = max(1, T // 10)
    global_std = float(np.nanstd(X_c))
    for i in range(N):
        start = rng.integers(0, max(1, T - burst_len))
        X_c[i, start:start + burst_len, 0] += burst_scale * global_std
    return X_c


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helper (re-use fitted preprocessor, then replace NaN)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_corrupted(X_corrupted: np.ndarray, preprocessor) -> np.ndarray:
    """Apply the already-fitted preprocessor to corrupted data."""
    if preprocessor is not None:
        return preprocessor.transform(X_corrupted)
    # Fallback: just replace NaN with 0
    X_c = X_corrupted.copy()
    X_c[np.isnan(X_c)] = 0.0
    return X_c.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Build corruption suite
# ─────────────────────────────────────────────────────────────────────────────

def build_corruption_suite(X_test: np.ndarray, seed: int = 42) -> dict:
    """
    Return dict of {corruption_name: corrupted_X} pairs.
    """
    return {
        "clean":          X_test.copy(),
        "missing_10":     corrupt_missing_random(X_test, 0.10, seed),
        "missing_20":     corrupt_missing_random(X_test, 0.20, seed),
        "spike_noise":    corrupt_spike_noise(X_test, 0.05, 5.0, seed),
        "gaussian_0.05":  corrupt_gaussian_noise(X_test, 0.05, seed),
        "gaussian_0.1":   corrupt_gaussian_noise(X_test, 0.10, seed),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate single model on corruption suite
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_robustness(
    model: nn.Module,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor,
    device: torch.device,
    dataset: str,
    seed: int = 42,
) -> list:
    """
    Evaluate model on all corruptions.
    Returns list of metric dicts.
    """
    n_classes   = len(np.unique(y_test))
    suite       = build_corruption_suite(X_test, seed)
    results     = []

    # Clean baseline metrics
    X_clean_pp  = _preprocess_corrupted(suite["clean"], preprocessor)
    clean_embs  = extract_embeddings(model, X_clean_pp, device)
    clean_km    = run_kmeans(clean_embs, n_classes, seed)
    clean_met   = compute_clustering_metrics(y_test, clean_km, clean_embs)
    clean_mse   = _safe_recon_mse(model, X_clean_pp, device)

    for corr_name, X_corr in suite.items():
        X_pp   = _preprocess_corrupted(X_corr, preprocessor)
        embs   = extract_embeddings(model, X_pp, device)
        km     = run_kmeans(embs, n_classes, seed)
        met    = compute_clustering_metrics(y_test, km, embs)
        mse    = _safe_recon_mse(model, X_pp, device)

        ari_drop = clean_met["ARI"] - met["ARI"]
        nmi_drop = clean_met["NMI"] - met["NMI"]

        results.append({
            "dataset":                 dataset,
            "model":                   model_name,
            "corruption_type":         corr_name,
            "ARI":                     met["ARI"],
            "NMI":                     met["NMI"],
            "CHI":                     met["CHI"],
            "reconstruction_mse":      mse,
            "ari_drop_from_clean":     ari_drop,
            "nmi_drop_from_clean":     nmi_drop,
        })

    return results


def _safe_recon_mse(model, X, device):
    try:
        return compute_reconstruction_mse(model, X, device)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Save robustness CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_robustness_csv(rows: list, path: str):
    """Save robustness results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[robustness] Saved -> {path}")
