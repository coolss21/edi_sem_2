"""
augmentations.py - Real time-series augmentations for contrastive learning.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Primitive augmentations
# ─────────────────────────────────────────────────────────────────────────────

def jitter(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Add Gaussian noise to x.  x shape: (N, T, 1) or (T, 1)."""
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)


def scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Multiply each sample by a random scalar near 1."""
    if x.ndim == 3:
        factor = np.random.normal(1.0, sigma, (x.shape[0], 1, 1)).astype(np.float32)
    else:
        factor = float(np.random.normal(1.0, sigma))
    return x * factor


def random_mask(x: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:
    """Randomly zero out mask_ratio of time steps."""
    x = x.copy()
    T = x.shape[-2]
    n_mask = max(1, int(T * mask_ratio))
    if x.ndim == 3:
        for i in range(x.shape[0]):
            idx = np.random.choice(T, n_mask, replace=False)
            x[i, idx, :] = 0.0
    else:
        idx = np.random.choice(T, n_mask, replace=False)
        x[idx, :] = 0.0
    return x


def random_crop_resize(x: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
    """
    Crop a contiguous window of length floor(T * crop_ratio)
    and resize it back to T using linear interpolation.
    """
    x = x.copy()
    T = x.shape[-2]
    crop_len = max(2, int(T * crop_ratio))

    if x.ndim == 3:
        out = np.zeros_like(x)
        for i in range(x.shape[0]):
            start = np.random.randint(0, T - crop_len + 1)
            crop  = x[i, start:start + crop_len, 0]
            resized = np.interp(
                np.linspace(0, crop_len - 1, T),
                np.arange(crop_len),
                crop,
            ).astype(np.float32)
            out[i, :, 0] = resized
        return out
    else:
        start = np.random.randint(0, T - crop_len + 1)
        crop  = x[start:start + crop_len, 0]
        resized = np.interp(
            np.linspace(0, crop_len - 1, T),
            np.arange(crop_len),
            crop,
        ).astype(np.float32)
        out = x.copy()
        out[:, 0] = resized
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Composite augmentation policies
# ─────────────────────────────────────────────────────────────────────────────

def weak_augment(x: np.ndarray, jitter_sigma: float = 0.01,
                 mask_ratio: float = 0.05) -> np.ndarray:
    """Mild augmentation: small jitter + small masking."""
    x = jitter(x, sigma=jitter_sigma)
    x = random_mask(x, mask_ratio=mask_ratio)
    return x


def medium_augment(x: np.ndarray, jitter_sigma: float = 0.03,
                   mask_ratio: float = 0.1) -> np.ndarray:
    """Medium augmentation: jitter + masking + crop-resize."""
    x = jitter(x, sigma=jitter_sigma)
    x = random_mask(x, mask_ratio=mask_ratio)
    x = random_crop_resize(x, crop_ratio=0.85)
    return x


def strong_augment(x: np.ndarray, jitter_sigma: float = 0.05,
                   mask_ratio: float = 0.2) -> np.ndarray:
    """Strong augmentation: large jitter + masking + scaling + crop-resize."""
    x = jitter(x, sigma=jitter_sigma)
    x = random_mask(x, mask_ratio=mask_ratio)
    x = scaling(x, sigma=0.15)
    x = random_crop_resize(x, crop_ratio=0.75)
    return x


def apply_augmentation(
    x: np.ndarray,
    strength: str = "medium",
    jitter_sigma: float = 0.03,
    mask_ratio: float = 0.1,
) -> np.ndarray:
    """
    Dispatch to weak / medium / strong based on `strength`.
    Used by teacher contrastive training.
    """
    if strength == "weak":
        return weak_augment(x, jitter_sigma=jitter_sigma, mask_ratio=mask_ratio)
    elif strength == "medium":
        return medium_augment(x, jitter_sigma=jitter_sigma, mask_ratio=mask_ratio)
    elif strength == "strong":
        return strong_augment(x, jitter_sigma=jitter_sigma, mask_ratio=mask_ratio)
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")
