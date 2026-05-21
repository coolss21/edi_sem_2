"""
preprocessing.py - Robust preprocessing for real-world time-series data.
Handles missing values, outlier clipping, normalization, and anomaly masks.
"""

import warnings
import numpy as np


class RobustTimeSeriesPreprocessor:
    """
    Fit on training data, then transform train/test consistently.

    Parameters
    ----------
    normalization      : "zscore" | "robust"
    missing_strategy   : "linear_interpolation" | "forward_fill" | "mean_fill"
    clip_outliers      : bool
    clip_quantile      : float  (e.g. 0.01 clips bottom 1% and top 1%)
    keep_anomaly_mask  : bool   (store binary mask of imputed/clipped positions)
    """

    def __init__(
        self,
        normalization: str = "zscore",
        missing_strategy: str = "linear_interpolation",
        clip_outliers: bool = True,
        clip_quantile: float = 0.01,
        keep_anomaly_mask: bool = False,
    ):
        self.normalization     = normalization
        self.missing_strategy  = missing_strategy
        self.clip_outliers     = clip_outliers
        self.clip_quantile     = clip_quantile
        self.keep_anomaly_mask = keep_anomaly_mask

        # Fitted statistics (computed on train)
        self._center = None      # mean or median  shape (T,) or scalar
        self._scale  = None      # std  or IQR      shape (T,) or scalar
        self._clip_lo = None
        self._clip_hi = None
        self._global_mean = None  # fallback fill value
        self.anomaly_mask_ = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "RobustTimeSeriesPreprocessor":
        """
        Compute statistics from X (N, T, 1).  NaN values are ignored.
        """
        assert X.ndim == 3 and X.shape[2] == 1, \
            f"Expected shape (N, T, 1), got {X.shape}"

        vals = X[:, :, 0]  # (N, T)

        # Global mean for entire-sample fallback
        self._global_mean = float(np.nanmean(vals))

        if self.normalization == "zscore":
            self._center = np.nanmean(vals, axis=0)          # (T,)
            self._scale  = np.nanstd(vals,  axis=0)
        elif self.normalization == "robust":
            self._center = np.nanmedian(vals, axis=0)        # (T,)
            q75 = np.nanpercentile(vals, 75, axis=0)
            q25 = np.nanpercentile(vals, 25, axis=0)
            self._scale  = q75 - q25
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        # Avoid division by zero
        self._scale = np.where(self._scale < 1e-8, 1.0, self._scale)

        # Clip bounds computed on raw training values (all time steps flattened)
        flat = vals.flatten()
        finite_vals = flat[np.isfinite(flat)]
        if len(finite_vals) > 0 and self.clip_outliers:
            self._clip_lo = float(np.percentile(finite_vals, self.clip_quantile * 100))
            self._clip_hi = float(np.percentile(finite_vals, (1 - self.clip_quantile) * 100))
        else:
            self._clip_lo = -np.inf
            self._clip_hi =  np.inf

        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to X (N, T, 1).
        Returns float32 array of same shape.
        """
        assert self._center is not None, "Call fit() before transform()."
        assert X.ndim == 3 and X.shape[2] == 1

        X = X.copy().astype(np.float32)
        N, T, _ = X.shape
        vals = X[:, :, 0]  # (N, T) view

        # 1. Track original NaN positions
        nan_mask = np.isnan(vals)

        # 2. Handle missing values per sample
        for i in range(N):
            row = vals[i]
            if np.all(np.isnan(row)):
                warnings.warn(
                    f"[preprocessing] Sample {i} is entirely NaN – replacing with zeros."
                )
                vals[i] = 0.0
                continue
            if np.any(np.isnan(row)):
                row = self._fill_missing(row)
                vals[i] = row

        # 3. Outlier clipping
        clip_mask = np.zeros_like(vals, dtype=bool)
        if self.clip_outliers:
            lo, hi = self._clip_lo, self._clip_hi
            clip_mask = (vals < lo) | (vals > hi)
            vals = np.clip(vals, lo, hi)

        # 4. Normalise
        vals = (vals - self._center) / self._scale

        X[:, :, 0] = vals

        # 5. Anomaly mask
        if self.keep_anomaly_mask:
            self.anomaly_mask_ = (nan_mask | clip_mask).astype(np.uint8)

        return X.astype(np.float32)

    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    def _fill_missing(self, row: np.ndarray) -> np.ndarray:
        """Fill NaN values in a 1-D time series."""
        method = self.missing_strategy

        # ── linear interpolation ──────────────────────────────────────
        if method == "linear_interpolation":
            row = _linear_interp(row)
            if np.any(np.isnan(row)):
                row = _forward_fill(row)
            if np.any(np.isnan(row)):
                row = _mean_fill(row, self._global_mean)
            return row

        # ── forward fill ──────────────────────────────────────────────
        elif method == "forward_fill":
            row = _forward_fill(row)
            if np.any(np.isnan(row)):
                row = _mean_fill(row, self._global_mean)
            return row

        # ── mean fill ────────────────────────────────────────────────
        elif method == "mean_fill":
            return _mean_fill(row, self._global_mean)

        else:
            raise ValueError(f"Unknown missing_strategy: {method}")

    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        """Return a JSON-serialisable dict of fitted parameters."""
        return {
            "normalization":    self.normalization,
            "missing_strategy": self.missing_strategy,
            "clip_outliers":    self.clip_outliers,
            "clip_quantile":    self.clip_quantile,
            "clip_lo":          float(self._clip_lo) if self._clip_lo is not None else None,
            "clip_hi":          float(self._clip_hi) if self._clip_hi is not None else None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _linear_interp(row: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D array."""
    row = row.copy()
    nans = np.isnan(row)
    if not nans.any():
        return row
    idx = np.arange(len(row))
    row[nans] = np.interp(idx[nans], idx[~nans], row[~nans])
    return row


def _forward_fill(row: np.ndarray) -> np.ndarray:
    """Forward fill NaN values."""
    row = row.copy()
    for i in range(1, len(row)):
        if np.isnan(row[i]):
            row[i] = row[i - 1]
    return row


def _mean_fill(row: np.ndarray, global_mean: float) -> np.ndarray:
    """Replace remaining NaNs with global training mean."""
    row = row.copy()
    local_mean = np.nanmean(row)
    fill_val = local_mean if np.isfinite(local_mean) else global_mean
    row[np.isnan(row)] = fill_val
    return row
