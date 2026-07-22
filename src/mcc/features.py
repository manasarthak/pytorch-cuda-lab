"""Hand-crafted features for classical models (Rung 1).

Each preprocessed curve is a length-``roi_len`` peak-centered derivative. These
features summarize its shape the way a melt-curve analyst would: where the peak
is, how tall/wide/skewed it is, how much area it covers, how many peaks there are.
Good features let a tree model (XGBoost) reach a strong baseline cheaply.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths


def _fwhm(curve: np.ndarray, peak_idx: int) -> float:
    try:
        widths, *_ = peak_widths(curve, [peak_idx], rel_height=0.5)
        return float(widths[0])
    except Exception:
        return 0.0


def curve_features(curve: np.ndarray) -> dict[str, float]:
    """Feature dict for a single curve."""
    x = np.arange(curve.shape[0], dtype=np.float64)
    total = curve.sum()
    peak_idx = int(np.argmax(curve))
    peak_val = float(curve[peak_idx])

    # Normalized distribution for moment-based shape stats.
    p = curve - curve.min()
    p_sum = p.sum()
    if p_sum > 0:
        w = p / p_sum
        mean = float((w * x).sum())
        var = float((w * (x - mean) ** 2).sum())
        std = np.sqrt(var) if var > 0 else 0.0
        skew = float((w * (x - mean) ** 3).sum() / (std**3)) if std > 0 else 0.0
        kurt = float((w * (x - mean) ** 4).sum() / (std**4)) if std > 0 else 0.0
    else:
        mean = std = skew = kurt = 0.0

    peaks, props = find_peaks(curve, height=peak_val * 0.25)

    return {
        "peak_idx": float(peak_idx),
        "peak_val": peak_val,
        "auc": float(np.trapezoid(curve)),
        "sum": float(total),
        "mean_pos": mean,
        "std_pos": std,
        "skew": skew,
        "kurtosis": kurt,
        "fwhm": _fwhm(curve, peak_idx),
        "n_peaks": float(len(peaks)),
        "curve_min": float(curve.min()),
        "left_area": float(np.trapezoid(curve[:peak_idx])) if peak_idx > 0 else 0.0,
        "right_area": float(np.trapezoid(curve[peak_idx:])),
    }


def feature_matrix(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Turn ``(N, roi_len)`` curves into an ``(N, n_features)`` matrix."""
    rows = [curve_features(c) for c in X]
    names = list(rows[0].keys())
    mat = np.array([[r[n] for n in names] for r in rows], dtype=np.float32)
    return mat, names
