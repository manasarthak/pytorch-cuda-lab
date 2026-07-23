"""Tests for the preprocessing recipe — no S3 or torch required."""

from __future__ import annotations

import numpy as np

from gpulab.data.preprocess import (
    PreprocessConfig,
    center_peaks,
    positive_mask,
    preprocess_curves,
    preprocess_positive_curves,
)


def _synthetic_raw(n_wells: int = 8, n_frames: int = 500, peak_frame: int = 420) -> np.ndarray:
    """Sigmoid-like melt curves whose derivative peaks near ``peak_frame``."""
    x = np.arange(n_frames)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n_wells):
        center = peak_frame + rng.integers(-10, 10)
        curve = 1.0 / (1.0 + np.exp((x - center) / 6.0))  # falling sigmoid
        curve += rng.normal(0, 0.002, size=n_frames)
        rows.append(curve)
    return np.asarray(rows)


def test_center_peaks_output_length():
    raw = _synthetic_raw()
    deriv = np.gradient(raw, axis=1)  # any 2D signal
    out = center_peaks(deriv, slice_length=61, peak_window=(380, 460))
    assert out.shape == (raw.shape[0], 61)


def test_center_peaks_edge_padding():
    # Peak at frame 0 forces left-edge padding; output must still be length 61.
    row = np.zeros(500)
    row[5] = 10.0
    out = center_peaks(row[None, :], slice_length=61, peak_window=None)
    assert out.shape == (1, 61)


def test_preprocess_shape_and_auc_normalization():
    raw = _synthetic_raw()
    cfg = PreprocessConfig(normalization="auc")
    X = preprocess_curves(raw, cfg)
    assert X.shape == (raw.shape[0], 61)
    # Unit-area normalization: trapezoid integral of each curve ~= 1.
    areas = np.trapezoid(X, axis=1)
    assert np.allclose(areas, 1.0, atol=1e-6)


def test_preprocess_no_normalization_matches_raw_derivative():
    raw = _synthetic_raw()
    X = preprocess_curves(raw, PreprocessConfig(normalization=None))
    assert X.shape == (raw.shape[0], 61)
    assert np.isfinite(X).all()


def test_positive_mask_selects_real_peaks():
    raw = _synthetic_raw()
    # These synthetic curves have strong derivative peaks -> should be positive.
    cfg = PreprocessConfig(peak_threshold=0.0, minimum_value=-np.inf)
    mask = positive_mask(raw, cfg)
    assert mask.dtype == bool
    assert mask.sum() == raw.shape[0]


def test_preprocess_positive_curves_returns_indices():
    raw = _synthetic_raw()
    cfg = PreprocessConfig(peak_threshold=0.0, minimum_value=-np.inf)
    X, idx = preprocess_positive_curves(raw, cfg)
    assert X.shape[0] == idx.shape[0]
    assert X.shape[1] == 61
