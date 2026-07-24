"""Tests for the preprocessing recipe — no S3 or torch required."""

from __future__ import annotations

import numpy as np

from gpulab.data.preprocess import (
    FIXED_WINDOW,
    PreprocessConfig,
    center_peaks,
    output_len,
    positive_mask,
    preprocess_curves,
    preprocess_positive_curves,
)

FIXED_LEN = FIXED_WINDOW[1] - FIXED_WINDOW[0]  # 120


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


def test_default_is_fixed_unaligned_window():
    # The default ROI is the fixed window (350, 470) -> length 120, unaligned,
    # so Tm/peak position is preserved.
    cfg = PreprocessConfig()
    assert cfg.roi == FIXED_WINDOW
    assert output_len(cfg) == FIXED_LEN
    X = preprocess_curves(_synthetic_raw(), cfg)
    assert X.shape == (8, FIXED_LEN)


def test_fixed_window_keeps_peak_position():
    # Two curve sets with different Tm must place their peak at different columns
    # (this is the whole point of NOT aligning).
    early = _synthetic_raw(peak_frame=395)
    late = _synthetic_raw(peak_frame=445)
    Xe = preprocess_curves(early, PreprocessConfig(normalization=None))
    Xl = preprocess_curves(late, PreprocessConfig(normalization=None))
    assert Xe.argmax(axis=1).mean() < Xl.argmax(axis=1).mean()


def test_peak_centered_mode_aligns_to_length_61():
    # roi=None -> org-pipeline peak-centered behavior, length slice_length (61).
    cfg = PreprocessConfig(roi=None, normalization=None)
    assert output_len(cfg) == 61
    X = preprocess_curves(_synthetic_raw(), cfg)
    assert X.shape == (8, 61)
    # Aligned: every curve's peak sits near the center column (30).
    assert np.allclose(X.argmax(axis=1), 30, atol=3)


def test_auc_normalization_gives_unit_area():
    X = preprocess_curves(_synthetic_raw(), PreprocessConfig(normalization="auc"))
    assert X.shape == (8, FIXED_LEN)
    areas = np.trapezoid(X, axis=1)
    assert np.allclose(areas, 1.0, atol=1e-6)


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
    assert X.shape[1] == FIXED_LEN
