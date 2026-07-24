"""Melt-curve preprocessing — a standalone reimplementation of the org pipeline.

The transform, applied to raw fluorescence curves of shape ``(n_wells, n_frames)``:

1. Savitzky-Golay smooth of the raw signal.
2. Negative first derivative (``-gradient``) -> melt transitions become positive peaks.
3. Second Savitzky-Golay smooth, on the derivative.
4. ROI crop. Two modes:
   * **Fixed, unaligned window** (the default, ``roi=(350, 470)``): a plain slice
     that leaves every peak at its true position, so the melt temperature (Tm =
     peak location) stays a discriminative feature for the model.
   * **Peak-centered window** (``roi=None``): the org-pipeline behavior — a
     length-``slice_length`` window centered on each peak. This aligns the curves
     and therefore DISCARDS Tm, leaving only peak shape.
5. Optional normalization: divide each curve by its trapezoidal area (unit AUC),
   which removes amplitude differences while preserving Tm and peak shape.

Positive-well *mining* runs the same transform, then keeps a well only if it has a
real melt peak in the target window AND its centered derivative never dips below
``minimum_value``. (Mining always uses peak-centering internally for the negative-
value check; it is independent of the model-input ROI chosen above.)

The math is pure numpy/scipy so it can be unit-tested and diffed against a GPU port.
If you want to exactly reproduce the org pipeline, set ``roi=None`` (length-61,
peak-aligned) which matches ``center_peaks(..., 61, (380, 460))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.signal import savgol_filter

# --- Constants (change only if the raw framing changes) ---
PEAK_WINDOW: tuple[int, int] = (380, 460)  # frames where the melt peak is expected
PEAK_THRESHOLD: float = 4.0                # min derivative peak to count as a positive
SLICE_LENGTH: int = 61                     # length of the peak-centered ROI (alignment mode)

# Default model-input ROI: a FIXED, UNALIGNED window (length 120) around the peak
# region. Unlike peak-centering, this keeps each curve's peak at its true position,
# so the melt temperature (Tm = peak location) survives as a discriminative feature.
# The window brackets PEAK_WINDOW (380-460) with margin on both sides.
FIXED_WINDOW: tuple[int, int] = (350, 470)


@dataclass(frozen=True)
class PreprocessConfig:
    raw_smooth_window: int = 13
    raw_smooth_polyorder: int = 2
    deriv_smooth_window: int = 13
    deriv_smooth_polyorder: int = 2
    peak_window: tuple[int, int] = PEAK_WINDOW
    peak_threshold: float = PEAK_THRESHOLD
    slice_length: int = SLICE_LENGTH
    # None -> raw derivative values; "auc" -> unit-area normalized (keeps Tm + shape).
    normalization: Optional[Literal["auc"]] = "auc"
    # ROI for model input:
    #   (start, end) -> fixed unaligned slice; KEEPS Tm/peak position (the default).
    #   None         -> peak-centered length-`slice_length` window; DISCARDS Tm,
    #                   leaving only peak shape (the org-pipeline behavior).
    roi: Optional[tuple[int, int]] = FIXED_WINDOW
    # Required for mining (positive-well selection); may stay None for plain preprocessing.
    minimum_value: Optional[float] = 0.0


def output_len(cfg: PreprocessConfig) -> int:
    """Number of columns ``preprocess_curves`` produces for this config."""
    return (cfg.roi[1] - cfg.roi[0]) if cfg.roi is not None else cfg.slice_length


def center_peaks(
    data: np.ndarray,
    slice_length: int = SLICE_LENGTH,
    peak_window: Optional[tuple[int, int]] = PEAK_WINDOW,
) -> np.ndarray:
    """Extract a ``slice_length`` window centered on each row's peak.

    The peak is the argmax within ``peak_window`` (or the global argmax if None).
    Windows running off either end are edge-padded, so output is always exactly
    ``(n_rows, slice_length)``.
    """
    half = slice_length // 2
    n_rows, row_len = data.shape
    out = np.zeros((n_rows, slice_length), dtype=data.dtype)

    for i, row in enumerate(data):
        if peak_window is not None:
            lo, hi = peak_window
            max_index = lo + int(np.argmax(row[lo:hi]))
        else:
            max_index = int(np.argmax(row))
        start = max_index - half
        end = max_index + slice_length - half
        clipped = row[max(0, start):min(row_len, end)]
        pad_left = -start if start < 0 else 0
        pad_right = end - row_len if end > row_len else 0
        padded = np.pad(clipped, (pad_left, pad_right), mode="edge")
        if padded.shape[0] != slice_length:
            raise ValueError(f"row {i}: got length {padded.shape[0]}, expected {slice_length}")
        out[i] = padded
    return out


def smoothed_negative_derivative(raw: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Steps 1-3: smooth, negative gradient, smooth again. Returns ``(n_wells, n_frames)``."""
    smooth = savgol_filter(raw, cfg.raw_smooth_window, cfg.raw_smooth_polyorder, axis=1)
    deriv = -np.gradient(smooth, axis=1)
    return savgol_filter(deriv, cfg.deriv_smooth_window, cfg.deriv_smooth_polyorder, axis=1)


def preprocess_curves(raw: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Full transform (steps 1-5). Input ``(n_wells, n_frames)`` -> ``(n_wells, roi_len)``."""
    raw = np.asarray(raw, dtype=np.float64)
    deriv = smoothed_negative_derivative(raw, cfg)

    if cfg.roi is not None:
        roi = deriv[:, cfg.roi[0]:cfg.roi[1]]
    else:
        roi = center_peaks(deriv, cfg.slice_length, cfg.peak_window)

    if cfg.normalization is None:
        return roi
    if cfg.normalization == "auc":
        auc = np.trapezoid(roi, axis=1)
        return roi / auc[:, np.newaxis]
    raise ValueError(f"Unknown normalization: {cfg.normalization!r}")


def positive_mask(raw: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Boolean mask of wells that count as positive samples.

    A well is positive iff its derivative peak inside ``peak_window`` exceeds
    ``peak_threshold`` AND its peak-centered derivative never falls below
    ``minimum_value``. ``minimum_value`` must be set.
    """
    if cfg.minimum_value is None:
        raise ValueError("minimum_value must be set for positive-well mining.")
    raw = np.asarray(raw, dtype=np.float64)
    deriv = smoothed_negative_derivative(raw, cfg)

    lo, hi = cfg.peak_window
    has_peak = deriv[:, lo:hi].max(axis=1) > cfg.peak_threshold
    centered = center_peaks(deriv, cfg.slice_length, cfg.peak_window)
    not_too_negative = centered.min(axis=1) >= cfg.minimum_value
    return has_peak & not_too_negative


def preprocess_positive_curves(
    raw: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """Mine positive wells and preprocess them.

    Returns ``(X, indices)`` where ``X`` is ``(n_positive, roi_len)`` and ``indices``
    are the original well row indices that were kept.
    """
    mask = positive_mask(raw, cfg)
    indices = np.flatnonzero(mask)
    X = preprocess_curves(raw[indices], cfg) if indices.size else np.empty((0, output_len(cfg)))
    return X, indices
