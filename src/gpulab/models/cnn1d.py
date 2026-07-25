"""A small from-scratch 1D-CNN over the derivative curve (Rung 2).

Deliberately simple: the goal of this rung is to learn the PyTorch/CUDA mechanics
(tensors, devices, the training loop, AMP), not to chase accuracy. Input is a
curve of shape ``(batch, 1, roi_len)`` -- channels-first, as Conv1d expects.

Head choice matters for what the model can use:

* ``head="flatten"`` (default) keeps absolute position. Pair it with the fixed,
  unaligned ROI (``roi=(350, 470)``) so the model can read the melt temperature
  (Tm = peak location) as a feature. Requires a fixed ``roi_len``.
* ``head="avg"`` uses global average pooling, which is translation-invariant: it
  DISCARDS where the peak is and sees only its shape. Only sensible with the
  peak-centered ROI (``roi=None``), where every curve is already aligned.

If you feed the fixed, unaligned window into an ``avg`` head, you throw the Tm
information away inside the model -- the exact thing the fixed window preserved.

Requires torch: ``uv sync --extra torch``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(
        self,
        n_classes: int,
        roi_len: int = 120,
        width: int = 32,
        head: Literal["flatten", "avg"] = "flatten",
    ):
        super().__init__()
        self.head_kind = head
        self.features = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
        )
        feat_channels = width * 2
        if head == "flatten":
            # Conv padding preserves length, so features are (B, C, roi_len).
            # Flatten keeps position -> the classifier can key on Tm.
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_channels * roi_len, n_classes),
            )
        elif head == "avg":
            # Global average pool over time -> (B, C): position-invariant.
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(feat_channels, n_classes),
            )
        else:
            raise ValueError(f"head must be 'flatten' or 'avg', got {head!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:            # (B, roi_len) -> (B, 1, roi_len)
            x = x.unsqueeze(1)
        return self.head(self.features(x))
