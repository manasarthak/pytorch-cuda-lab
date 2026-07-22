"""A small from-scratch 1D-CNN over the derivative curve (Rung 2).

Deliberately simple: the goal of this rung is to learn the PyTorch/CUDA mechanics
(tensors, devices, the training loop, AMP), not to chase accuracy. Input is a
curve of shape ``(batch, 1, roi_len)`` — channels-first, as Conv1d expects.

Requires torch: ``uv add torch --index-url https://download.pytorch.org/whl/cu126``.
"""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, n_classes: int, roi_len: int = 61, width: int = 32):
        super().__init__()
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
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # global average pool over time -> (B, C, 1)
            nn.Flatten(),
            nn.Linear(width * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:            # (B, roi_len) -> (B, 1, roi_len)
            x = x.unsqueeze(1)
        return self.head(self.features(x))
