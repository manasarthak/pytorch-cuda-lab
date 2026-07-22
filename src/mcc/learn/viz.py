"""Visual inspectors -- turn tensors and training state into pictures.

These return a matplotlib Figure so they work in notebooks (they display) and in
scripts (call ``fig.savefig(...)``). Accept torch tensors or numpy arrays.

Import: ``from mcc.learn import viz``.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):          # torch tensor
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def plot_curve(curve: Any, title: str = "curve") -> "plt.Figure":
    """Plot a single 1-D signal -- e.g. a preprocessed -dF/dT melt curve."""
    y = _to_numpy(curve).ravel()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(y, lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("value")
    fig.tight_layout()
    return fig


def plot_curves(curves: Any, labels: Any = None, max_curves: int = 40, title: str = "curves") -> "plt.Figure":
    """Overlay many curves (optionally colored by integer label)."""
    X = _to_numpy(curves)
    fig, ax = plt.subplots(figsize=(6, 4))
    n = min(len(X), max_curves)
    lab = _to_numpy(labels).astype(int) if labels is not None else None
    for i in range(n):
        color = None if lab is None else plt.cm.tab10(lab[i] % 10)
        ax.plot(X[i], lw=0.8, alpha=0.6, color=color)
    ax.set_title(f"{title} (showing {n})")
    ax.set_xlabel("frame")
    fig.tight_layout()
    return fig


def heatmap(mat: Any, title: str = "tensor") -> "plt.Figure":
    """imshow a 2-D tensor/matrix -- good for seeing a batch, weights, or a rec-plot."""
    M = _to_numpy(mat)
    if M.ndim == 1:
        M = M[None, :]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{title}  shape={M.shape}")
    fig.tight_layout()
    return fig


def plot_history(history: list[dict], title: str = "training") -> "plt.Figure":
    """Loss + accuracy over epochs, from the list-of-dicts your training loop returns."""
    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], "C0-", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="C0")
    if "test_acc" in history[0]:
        ax2 = ax1.twinx()
        ax2.plot(epochs, [h["test_acc"] for h in history], "C1-", label="test acc")
        ax2.set_ylabel("accuracy", color="C1")
    ax1.set_title(title)
    fig.tight_layout()
    return fig


def plot_grad_flow(module: Any, title: str = "gradient flow") -> "plt.Figure":
    """Bar chart of per-layer gradient norms -- run AFTER ``loss.backward()``.

    Flat bars near zero in early layers = vanishing gradients; spikes = exploding.
    """
    names, norms = [], []
    for pname, p in module.named_parameters():
        if p.grad is not None and p.requires_grad:
            names.append(pname.replace(".weight", ".w").replace(".bias", ".b"))
            norms.append(p.grad.detach().norm().item())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(norms)), norms)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("grad L2 norm")
    ax.set_title(title)
    fig.tight_layout()
    return fig
