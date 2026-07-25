"""A whole-dataset-on-GPU training loop with AMP (Rung 2).

Because the data is tiny, we skip DataLoader entirely: move ``X``/``y`` to the GPU
once, then slice minibatches on-device. This isolates the compute so you can study
GPU utilization without an input pipeline in the way. Mixed precision uses bf16,
the clean default on Ada (RTX 40-series) — no gradient scaler needed.

Requires torch: ``uv sync --extra torch``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .cuda_utils import describe_device, device, memory_summary


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 0


def _accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            logits = model(X[i:i + batch_size])
            correct += (logits.argmax(1) == y[i:i + batch_size]).sum().item()
    return correct / X.shape[0]


def train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig = TrainConfig(),
) -> dict:
    torch.manual_seed(cfg.seed)
    dev = device()
    print(f"[device] {describe_device()}")

    # Move the entire dataset onto the GPU once.
    Xtr = torch.as_tensor(X_train, dtype=torch.float32, device=dev)
    ytr = torch.as_tensor(y_train, dtype=torch.long, device=dev)
    Xte = torch.as_tensor(X_test, dtype=torch.float32, device=dev)
    yte = torch.as_tensor(y_test, dtype=torch.long, device=dev)

    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32

    n = Xtr.shape[0]
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        running = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=cfg.amp):
                loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            running += loss.item() * idx.shape[0]

        train_loss = running / n
        test_acc = _accuracy(model, Xte, yte, cfg.batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_acc": test_acc})
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:3d}  loss {train_loss:.4f}  test_acc {test_acc:.4f}  {memory_summary()}")

    return {"history": history, "final_test_acc": history[-1]["test_acc"]}
