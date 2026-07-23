"""Classical baseline: XGBoost on hand-crafted features (Rung 1).

This is the "robust simple model" and the accuracy bar the neural nets must beat.
XGBoost can use the GPU (``device="cuda"``) but that barely exercises CUDA — the
real PyTorch/CUDA learning starts in ``gpulab.models.cnn1d`` (Rung 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from ..features import feature_matrix


@dataclass
class XGBResult:
    model: Any
    classes: list[str]
    feature_names: list[str]
    accuracy: float
    report: str
    importances: dict[str, float] = field(default_factory=dict)


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    device: str = "cuda",
    **xgb_kwargs: Any,
) -> XGBResult:
    """Featurize, fit XGBoost, and evaluate on the held-out chips."""
    from xgboost import XGBClassifier

    F_train, names = feature_matrix(X_train)
    F_test, _ = feature_matrix(X_test)

    params: dict[str, Any] = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=device,
        eval_metric="mlogloss",
    )
    params.update(xgb_kwargs)

    model = XGBClassifier(**params)
    model.fit(F_train, y_train)

    pred = model.predict(F_test)
    acc = float(accuracy_score(y_test, pred))
    report = classification_report(y_test, pred, target_names=classes, zero_division=0)
    importances = dict(zip(names, model.feature_importances_.tolist()))

    return XGBResult(
        model=model,
        classes=classes,
        feature_names=names,
        accuracy=acc,
        report=report,
        importances=dict(sorted(importances.items(), key=lambda kv: -kv[1])),
    )
