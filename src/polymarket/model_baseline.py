"""Baseline p_hat: Ridge on logit(market implied) with clipped probabilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def _logit(p: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> np.ndarray:
    p = np.clip(p, lo, hi)
    return np.log(p / (1.0 - p))


def fit_ridge_logit_implied(X_train: np.ndarray, implied_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    y = _logit(implied_train.astype(float))
    mask = np.isfinite(y) & np.isfinite(X_train).all(axis=1)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train[mask], y[mask])
    return model


def predict_p_hat(model: Ridge, X: np.ndarray) -> np.ndarray:
    z = model.predict(X)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 0.02, 0.98)
