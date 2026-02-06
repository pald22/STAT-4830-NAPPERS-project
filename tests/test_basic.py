# tests/test_basic.py
"""
Basic unit tests for STAT 4830 Week 3 (NAPPERS) code.

Run with:
    pytest tests/test_basic.py
"""

import numpy as np
import pandas as pd

from src.model import estimate_moments, optimize_weights


def test_estimate_moments_shapes_and_symmetry():
    """estimate_moments should return correctly-shaped, symmetric outputs."""
    data = {
        "a": [0.01, 0.02, 0.015, 0.005],
        "b": [0.00, 0.01, -0.005, 0.002],
        "c": [0.03, -0.01, 0.02, 0.01],
    }
    df = pd.DataFrame(data)

    mu, Sigma = estimate_moments(df)

    # Shapes
    assert mu.shape == (3,)
    assert Sigma.shape == (3, 3)

    # Symmetry of covariance
    assert np.allclose(Sigma, Sigma.T, atol=1e-8)

    # Diagonal entries should be non-negative variances
    assert np.all(np.diag(Sigma) >= 0)


def test_optimize_weights_simplex():
    """optimize_weights should return weights in the probability simplex."""
    mu = np.array([0.02, 0.01, 0.005])
    # positive definite diagonal covariance
    Sigma = np.diag([0.10, 0.05, 0.02]) ** 2
    w_prev = np.ones(3) / 3

    w = optimize_weights(
        mu,
        Sigma,
        w_prev=w_prev,
        gamma=1.0,
        kappa=0.0,
        steps=100,
        lr=0.1,
        device="cpu",
    )

    # Shape
    assert w.shape == (3,)

    # Nonnegative & sum to 1 (simplex)
    assert np.all(w >= -1e-8)
    assert abs(w.sum() - 1.0) < 1e-6


def test_turnover_penalty_influence():
    """
    With a large turnover penalty, solution should stay close to w_prev
    compared to the no-penalty case (same mu, Sigma, gamma).
    """
    mu = np.array([0.03, 0.01, 0.0])
    Sigma = np.diag([0.15, 0.10, 0.05]) ** 2
    w_prev = np.array([0.2, 0.4, 0.4])

    # No turnover penalty
    w_no_pen = optimize_weights(
        mu,
        Sigma,
        w_prev=w_prev,
        gamma=1.0,
        kappa=0.0,
        steps=150,
        lr=0.05,
        device="cpu",
    )

    # Large turnover penalty
    w_pen = optimize_weights(
        mu,
        Sigma,
        w_prev=w_prev,
        gamma=1.0,
        kappa=10.0,
        steps=150,
        lr=0.05,
        device="cpu",
    )

    # Distance to previous weights should be smaller when kappa is large
    dist_no_pen = np.sum(np.abs(w_no_pen - w_prev))
    dist_pen = np.sum(np.abs(w_pen - w_prev))

    assert dist_pen <= dist_no_pen + 1e-6  # allow tiny numerical slack
