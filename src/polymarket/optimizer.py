"""Projected (sub)gradient ascent for constrained portfolio objective."""

from __future__ import annotations

import numpy as np


def _subgrad_l1(w: np.ndarray, w_prev: np.ndarray, kappa: float) -> np.ndarray:
    d = w - w_prev
    g = np.zeros_like(w)
    mask = np.abs(d) > 1e-12
    g[mask] = kappa * np.sign(d[mask])
    return g


def smooth_gradient(w: np.ndarray, a: np.ndarray, p_hat: np.ndarray, gamma: float) -> np.ndarray:
    """Gradient of sum w_i a_i - gamma * sum w_i^2 * p_i(1-p_i)."""
    v = p_hat * (1.0 - p_hat)
    return a - 2.0 * gamma * w * v


def _unique_event_indices(event_codes: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for e in np.unique(event_codes):
        idx = np.flatnonzero(event_codes == e)
        if len(idx) > 1:
            out.append(idx)
    return out


def project_feasible_fast_v2(
    w: np.ndarray,
    w_cap: np.ndarray,
    event_groups: list[np.ndarray],
    w_event_max: float,
    liq_load: np.ndarray,
    liq_budget: float,
    n_passes: int = 50,
) -> np.ndarray:
    w = np.clip(w.astype(float), 0, w_cap)
    for _ in range(n_passes):
        if w.sum() > 1.0 + 1e-12:
            w *= 1.0 / w.sum()
        w = np.clip(w, 0, w_cap)
        for idx in event_groups:
            s = float(w[idx].sum())
            if s > w_event_max + 1e-12:
                w[idx] *= w_event_max / s
        w = np.clip(w, 0, w_cap)
        liq_dot = float(np.dot(liq_load, w))
        if liq_budget < 1e9 and liq_dot > liq_budget + 1e-12:
            w *= liq_budget / (liq_dot + 1e-12)
        w = np.clip(w, 0, w_cap)
    return w


def maximize_portfolio_projected_gradient(
    a: np.ndarray,
    p_hat: np.ndarray,
    w_prev: np.ndarray,
    gamma: float,
    kappa: float,
    w_contract_cap: np.ndarray,
    event_codes: np.ndarray,
    liq_load: np.ndarray,
    w_event_max: float,
    liq_budget: float,
    n_iter: int = 800,
    eta0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximize F(w) = sum w*a - gamma*sum w^2 p(1-p) - kappa*||w-w_prev||_1
    via projected subgradient ascent + cyclic projection onto linear constraints.

    Returns (w_opt, history of smooth objective).
    """
    n = len(a)
    w = np.zeros(n)
    v = p_hat * (1.0 - p_hat)
    lip = 2.0 * gamma * (np.max(v) + 1e-8) + 1e-8
    eta = eta0 if eta0 is not None else 0.5 / lip

    event_groups = _unique_event_indices(event_codes)

    hist = []
    smooth_obj = 0.0
    for it in range(n_iter):
        g_s = smooth_gradient(w, a, p_hat, gamma)
        g_l1 = _subgrad_l1(w, w_prev, kappa)
        g = g_s - g_l1
        w = w + eta * g
        w = project_feasible_fast_v2(w, w_contract_cap, event_groups, w_event_max, liq_load, liq_budget)
        smooth_obj = float(np.dot(w, a) - gamma * np.sum((w**2) * v))
        hist.append(smooth_obj)
        if it > 15 and abs(hist[-1] - hist[-2]) < 1e-9 * (abs(hist[-1]) + 1.0):
            break
        eta *= 0.9997

    return w, np.array(hist)


def total_objective(
    w: np.ndarray,
    a: np.ndarray,
    p_hat: np.ndarray,
    w_prev: np.ndarray,
    gamma: float,
    kappa: float,
) -> float:
    v = p_hat * (1.0 - p_hat)
    smooth = float(np.dot(w, a) - gamma * np.sum((w**2) * v))
    pen = kappa * float(np.sum(np.abs(w - w_prev)))
    return smooth - pen
