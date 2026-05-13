"""Projected (sub)gradient ascent for constrained portfolio objective."""

from __future__ import annotations

from typing import Any

import numpy as np


def _subgrad_l1(w: np.ndarray, w_prev: np.ndarray, kappa: float) -> np.ndarray:
    d = w - w_prev
    g = np.zeros_like(w)
    mask = np.abs(d) > 1e-12
    g[mask] = kappa * np.sign(d[mask])
    return g


def smooth_gradient(
    w: np.ndarray,
    a: np.ndarray,
    p_hat: np.ndarray,
    gamma: float,
    Sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Gradient of smooth part: a^T w - gamma * diag Bernoulli risk, or a^T w - gamma * w^T Sigma w."""
    if Sigma is None:
        v = p_hat * (1.0 - p_hat)
        return a - 2.0 * gamma * w * v
    return a - 2.0 * gamma * (Sigma @ w)


def build_contract_covariance(
    history_df: Any,
    active_contract_ids: list[Any] | np.ndarray,
    decision_date: Any,
    price_col: str = "implied_0",
    contract_col: str = "id",
    date_col: str = "_decision_time",
    lookback_days: int = 60,
    eps: float = 1e-6,
    p_hat: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build chronology-safe covariance matrix for active Polymarket contracts.

    Uses only observations strictly before decision_date.
    Estimates covariance of contract-level implied-probability (quote) changes.
    Falls back to diagonal Bernoulli variance if insufficient history exists.
    """
    import pandas as pd

    active_contract_ids = list(active_contract_ids)
    n = len(active_contract_ids)

    if p_hat is not None:
        p = np.asarray(p_hat, dtype=float)
        diag = np.clip(p * (1.0 - p), 1e-8, None)
    else:
        diag = np.full(n, 1e-4, dtype=float)

    fallback = np.diag(diag)

    def _ridge(M: np.ndarray) -> np.ndarray:
        return M + eps * np.eye(M.shape[0], dtype=float)

    if n == 0:
        return np.zeros((0, 0), dtype=float)

    df = history_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    decision_date = pd.to_datetime(decision_date, utc=True, errors="coerce")

    start_date = decision_date - pd.Timedelta(days=lookback_days)

    hist = df[
        (df[date_col] < decision_date)
        & (df[date_col] >= start_date)
        & (df[contract_col].isin(active_contract_ids))
    ][[date_col, contract_col, price_col]].copy()
    hist[price_col] = pd.to_numeric(hist[price_col], errors="coerce")
    hist = hist.dropna()

    if hist.empty:
        return _ridge(fallback)

    panel = (
        hist.pivot_table(
            index=date_col,
            columns=contract_col,
            values=price_col,
            aggfunc="last",
        )
        .sort_index()
        .reindex(columns=active_contract_ids)
    )

    changes = panel.diff()

    if changes.shape[0] < 3:
        return _ridge(fallback)

    Sigma_df = changes.cov(min_periods=3)
    Sigma = Sigma_df.reindex(
        index=active_contract_ids,
        columns=active_contract_ids,
    ).to_numpy(dtype=float)

    Sigma = np.where(np.isfinite(Sigma), Sigma, 0.0)

    d = np.diag(Sigma).copy()
    fallback_d = np.diag(fallback)
    bad_diag = (~np.isfinite(d)) | (d <= 0)
    d[bad_diag] = fallback_d[bad_diag]
    np.fill_diagonal(Sigma, d)

    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma + eps * np.eye(n)

    return Sigma


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
    Sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximize F(w) = a^T w - gamma * risk(w) - kappa * ||w-w_prev||_1
    with risk(w) = sum w_i^2 p_i(1-p_i) if Sigma is None, else w^T Sigma w.

    Via projected subgradient ascent + cyclic projection onto linear constraints.

    Returns (w_opt, history of smooth objective).
    """
    n = len(a)
    w = np.zeros(n)
    v = p_hat * (1.0 - p_hat)
    if Sigma is None:
        lip = 2.0 * gamma * (np.max(v) + 1e-8) + 1e-8
    else:
        sig_norm = float(np.linalg.norm(Sigma, ord=2)) if n else 0.0
        lip = 2.0 * gamma * (sig_norm + 1e-8) + 1e-8
    eta = eta0 if eta0 is not None else 0.5 / lip

    event_groups = _unique_event_indices(event_codes)

    hist = []
    smooth_obj = 0.0
    for it in range(n_iter):
        g_s = smooth_gradient(w, a, p_hat, gamma, Sigma=Sigma)
        g_l1 = _subgrad_l1(w, w_prev, kappa)
        g = g_s - g_l1
        w = w + eta * g
        w = project_feasible_fast_v2(w, w_contract_cap, event_groups, w_event_max, liq_load, liq_budget)
        if Sigma is None:
            smooth_obj = float(np.dot(w, a) - gamma * np.sum((w**2) * v))
        else:
            smooth_obj = float(np.dot(w, a) - gamma * float(w @ Sigma @ w))
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
    Sigma: np.ndarray | None = None,
) -> float:
    if Sigma is None:
        v = p_hat * (1.0 - p_hat)
        smooth = float(np.dot(w, a) - gamma * np.sum((w**2) * v))
    else:
        smooth = float(np.dot(w, a) - gamma * float(w @ Sigma @ w))
    pen = kappa * float(np.sum(np.abs(w - w_prev)))
    return smooth - pen
