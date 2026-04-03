"""Walk-forward backtest: train only on prior listing-time folds (no future rows)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polymarket.execution import executable_yes_price, liquidity_loadings
from polymarket.features import fit_transform_features
from polymarket.model_baseline import fit_ridge_logit_implied, predict_p_hat
from polymarket.optimizer import maximize_portfolio_projected_gradient, total_objective
from polymarket import io as poly_io


@dataclass
class MispricingConfig:
    n_folds: int = 5
    min_train_rows: int = 30
    ridge_alpha: float = 5.0
    gamma: float = 2.0
    kappa: float = 0.05
    w_contract_max: float = 0.08
    w_event_max: float = 0.25
    liq_budget: float = 2.5
    rng_seed: int = 42
    n_opt_iter: int = 800


def assign_time_folds(df: pd.DataFrame, n_folds: int) -> pd.Series:
    if "createdAt" not in df.columns or df["createdAt"].isna().all():
        return pd.Series(0, index=df.index)
    r = df["createdAt"].rank(method="first")
    n_folds = max(2, min(n_folds, len(r)))
    try:
        folds = pd.qcut(r, q=n_folds, labels=False, duplicates="drop")
        return folds.astype(int)
    except ValueError:
        return pd.Series(0, index=df.index)


def event_code_array(df: pd.DataFrame) -> np.ndarray:
    if "event_id" not in df.columns:
        return np.arange(len(df), dtype=int)
    s = df["event_id"].fillna("__na__").astype(str)
    return pd.factorize(s)[0].astype(int)


def run_walk_forward(
    df: pd.DataFrame,
    cfg: MispricingConfig | None = None,
    out_dir: Path | str | None = None,
) -> dict[str, Any]:
    cfg = cfg or MispricingConfig()
    rng = np.random.default_rng(cfg.rng_seed)

    df = df.sort_values("createdAt").reset_index(drop=True) if "createdAt" in df.columns else df.reset_index(drop=True)
    df["_fold"] = assign_time_folds(df, cfg.n_folds)

    metrics_rows: list[dict] = []
    all_weights: list[pd.DataFrame] = []
    trades_rows: list[dict] = []
    pnl_rows: list[dict] = []

    w_prev_by_id: dict[Any, float] = {}

    max_fold = int(df["_fold"].max()) if len(df) else 0
    for f in range(max_fold + 1):
        train = df[df["_fold"] < f].copy()
        test = df[df["_fold"] == f].copy()
        if len(test) == 0:
            continue
        if len(train) < cfg.min_train_rows:
            continue

        ref_time = train["createdAt"].max()
        implied_tr = pd.to_numeric(train["implied_0"], errors="coerce")
        mask_tr = implied_tr.notna() & np.isfinite(implied_tr)
        train_fit = train.loc[mask_tr]
        if len(train_fit) < cfg.min_train_rows:
            continue

        test = test[pd.to_numeric(test["implied_0"], errors="coerce").notna()].copy()
        if len(test) == 0:
            continue

        X_tr, _scaler, X_te = fit_transform_features(train_fit, test, ref_time=ref_time)
        implied_train = train_fit["implied_0"].values.astype(float)

        model = fit_ridge_logit_implied(X_tr, implied_train, alpha=cfg.ridge_alpha)
        p_hat_te = predict_p_hat(model, X_te)

        test = test.copy()
        test["p_hat"] = p_hat_te
        test["implied_0"] = pd.to_numeric(test["implied_0"], errors="coerce")
        c_exec = executable_yes_price(test).values
        p_hat = np.clip(test["p_hat"].values, 1e-6, 1 - 1e-6)
        a = p_hat - c_exec

        n = len(test)
        liq_load = liquidity_loadings(test)
        ev_codes = event_code_array(test)
        w_cap = np.full(n, cfg.w_contract_max)

        if "id" in test.columns:
            w_prev = np.array([w_prev_by_id.get(i, 0.0) for i in test["id"].values], dtype=float)
        else:
            w_prev = np.zeros(n)

        w_opt, _hist = maximize_portfolio_projected_gradient(
            a,
            p_hat,
            w_prev,
            gamma=cfg.gamma,
            kappa=cfg.kappa,
            w_contract_cap=w_cap,
            event_codes=ev_codes,
            liq_load=liq_load,
            w_event_max=cfg.w_event_max,
            liq_budget=cfg.liq_budget,
            n_iter=cfg.n_opt_iter,
        )

        if "id" in test.columns:
            for i, mid in enumerate(test["id"].values):
                w_prev_by_id[mid] = float(w_opt[i])

        obj = total_objective(w_opt, a, p_hat, w_prev, cfg.gamma, cfg.kappa)
        turnover = float(np.sum(np.abs(w_opt - w_prev)))
        implied_t = test["implied_0"].values
        mse = float(np.nanmean((p_hat - implied_t) ** 2))

        ph = np.clip(implied_t, 0.01, 0.99)
        y_sim = rng.binomial(1, ph)
        pnl_sim = float(np.nansum(w_opt * (y_sim - c_exec)))

        metrics_rows.append(
            {
                "fold": f,
                "n_train": int(len(train_fit)),
                "n_test": int(n),
                "objective": obj,
                "mse_p_vs_implied": mse,
                "mean_abs_edge": float(np.nanmean(np.abs(a))),
                "sum_w": float(np.sum(w_opt)),
                "turnover_l1": turnover,
                "pnl_sim_risk_neutral": pnl_sim,
            }
        )

        wdf = pd.DataFrame(
            {
                "fold": f,
                "id": test["id"].values if "id" in test.columns else np.arange(n),
                "weight": w_opt,
                "p_hat": p_hat,
                "c_exec": c_exec,
                "implied": implied_t,
                "edge": a,
            }
        )
        all_weights.append(wdf)

        if "id" in test.columns:
            for i, mid in enumerate(test["id"].values):
                trades_rows.append(
                    {
                        "fold": f,
                        "id": mid,
                        "delta_w": float(w_opt[i] - w_prev[i]),
                        "w": float(w_opt[i]),
                    }
                )

        pnl_rows.append({"fold": f, "pnl_sim": pnl_sim, "objective": obj})

    summary = {
        "config": cfg.__dict__.copy(),
        "folds": metrics_rows,
        "mean_objective": float(np.mean([m["objective"] for m in metrics_rows])) if metrics_rows else 0.0,
        "mean_mse": float(np.mean([m["mse_p_vs_implied"] for m in metrics_rows])) if metrics_rows else 0.0,
        "mean_turnover": float(np.mean([m["turnover_l1"] for m in metrics_rows])) if metrics_rows else 0.0,
    }

    result = {
        "metrics": metrics_rows,
        "summary": summary,
        "weights": pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame(),
        "trades": pd.DataFrame(trades_rows),
        "pnl": pd.DataFrame(pnl_rows),
    }

    if out_dir is not None:
        poly_io.save_run(Path(out_dir), result)

    return result
