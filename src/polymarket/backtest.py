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
    train_ratio: float = 0.6
    valid_ratio: float = 0.2


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


def assign_splits_by_time(df: pd.DataFrame, train_ratio: float, valid_ratio: float) -> pd.Series:
    if "createdAt" not in df.columns or df["createdAt"].isna().all():
        return pd.Series("train", index=df.index)
    created = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
    unique_dates = np.sort(created.dropna().unique())
    n_dates = len(unique_dates)
    if n_dates == 0:
        return pd.Series("train", index=df.index)
    i_train = max(1, int(np.floor(n_dates * train_ratio)))
    i_valid = max(i_train + 1, int(np.floor(n_dates * (train_ratio + valid_ratio))))
    i_valid = min(i_valid, n_dates)
    train_cut = pd.Timestamp(unique_dates[i_train - 1])
    valid_cut = pd.Timestamp(unique_dates[i_valid - 1] if i_valid > 0 else train_cut)

    labels = np.where(
        created <= train_cut,
        "train",
        np.where(created <= valid_cut, "validation", "test"),
    )
    return pd.Series(labels, index=df.index)


def _sharpe_from_series(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) < 2:
        return 0.0
    std = float(vals.std())
    if std <= 1e-12:
        return 0.0
    return float(vals.mean() / std * np.sqrt(252.0))


def run_walk_forward(
    df: pd.DataFrame,
    cfg: MispricingConfig | None = None,
    out_dir: Path | str | None = None,
) -> dict[str, Any]:
    cfg = cfg or MispricingConfig()

    df = df.sort_values("createdAt").reset_index(drop=True) if "createdAt" in df.columns else df.reset_index(drop=True)
    if "createdAt" in df.columns:
        df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
        df["_decision_time"] = df["createdAt"].dt.floor("D")
    else:
        df["_decision_time"] = pd.NaT
    df["_fold"] = assign_time_folds(df, cfg.n_folds)
    tmp = df.rename(columns={"createdAt": "_createdAt_original", "_decision_time": "createdAt"})
    tmp["_split"] = assign_splits_by_time(tmp, cfg.train_ratio, cfg.valid_ratio)
    df["_split"] = tmp["_split"].values

    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    implied = pd.to_numeric(df.get("implied_0"), errors="coerce")
    df["implied_0"] = implied
    df = df.sort_values(["id", "_decision_time", "createdAt"]).reset_index(drop=True)
    df["_next_implied"] = df.groupby("id")["implied_0"].shift(-1)

    metrics_rows: list[dict] = []
    all_weights: list[pd.DataFrame] = []
    trades_rows: list[dict] = []
    pnl_rows: list[dict] = []

    w_prev_by_id: dict[Any, float] = {}

    all_dates = np.sort(df["_decision_time"].dropna().unique())
    for t, date_key in enumerate(all_dates):
        test = df[df["_decision_time"] == date_key].copy()
        if len(test) == 0:
            continue
        train = df[df["_decision_time"] < date_key].copy()
        train = train[pd.to_numeric(train["implied_0"], errors="coerce").notna()]
        train_fit = train.copy()
        if len(train_fit) < cfg.min_train_rows:
            continue

        test = test[pd.to_numeric(test["implied_0"], errors="coerce").notna()].copy()
        test = test[pd.to_numeric(test["_next_implied"], errors="coerce").notna()].copy()
        if len(test) == 0:
            continue

        ref_time = pd.Timestamp(date_key)
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

        w_prev = np.array([w_prev_by_id.get(i, 0.0) for i in test["id"].values], dtype=float)

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

        for i, mid in enumerate(test["id"].values):
            w_prev_by_id[mid] = float(w_opt[i])

        obj = total_objective(w_opt, a, p_hat, w_prev, cfg.gamma, cfg.kappa)
        turnover = float(np.sum(np.abs(w_opt - w_prev)))
        implied_t = test["implied_0"].values
        mse = float(np.nanmean((p_hat - implied_t) ** 2))
        next_implied = pd.to_numeric(test["_next_implied"], errors="coerce").values
        pnl_realized = float(np.nansum(w_opt * (next_implied - c_exec)))
        split_name = str(test["_split"].iloc[0]) if "_split" in test.columns and len(test) else "unknown"

        metrics_rows.append(
            {
                "fold": t,
                "period_idx": t,
                "date": str(pd.Timestamp(date_key)),
                "split": split_name,
                "n_train": int(len(train_fit)),
                "n_test": int(n),
                "objective": obj,
                "mse_p_vs_implied": mse,
                "mean_abs_edge": float(np.nanmean(np.abs(a))),
                "sum_w": float(np.sum(w_opt)),
                "turnover_l1": turnover,
                "pnl_realized_next_step": pnl_realized,
            }
        )

        wdf = pd.DataFrame(
            {
                "fold": t,
                "period_idx": t,
                "date": str(pd.Timestamp(date_key)),
                "split": split_name,
                "id": test["id"].values,
                "weight": w_opt,
                "p_hat": p_hat,
                "c_exec": c_exec,
                "implied": implied_t,
                "next_implied": next_implied,
                "edge": a,
            }
        )
        all_weights.append(wdf)

        for i, mid in enumerate(test["id"].values):
            trades_rows.append(
                {
                    "fold": t,
                    "period_idx": t,
                    "date": str(pd.Timestamp(date_key)),
                    "split": split_name,
                    "id": mid,
                    "delta_w": float(w_opt[i] - w_prev[i]),
                    "w": float(w_opt[i]),
                }
            )

        pnl_rows.append(
            {
                "fold": t,
                "period_idx": t,
                "date": str(pd.Timestamp(date_key)),
                "split": split_name,
                "pnl_realized_next_step": pnl_realized,
                "objective": obj,
                "turnover_l1": turnover,
            }
        )

    pnl_df = pd.DataFrame(pnl_rows)
    pnl_by_split: dict[str, Any] = {}
    if len(pnl_df):
        for split_name, g in pnl_df.groupby("split"):
            pnl_by_split[str(split_name)] = {
                "n_periods": int(len(g)),
                "total_pnl": float(g["pnl_realized_next_step"].sum()),
                "mean_period_pnl": float(g["pnl_realized_next_step"].mean()),
                "sharpe_period_pnl": _sharpe_from_series(g["pnl_realized_next_step"]),
                "mean_objective": float(g["objective"].mean()),
                "mean_turnover": float(g["turnover_l1"].mean()),
            }

    summary = {
        "config": cfg.__dict__.copy(),
        "folds": metrics_rows,
        "pnl_by_split": pnl_by_split,
        "mean_objective": float(np.mean([m["objective"] for m in metrics_rows])) if metrics_rows else 0.0,
        "mean_mse": float(np.mean([m["mse_p_vs_implied"] for m in metrics_rows])) if metrics_rows else 0.0,
        "mean_turnover": float(np.mean([m["turnover_l1"] for m in metrics_rows])) if metrics_rows else 0.0,
    }

    result = {
        "metrics": metrics_rows,
        "summary": summary,
        "weights": pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame(),
        "trades": pd.DataFrame(trades_rows),
        "pnl": pnl_df,
    }

    if out_dir is not None:
        poly_io.save_run(Path(out_dir), result)

    return result
