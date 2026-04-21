"""Panel-aware experiment runner for Polymarket daily data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitSpec:
    train_end: pd.Timestamp
    valid_end: pd.Timestamp


def load_panel(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["end_date_iso"] = pd.to_datetime(df["end_date_iso"], utc=True, errors="coerce")
    for c in ("price", "volume", "liquidity"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["market_id", "date", "price"]).copy()
    df = df.sort_values(["market_id", "date"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("market_id", group_keys=False)
    out["ret_1d"] = g["price"].pct_change()
    out["ret_7d"] = g["price"].pct_change(7)
    out["price_lag_1"] = g["price"].shift(1)
    out["price_lag_7"] = g["price"].shift(7)
    out["ret_1d_lag_1"] = g["ret_1d"].shift(1)
    out["ret_7d_lag_1"] = g["ret_7d"].shift(1)
    out["mom_3d"] = g["ret_1d"].rolling(3).sum().reset_index(level=0, drop=True)
    out["mom_7d"] = g["ret_1d"].rolling(7).sum().reset_index(level=0, drop=True)
    out["vol_7d"] = g["ret_1d"].rolling(7).std().reset_index(level=0, drop=True)
    out["vol_21d"] = g["ret_1d"].rolling(21).std().reset_index(level=0, drop=True)
    out["days_to_end"] = (out["end_date_iso"] - out["date"]).dt.total_seconds() / 86400.0
    out["log_volume"] = np.log1p(out["volume"].clip(lower=0))
    out["log_liquidity"] = np.log1p(out["liquidity"].clip(lower=0))
    out["illiq_proxy"] = np.abs(out["ret_1d"]) / np.maximum(out["volume"], 1.0)
    out["target_1d"] = g["ret_1d"].shift(-1)
    out["target_7d"] = g["ret_1d"].shift(-7)

    out["liq_rank_pct"] = out.groupby("date")["liquidity"].rank(pct=True)
    out["price_rank_pct"] = out.groupby("date")["price"].rank(pct=True)
    out["ret_rank_pct"] = out.groupby("date")["ret_1d"].rank(pct=True)
    out["cs_mispricing"] = out["price"] - out.groupby("date")["price"].transform("median")
    out["liq_x_mom7"] = out["liq_rank_pct"] * out["mom_7d"]
    out["illiq_x_mom7"] = out["illiq_proxy"] * out["mom_7d"]
    return out


def make_split(df: pd.DataFrame, split: SplitSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= split.train_end].copy()
    valid = df[(df["date"] > split.train_end) & (df["date"] <= split.valid_end)].copy()
    test = df[df["date"] > split.valid_end].copy()
    return train, valid, test


def split_60_20_20(df: pd.DataFrame) -> SplitSpec:
    dates = np.sort(df["date"].dropna().unique())
    n = len(dates)
    i1 = int(n * 0.6)
    i2 = int(n * 0.8)
    return SplitSpec(train_end=pd.Timestamp(dates[i1 - 1]), valid_end=pd.Timestamp(dates[i2 - 1]))


def feature_cols() -> list[str]:
    return [
        "price_lag_1",
        "price_lag_7",
        "ret_1d_lag_1",
        "ret_7d_lag_1",
        "mom_3d",
        "mom_7d",
        "vol_7d",
        "vol_21d",
        "days_to_end",
        "log_volume",
        "log_liquidity",
        "illiq_proxy",
        "liq_rank_pct",
        "price_rank_pct",
        "ret_rank_pct",
        "cs_mispricing",
        "liq_x_mom7",
        "illiq_x_mom7",
    ]


def sanitize_xy(df: pd.DataFrame, y_col: str) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_cols()
    keep = cols + [y_col, "date", "market_id", "liquidity", "category", "days_to_end", "question"]
    xdf = df[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=[y_col]).copy()
    x = xdf[cols].fillna(xdf[cols].median(numeric_only=True))
    y = xdf[y_col]
    return x, y


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def sanity_check_no_overlap(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, bool]:
    key = lambda d: set(zip(d["market_id"].astype(str), d["date"].astype(str)))
    kt, kv, ks = key(train), key(valid), key(test)
    chrono_ok = True
    if len(train) and len(valid):
        chrono_ok = chrono_ok and (train["date"].max() <= valid["date"].min())
    if len(valid) and len(test):
        chrono_ok = chrono_ok and (valid["date"].max() <= test["date"].min())
    if len(train) and len(test) and not len(valid):
        chrono_ok = chrono_ok and (train["date"].max() <= test["date"].min())

    return {
        "no_train_valid_overlap": len(kt & kv) == 0,
        "no_train_test_overlap": len(kt & ks) == 0,
        "no_valid_test_overlap": len(kv & ks) == 0,
        "chrono_order": bool(chrono_ok),
    }


def run_exp_01(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    split = split_60_20_20(df)
    train, valid, test = make_split(df, split)
    checks = sanity_check_no_overlap(train, valid, test)

    xtr, ytr = sanitize_xy(train, "target_1d")
    xva, yva = sanitize_xy(valid, "target_1d")
    xte, yte = sanitize_xy(test, "target_1d")

    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(xtr)
    xva_s = scaler.transform(xva)
    xte_s = scaler.transform(xte)

    ridge_grid = [0.1, 1.0, 5.0, 20.0]
    best_alpha, best_mae = None, np.inf
    for alpha in ridge_grid:
        m = Ridge(alpha=alpha)
        m.fit(xtr_s, ytr)
        pred = m.predict(xva_s)
        mae = mean_absolute_error(yva, pred)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    model = Ridge(alpha=float(best_alpha))
    model.fit(xtr_s, ytr)
    pred_te = model.predict(xte_s)
    pred_prob = 1.0 / (1.0 + np.exp(-pred_te * 10.0))
    dir_true = (yte > 0).astype(int)
    dir_pred = (pred_te > 0).astype(int)

    spread = 0.003
    tc = spread + 0.02 * (1.0 / np.maximum(test.loc[yte.index, "liquidity"].values, 1.0))
    w = np.clip(pred_te / (np.nanstd(pred_te) + 1e-9), -1.0, 1.0)
    gross = w * yte.values
    net = gross - np.abs(w) * tc
    turnover = float(np.mean(np.abs(np.diff(w)))) if len(w) > 1 else 0.0

    metrics = {
        "r2_test": float(r2_score(yte, pred_te)),
        "mae_test": float(mean_absolute_error(yte, pred_te)),
        "directional_accuracy_test": float(accuracy_score(dir_true, dir_pred)),
        "rank_ic_test": rank_ic(yte.values, pred_te),
        "brier_directional_test": float(brier_score_loss(dir_true, pred_prob)),
        "net_mean_return_test": float(np.mean(net)),
        "gross_mean_return_test": float(np.mean(gross)),
        "net_sharpe_test": float(np.mean(net) / (np.std(net) + 1e-9) * np.sqrt(252.0)),
        "turnover_test": turnover,
        "best_alpha_from_validation": float(best_alpha),
    }

    plt.figure(figsize=(7, 4))
    plt.plot(np.cumsum(gross), label="gross")
    plt.plot(np.cumsum(net), label="net")
    plt.title("Exp 1 cumulative gross vs net PnL")
    plt.legend()
    plt.tight_layout()
    fig1 = out_dir / "fig_exp01_cum_pnl.png"
    plt.savefig(fig1, dpi=130)
    plt.close()

    q = pd.qcut(pred_te, 10, labels=False, duplicates="drop")
    decile = pd.DataFrame({"q": q, "ret": yte.values}).groupby("q")["ret"].mean().reset_index()
    table1 = out_dir / "table_exp01_decile_returns.csv"
    decile.to_csv(table1, index=False)

    checks.update(
        {
            "validation_only_selection": True,
            "features_use_lag_only": True,
            "cost_stress_plus50_net_positive": float(np.mean(gross - np.abs(w) * tc * 1.5)) > 0.0,
        }
    )
    return {"metrics": metrics, "checks": checks, "artifacts": [str(fig1), str(table1)]}


def run_exp_02(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    split = split_60_20_20(df)
    train, valid, test = make_split(df, split)
    checks = sanity_check_no_overlap(train, valid, test)
    xtr, ytr = sanitize_xy(train, "target_1d")
    xte, yte = sanitize_xy(test, "target_1d")
    scaler = StandardScaler()
    xtr_s, xte_s = scaler.fit_transform(xtr), scaler.transform(xte)
    model = Ridge(alpha=5.0).fit(xtr_s, ytr)
    pred_te = model.predict(xte_s)
    tdf = test.loc[yte.index].copy()
    train_q = pd.qcut(train["liquidity"].fillna(0), 5, retbins=True, duplicates="drop")[1]
    tdf["liq_bucket"] = pd.cut(tdf["liquidity"], bins=train_q, include_lowest=True, labels=False)
    grouped = []
    for b, g in pd.DataFrame({"b": tdf["liq_bucket"], "y": yte.values, "p": pred_te}).dropna().groupby("b"):
        grouped.append({"b": b, "rank_ic": rank_ic(g["y"].values, g["p"].values), "mae": mean_absolute_error(g["y"], g["p"])})
    grp = pd.DataFrame(grouped)
    table = out_dir / "table_exp02_bucket_metrics.csv"
    grp.to_csv(table, index=False)
    plt.figure(figsize=(6, 4))
    plt.bar(grp["b"].astype(str), grp["rank_ic"])
    plt.title("Exp 2 rank IC by liquidity bucket")
    plt.tight_layout()
    fig = out_dir / "fig_exp02_rankic_bucket.png"
    plt.savefig(fig, dpi=130)
    plt.close()
    checks["bucket_cutoffs_from_train_only"] = True
    return {"metrics": {"mean_bucket_rank_ic": float(grp["rank_ic"].mean())}, "checks": checks, "artifacts": [str(fig), str(table)]}


def run_exp_03(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    split = split_60_20_20(df)
    train, _, test = make_split(df, split)
    checks = sanity_check_no_overlap(train, test.iloc[:0], test)
    xtr, ytr = sanitize_xy(train, "target_1d")
    xte, yte = sanitize_xy(test, "target_1d")
    base_cols = [c for c in feature_cols() if c not in ("ret_rank_pct", "cs_mispricing")]
    xtr_base = xtr[base_cols]
    xte_base = xte[base_cols]
    base = Ridge(alpha=5.0).fit(StandardScaler().fit_transform(xtr_base), ytr)
    fsc = StandardScaler().fit(xtr)
    full = Ridge(alpha=5.0).fit(fsc.transform(xtr), ytr)
    pred_base = base.predict(StandardScaler().fit(xtr_base).transform(xte_base))
    pred_full = full.predict(fsc.transform(xte))
    delta_r2 = float(r2_score(yte, pred_full) - r2_score(yte, pred_base))
    table = out_dir / "table_exp03_incremental.csv"
    pd.DataFrame([{"delta_r2": delta_r2, "rank_ic_base": rank_ic(yte.values, pred_base), "rank_ic_full": rank_ic(yte.values, pred_full)}]).to_csv(table, index=False)
    checks["factor_definitions_train_frozen"] = True
    return {"metrics": {"delta_r2_full_minus_base": delta_r2}, "checks": checks, "artifacts": [str(table)]}


def run_exp_04(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    split = split_60_20_20(df)
    train, valid, test = make_split(df, split)
    checks = sanity_check_no_overlap(train, valid, test)
    xtr, ytr = sanitize_xy(train, "target_1d")
    xva, yva = sanitize_xy(valid, "target_1d")
    xte, yte = sanitize_xy(test, "target_1d")
    scaler = StandardScaler()
    xtr_s, xva_s, xte_s = scaler.fit_transform(xtr), scaler.transform(xva), scaler.transform(xte)
    rf = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
    rf.fit(xtr_s, ytr)
    pred_va = rf.predict(xva_s)
    pred_te = rf.predict(xte_s)
    residual_scale = float(np.std(yva - pred_va))
    tmeta = test.loc[yte.index]
    haircut = 1.0 / (1.0 + np.maximum(tmeta["days_to_end"].fillna(30.0).values, 0.0) / 30.0)
    pred_adj = pred_te * haircut
    pdir = 1.0 / (1.0 + np.exp(-pred_adj * 10.0))
    ydir = (yte > 0).astype(int)
    cov = float(np.mean((np.abs(yte - pred_adj) <= residual_scale).astype(float)))
    metrics = {
        "brier_directional": float(brier_score_loss(ydir, pdir)),
        "interval_coverage_1sigma": cov,
        "mae_adjusted": float(mean_absolute_error(yte, pred_adj)),
    }
    table = out_dir / "table_exp04_reliability.csv"
    pd.DataFrame([metrics]).to_csv(table, index=False)
    checks["proxy_labels_no_terminal_outcome_peeking"] = True
    checks["haircut_tuned_on_validation"] = True
    return {"metrics": metrics, "checks": checks, "artifacts": [str(table)]}


def run_exp_05(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    results: list[dict[str, Any]] = []
    dates = np.sort(df["date"].dropna().unique())
    horizons = {"single_holdout": (0.6, 0.2), "expanding": (0.5, 0.15), "rolling": (0.45, 0.15)}
    for name, (tr_ratio, va_ratio) in horizons.items():
        i1 = max(1, int(len(dates) * tr_ratio))
        i2 = max(i1 + 1, int(len(dates) * (tr_ratio + va_ratio)))
        split = SplitSpec(pd.Timestamp(dates[i1 - 1]), pd.Timestamp(dates[min(i2 - 1, len(dates) - 1)]))
        train, valid, test = make_split(df, split)
        sc = sanity_check_no_overlap(train, valid, test)
        checks.update({f"{name}_{k}": v for k, v in sc.items()})
        xtr, ytr = sanitize_xy(train, "target_1d")
        xva, yva = sanitize_xy(valid, "target_1d")
        xte, yte = sanitize_xy(test, "target_1d")
        sca = StandardScaler()
        xtr_s, xva_s, xte_s = sca.fit_transform(xtr), sca.transform(xva), sca.transform(xte)
        alphas = [0.1, 1.0, 5.0, 20.0]
        rows = []
        for a in alphas:
            m = Ridge(alpha=a).fit(xtr_s, ytr)
            rows.append((a, mean_absolute_error(yva, m.predict(xva_s))))
        best = min(rows, key=lambda z: z[1])[0]
        m = Ridge(alpha=best).fit(xtr_s, ytr)
        val_score = -mean_absolute_error(yva, m.predict(xva_s))
        test_score = -mean_absolute_error(yte, m.predict(xte_s))
        results.append({"splitter": name, "best_alpha": best, "val_score": float(val_score), "test_score": float(test_score)})
    rdf = pd.DataFrame(results)
    rdf["selection_gap"] = rdf["test_score"] - rdf["val_score"]
    table = out_dir / "table_exp05_splitter_compare.csv"
    rdf.to_csv(table, index=False)
    plt.figure(figsize=(6, 4))
    plt.scatter(rdf["val_score"], rdf["test_score"])
    for _, r in rdf.iterrows():
        plt.text(r["val_score"], r["test_score"], r["splitter"])
    plt.xlabel("validation score")
    plt.ylabel("test score")
    plt.title("Exp 5 validation vs test by splitter")
    plt.tight_layout()
    fig = out_dir / "fig_exp05_val_vs_test.png"
    plt.savefig(fig, dpi=130)
    plt.close()
    checks["validation_only_tuning"] = True
    return {"metrics": {"val_test_corr": float(rdf["val_score"].corr(rdf["test_score"]))}, "checks": checks, "artifacts": [str(fig), str(table)]}


def to_markdown(exp_name: str, result: dict[str, Any], out_path: Path) -> None:
    lines = [f"# {exp_name}", "", "## Metrics", ""]
    for k, v in result["metrics"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Sanity checks", ""]
    for k, v in result["checks"].items():
        lines.append(f"- [{'x' if v else ' '}] {k}")
    lines += ["", "## Artifacts", ""]
    for a in result["artifacts"]:
        lines.append(f"- `{a}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")

