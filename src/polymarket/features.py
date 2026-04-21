"""Feature matrix for baseline p_hat (no future information in transform)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS_DEFAULT = (
    "log_volume",
    "log_liquidity",
    "spread",
    "competitive",
    "log_days_to_end",
    "neg_log_liquidity",  # inverse depth proxy
)


def _safe_log1p(x: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(x, errors="coerce").clip(lower=0.0))


def build_raw_features(df: pd.DataFrame, ref_time: pd.Timestamp) -> pd.DataFrame:
    """
    Row-wise features. ref_time should be the evaluation-time clock for the fold
    (e.g. max(createdAt) in the training set) so days_to_end uses no future listing times.
    """
    feats = pd.DataFrame(index=df.index)
    feats["log_volume"] = _safe_log1p(df.get("volumeNum", df.get("volume", 0)))
    feats["log_liquidity"] = _safe_log1p(df.get("liquidityNum", df.get("liquidity", 0)))
    feats["spread"] = pd.to_numeric(df.get("spread", 0.0), errors="coerce")
    feats["competitive"] = pd.to_numeric(df.get("competitive", 0.0), errors="coerce")

    end = df["endDate_parsed"] if "endDate_parsed" in df.columns else pd.Series(pd.NaT, index=df.index)
    if not isinstance(ref_time, pd.Timestamp):
        ref_time = pd.Timestamp(ref_time)
    days = (pd.to_datetime(end, utc=True) - ref_time).dt.total_seconds() / 86400.0
    feats["log_days_to_end"] = np.log1p(np.maximum(days.fillna(30.0), 1.0))

    liq_source = df.get("liquidityNum", df.get("liquidity", pd.Series(1.0, index=df.index)))
    liq = pd.to_numeric(liq_source, errors="coerce").fillna(1.0).clip(lower=1.0)
    feats["neg_log_liquidity"] = -np.log(liq)

    return feats


def fit_transform_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
    ref_time: pd.Timestamp,
) -> tuple[np.ndarray, StandardScaler | None, np.ndarray | None]:
    """Fit StandardScaler on train; transform train and optional test."""
    raw_tr = build_raw_features(df_train, ref_time)
    raw_tr = raw_tr.replace([np.inf, -np.inf], np.nan).fillna(raw_tr.median(numeric_only=True))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(raw_tr.values)

    if df_test is None:
        return X_train, scaler, None

    raw_te = build_raw_features(df_test, ref_time=ref_time)
    raw_te = raw_te.replace([np.inf, -np.inf], np.nan)
    for c in raw_tr.columns:
        if c in raw_te.columns:
            raw_te[c] = raw_te[c].fillna(raw_tr[c].median())
    X_test = scaler.transform(raw_te[raw_tr.columns].values)
    return X_train, scaler, X_test
