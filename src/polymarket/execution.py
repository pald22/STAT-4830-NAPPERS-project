"""Executable Yes-price approximation from snapshot fields."""

from __future__ import annotations

import numpy as np
import pandas as pd


def executable_yes_price(df: pd.DataFrame) -> pd.Series:
    """
    Conservative long-Yes execution price:
    - use bestAsk when present (buying Yes lifts the offer);
    - else mid from bid/ask if both present;
    - else implied_0 + half spread when spread exists;
    - else implied_0.
    """
    implied = pd.to_numeric(df.get("implied_0"), errors="coerce")
    bid = pd.to_numeric(df.get("bestBid"), errors="coerce")
    ask = pd.to_numeric(df.get("bestAsk"), errors="coerce")
    sp = pd.to_numeric(df.get("spread"), errors="coerce")

    out = implied.copy()
    if implied.isna().any() and ask.notna().any():
        out = out.where(~implied.isna(), ask)
    mid = (bid + ask) / 2.0
    use_mid = bid.notna() & ask.notna()
    out = out.where(~use_mid, mid)
    use_ask = ask.notna()
    out = out.where(~use_ask, ask)

    half = sp / 2.0
    fallback = (implied + half).where(implied.notna() & sp.notna(), implied)
    mask = out.isna() & implied.notna()
    out = out.where(~mask, fallback[mask])

    return out.clip(0.001, 0.999)


def liquidity_loadings(df: pd.DataFrame, eps: float = 1.0) -> np.ndarray:
    """Coefficients l_i >= 0 for constraint sum_i w_i * l_i <= budget (inverse depth)."""
    liq = pd.to_numeric(df.get("liquidityNum", 1.0), errors="coerce").fillna(1.0).values.astype(float)
    liq = np.maximum(liq, eps)
    return 1.0 / np.sqrt(liq)
