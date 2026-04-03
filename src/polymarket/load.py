"""Load and enrich Polymarket CSV (same semantics as polymarket_exploration.ipynb)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_markets_csv(root: Path | None = None) -> Path:
    if root is None:
        root = Path.cwd()
    for candidate in (root / "polymarket_markets_rich.csv", root / "data" / "polymarket_markets_rich.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("polymarket_markets_rich.csv not found at repo root or data/.")


def first_outcome_price(raw: Any) -> float:
    if pd.isna(raw) or raw == "":
        return np.nan
    try:
        arr = json.loads(raw)
        return float(arr[0]) if arr else np.nan
    except (json.JSONDecodeError, TypeError, ValueError, IndexError):
        return np.nan


def first_event_id(raw: Any) -> str | float:
    if pd.isna(raw) or raw == "":
        return np.nan
    try:
        evs = json.loads(raw)
        if not evs:
            return np.nan
        return str(evs[0].get("id", np.nan))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return np.nan


def first_event_slug(raw: Any) -> str | float:
    if pd.isna(raw) or raw == "":
        return np.nan
    try:
        evs = json.loads(raw)
        if not evs:
            return np.nan
        return str(evs[0].get("slug", np.nan))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return np.nan


def load_polymarket_snapshot(path: Path | None = None, **read_csv_kw) -> pd.DataFrame:
    path = path or find_markets_csv()
    kw = {"low_memory": False}
    kw.update(read_csv_kw)
    return pd.read_csv(path, **kw)


def enrich_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Add implied_0, event_id, event_slug — mirrors the exploration notebook."""
    out = df.copy()
    if "outcomePrices" in out.columns:
        out["implied_0"] = out["outcomePrices"].map(first_outcome_price)
    if "events" in out.columns:
        out["event_id"] = out["events"].map(first_event_id)
        out["event_slug"] = out["events"].map(first_event_slug)
    if "createdAt" in out.columns:
        out["createdAt"] = pd.to_datetime(out["createdAt"], utc=True, errors="coerce")
    if "endDateIso" in out.columns:
        out["endDate_parsed"] = pd.to_datetime(out["endDateIso"], utc=True, errors="coerce")
    elif "endDate" in out.columns:
        out["endDate_parsed"] = pd.to_datetime(out["endDate"], utc=True, errors="coerce")
    return out
