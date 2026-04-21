import requests
import pandas as pd
import ast
import time
from typing import Any, Optional, List

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


# ----------------------------
# Helpers
# ----------------------------

def safe_parse_list(value: Any) -> List[Any]:
    """
    Parse list-like CSV values such as:
    '["Yes","No"]' or "['123','456']"
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            return []
        except Exception:
            return []
    return []


def get_first_token_id(clob_token_ids: Any) -> Optional[str]:
    """
    For binary markets, the first token is usually one side of the market.
    We just pick the first token so we have one consistent price series per market.
    """
    tokens = safe_parse_list(clob_token_ids)
    if not tokens:
        return None
    return str(tokens[0])


# ----------------------------
# Step 1: Fetch many markets
# ----------------------------

def fetch_markets_snapshot(
    total_markets: int = 1000,
    active: bool = True,
    closed: bool = False,
    page_size: int = 200,
) -> pd.DataFrame:
    """
    Pull many market metadata rows from Polymarket Gamma API.
    """
    rows = []
    offset = 0

    while len(rows) < total_markets:
        batch_limit = min(page_size, total_markets - len(rows))
        params = {
            "limit": batch_limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }

        resp = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        rows.extend(data)
        offset += len(data)
        print(f"Fetched {len(rows)} market rows so far...")
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    return df


# ----------------------------
# Step 2: Historical daily prices
# ----------------------------

def fetch_daily_price_history(token_id: str) -> pd.DataFrame:
    """
    Pull all available history for a token at daily granularity.

    Polymarket docs:
    - interval='max' gives all available data
    - fidelity is in minutes
    - daily data => fidelity=1440
    """
    params = {
        "market": token_id,   # docs say this is actually the asset/token id
        "interval": "max",
        "fidelity": 1440,     # 1440 minutes = 1 day
    }

    resp = requests.get(f"{CLOB_BASE}/prices-history", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    history = data.get("history", [])
    df = pd.DataFrame(history)

    if df.empty:
        return df

    df = df.rename(columns={"t": "timestamp", "p": "price"})
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["date"] = df["datetime_utc"].dt.date
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    return df


# ----------------------------
# Step 3: Build long panel dataset
# ----------------------------

def build_polymarket_daily_panel(
    markets_df: pd.DataFrame,
    min_days: int = 60,
    max_markets_to_try: int = 300,
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    """
    Create a long-format dataset:
    one row per market per day.

    Keeps only markets with at least min_days observations.
    """
    panel_rows = []

    # A simple ranking so we try better markets first
    # If these columns are missing, fill them with 0
    for col in ["volume", "liquidity"]:
        if col not in markets_df.columns:
            markets_df[col] = 0

    candidate_df = markets_df.copy()

    # Convert numeric-looking fields
    for col in ["volume", "liquidity"]:
        candidate_df[col] = pd.to_numeric(candidate_df[col], errors="coerce").fillna(0)

    candidate_df = candidate_df.sort_values(
        ["liquidity", "volume"],
        ascending=False
    ).reset_index(drop=True)

    tried = 0
    kept = 0

    for _, row in candidate_df.iterrows():
        if tried >= max_markets_to_try:
            break

        tried += 1

        token_id = get_first_token_id(row.get("clobTokenIds"))
        if token_id is None:
            continue

        try:
            hist = fetch_daily_price_history(token_id)
        except Exception as e:
            print(f"Skipping market {row.get('id')} due to error: {e}")
            continue

        if hist.empty:
            continue

        # Deduplicate to one row per day if needed
        hist = (
            hist.sort_values("datetime_utc")
                .drop_duplicates(subset=["date"], keep="last")
                .reset_index(drop=True)
        )

        if len(hist) < min_days:
            continue

        hist["market_id"] = row.get("id")
        hist["question"] = row.get("question")
        hist["slug"] = row.get("slug")
        hist["token_id"] = token_id
        hist["volume"] = row.get("volume")
        hist["liquidity"] = row.get("liquidity")
        hist["active"] = row.get("active")
        hist["closed"] = row.get("closed")
        hist["end_date_iso"] = row.get("endDate")
        hist["category"] = row.get("category")
        hist["outcomes"] = str(row.get("outcomes"))
        hist["outcomePrices"] = str(row.get("outcomePrices"))

        panel_rows.append(hist)
        kept += 1

        print(
            f"Kept market {kept}: {row.get('question')} "
            f"({len(hist)} daily observations)"
        )

        time.sleep(sleep_seconds)

    if not panel_rows:
        return pd.DataFrame()

    panel_df = pd.concat(panel_rows, ignore_index=True)

    # Reorder columns nicely
    preferred_cols = [
        "market_id", "question", "slug", "token_id",
        "date", "datetime_utc", "timestamp", "price",
        "volume", "liquidity", "active", "closed",
        "end_date_iso", "category", "outcomes", "outcomePrices"
    ]
    existing_cols = [c for c in preferred_cols if c in panel_df.columns]
    other_cols = [c for c in panel_df.columns if c not in existing_cols]
    panel_df = panel_df[existing_cols + other_cols]

    return panel_df


# ----------------------------
# Step 4: Main
# ----------------------------

if __name__ == "__main__":
    # Pull a large market universe first
    markets_df = fetch_markets_snapshot(
        total_markets=1000,   # increase if you want more candidate entries
        active=True,
        closed=False,
        page_size=200,
    )

    markets_df.to_csv("polymarket_markets_snapshot.csv", index=False)
    print(f"Saved snapshot with {len(markets_df)} rows.")

    # Build daily long-format panel
    panel_df = build_polymarket_daily_panel(
        markets_df=markets_df,
        min_days=60,            # your team asked for 60 time periods
        max_markets_to_try=300, # tries many markets to get enough usable ones
        sleep_seconds=0.15,
    )

    panel_df.to_csv("polymarket_daily_panel_60plus.csv", index=False)
    print(f"Saved panel with {len(panel_df)} rows.")
    print(panel_df.head())

    # Optional summary file: how many days per market
    summary_df = (
        panel_df.groupby(["market_id", "question", "slug"], as_index=False)
        .agg(
            n_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            avg_price=("price", "mean"),
        )
        .sort_values("n_days", ascending=False)
    )

    summary_df.to_csv("polymarket_daily_panel_summary.csv", index=False)
    print(f"Saved summary with {len(summary_df)} markets.")