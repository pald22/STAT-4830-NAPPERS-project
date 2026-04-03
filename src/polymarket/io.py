"""Save pipeline outputs (weights, trades, pnl, metrics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_run(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(result.get("weights"), pd.DataFrame) and len(result["weights"]):
        result["weights"].to_csv(out_dir / "weights.csv", index=False)
    if isinstance(result.get("trades"), pd.DataFrame) and len(result["trades"]):
        result["trades"].to_csv(out_dir / "trades.csv", index=False)
    if isinstance(result.get("pnl"), pd.DataFrame) and len(result["pnl"]):
        result["pnl"].to_csv(out_dir / "pnl_by_fold.csv", index=False)

    metrics = result.get("metrics", [])
    with open(out_dir / "metrics_by_fold.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = result.get("summary", {})
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    lines = [
        "Polymarket mispricing baseline — summary",
        "",
        f"mean_objective: {summary.get('mean_objective', '')}",
        f"mean_mse_p_vs_implied: {summary.get('mean_mse', '')}",
        f"mean_turnover: {summary.get('mean_turnover', '')}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
