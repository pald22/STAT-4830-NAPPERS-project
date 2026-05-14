"""Run the Polymarket mispricing baseline pipeline (CLI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from polymarket.backtest import MispricingConfig, run_walk_forward
from polymarket.diagnostics import plot_run
from polymarket.load import enrich_snapshot, load_polymarket_snapshot


def main() -> None:
    p = argparse.ArgumentParser(description="Polymarket mispricing portfolio baseline")
    p.add_argument("--csv", type=Path, default=None, help="Path to Polymarket CSV (snapshot or daily panel)")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "polymarket_mispricing")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--kappa", type=float, default=0.05)
    p.add_argument("--ridge-alpha", type=float, default=5.0)
    args = p.parse_args()

    path = args.csv
    if path is None:
        for candidate in (
            ROOT / "polymarket_daily_panel_60plus.csv",
            ROOT / "polymarket_markets_rich.csv",
            ROOT / "data" / "polymarket_markets_rich.csv",
        ):
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError("No Polymarket CSV found (checked 60plus panel and markets snapshots).")

    raw = load_polymarket_snapshot(path)
    df = enrich_snapshot(raw)

    cfg = MispricingConfig(
        n_folds=args.folds,
        ridge_alpha=args.ridge_alpha,
        gamma=args.gamma,
        kappa=args.kappa,
    )
    result = run_walk_forward(df, cfg=cfg, out_dir=args.out)
    plot_run(result, out_dir=args.out)

    print("Summary:", result["summary"])


if __name__ == "__main__":
    main()
