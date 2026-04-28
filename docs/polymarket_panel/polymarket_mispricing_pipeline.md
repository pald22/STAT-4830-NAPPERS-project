# Polymarket mispricing portfolio (baseline pipeline)

## What the exploration notebook provides

The notebook `polymarket_exploration.ipynb` loads **`polymarket_markets_rich.csv`** (500×92 in the checked-in snapshot), resolves the file from the repo root or `data/`, and builds:

| Artifact | Description |
|----------|-------------|
| Raw columns | IDs, `question`, `conditionId`, `slug`, `outcomes`, `outcomePrices`, volume/liquidity (`volumeNum`, `liquidityNum`, CLOB variants), order book (`bestBid`, `bestAsk`, `spread`), flags (`active`, `closed`, …), JSON `events`, timestamps (`createdAt`, `endDate` / `endDateIso`), price changes, etc. |
| `implied_0` | First entry of JSON `outcomePrices` (binary “Yes” price in the sample). |
| `event_id` / `event_slug` | Parsed from the first object in the `events` JSON array (for grouping). |

The checked-in snapshot is **all open markets** (`closed=False`); there are **no realized outcomes** for resolution PnL. The pipeline therefore uses **walk-forward splits by `createdAt`** (earlier listings = training folds) and **simulated** Bernoulli payoffs under the **risk-neutral** probability given by market `implied_0` for diagnostics only.

## Assumptions (read before trusting numbers)

1. **Single CSV snapshot** — Not a full panel of historical prices; “time” is **listing order** via `createdAt`, not a sequence of market prices for the same contract.
2. **p_hat model** — Ridge regression on `logit(implied_0)` using **order-book / liquidity / time-to-expiry** features. It learns a **calibrated** mapping to the current mid; **generalization** across folds drives non-zero test error.
3. **Execution** — `c_exec` approximates buying **Yes** via `bestAsk` when present, else mid or implied + half-spread (see `execution.py`).
4. **Optimization** — Projected **subgradient** ascent with a **cyclic approximate projection** (sum of weights ≤ 1, per-contract cap, per-event cap, liquidity loading constraint). This is a research baseline, not a production solver.
5. **PnL** — `pnl_sim_risk_neutral` draws `y ~ Bernoulli(implied)` independently of the model; under exact risk-neutral pricing, **expected** PnL against that data-generating process is not informative about alpha — use **objective value**, **MSE**, **turnover**, and **constraint slack** instead.

## Layout

- `src/polymarket/load.py` — CSV load + enrich (same logic as the notebook).
- `src/polymarket/features.py` — Features and `StandardScaler` **fit on train only** per fold.
- `src/polymarket/execution.py` — Executable Yes price and liquidity loadings for constraints.
- `src/polymarket/model_baseline.py` — Ridge → `p_hat`.
- `src/polymarket/optimizer.py` — Projected subgradient + projection.
- `src/polymarket/backtest.py` — Walk-forward loop, `MispricingConfig`.
- `src/polymarket/io.py` — Writes CSV/JSON/text under `outputs/`.
- `src/polymarket/diagnostics.py` — Plots.
- `scripts/run_polymarket_mispricing.py` — CLI entry.

## Run instructions

From the repository root, with `src` on `PYTHONPATH`:

```bash
uv run python scripts/run_polymarket_mispricing.py --out outputs/polymarket_mispricing
```

Or:

```bash
set PYTHONPATH=src
python scripts/run_polymarket_mispricing.py
```

Optional flags: `--csv path/to/polymarket_markets_rich.csv`, `--folds 5`, `--gamma 2.0`, `--kappa 0.05`, `--ridge-alpha 5.0`.

Outputs:

- `weights.csv`, `trades.csv`, `pnl_by_fold.csv`
- `metrics_by_fold.json`, `summary.json`, `summary.txt`
- `diagnostics.png`

The notebook `notebooks/polymarket_mispricing_pipeline.ipynb` runs the same pipeline in a few cells.
