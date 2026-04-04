# Poly market exploration using data markets rich.csv
_Exported on 4/3/2026 at 19:53:45 EDT from Cursor (3.0.9)_

---

**User**

poly market exploration based on the data markets rich.csv

---

**Cursor**

Locating the CSV and inspecting its structure so we can build Polymarket exploration in the notebook.



Creating a complete exploration notebook: resolving the CSV path, loading with pandas, summarizing key fields, parsing JSON columns, and adding visualizations.



Here is what is set up for you.

### Data file
The export is **`polymarket_markets_rich.csv`** at the repo root (there is also a copy under `data/`). Columns include `question`, `volume` / `volumeNum`, `liquidity` / `liquidityNum`, `outcomes`, `outcomePrices` (JSON), `events` (JSON), `active`, `closed`, `spread`, `bestBid` / `bestAsk`, timing fields, etc.

### Notebook: `polymarket_exploration.ipynb`
The notebook was empty; it now contains an end-to-end exploration:

1. **Path handling** — loads `polymarket_markets_rich.csv` from the project root, or `data/polymarket_markets_rich.csv` if needed.
2. **Load** — `pd.read_csv(..., low_memory=False)` for mixed types.
3. **Schema & missingness** — `head`, full column list, top columns by missing %.
4. **Flags** — `value_counts` for `active`, `closed`, `featured`, `archived`, `restricted`, `acceptingOrders` when present.
5. **Numeric summary** — `describe` on volume/liquidity/volume windows/spread/order book fields.
6. **Top markets** — 15 rows by `volumeNum` (or `volume`).
7. **Parsed fields** — `implied_0` from the first value in `outcomePrices`; `event_title` from the first object in `events`.
8. **Plots** — log histogram of volume; log–log scatter volume vs liquidity; histogram of `implied_0` if parsed.
9. **End dates** — year counts from `endDateIso` or `endDate`.

Open **`polymarket_exploration.ipynb`** and run with your project **`.venv`** kernel (same idea as `notebooks/week6_implementation.ipynb`). If you want, we can add slices (e.g. only `active & ~closed`) or keyword tagging on `question` / event titles next.

---

**User**

Use the completed Polymarket exploration notebook as the starting point and implement a baseline mispricing-portfolio research pipeline.

Objective:
maximize

    sum_i w_i * (p_hat_i - c_exec_i)
    - gamma * sum_i w_i^2 * p_hat_i * (1 - p_hat_i)
    - kappa * ||w_t - w_{t-1}||_1

subject to:
- w_i >= 0
- sum_i w_i <= 1
- per-contract cap
- per-event cap if event structure exists
- liquidity cap if volume/order book depth exists

Use projected gradient as the main optimizer.
Allow cash; do not force full investment.

Tasks:
1. Inspect the existing exploration notebook and summarize available cleaned data/tables/columns.
2. Reuse existing data structures instead of rebuilding from scratch.
3. Implement:
   - feature prep
   - baseline p_hat model
   - executable price approximation
   - projected-gradient optimizer
   - backtest loop
   - diagnostics and plots
4. Save weights, trades, pnl, turnover, and summary metrics.
5. Document assumptions clearly and provide run instructions.

Use realistic no-leakage backtesting and readable modular Python code.
Start by summarizing the notebook contents and then propose the file structure before coding.

---

**Cursor**

Inspecting the exploration notebook and project layout, then summarizing findings and proposing a modular file structure before implementation.



Summarizing: the exploration notebook loads a single CSV snapshot (500×92), derives `implied_0` and `event_title` from JSON, and profiles volume/liquidity/spread. All rows are open markets — no realized outcomes. We'll implement walk-forward splits by `createdAt`, use simulated PnL where needed, and document limitations.

Proposed layout:
- `src/polymarket/` — `load.py`, `features.py`, `execution.py`, `model_baseline.py`, `optimizer.py`, `backtest.py`, `io.py`
- `scripts/run_polymarket_mispricing.py` — CLI
- `docs/polymarket_mispricing_pipeline.md` — assumptions + run instructions
- `notebooks/polymarket_mispricing_pipeline.ipynb` — thin driver

Implementing the package.



Fixing the optimizer (trust-constr for projection), tightening feature `ref_time` handling, and implementing the backtest with fold-wise alignment of `w_prev`.



Replacing the expensive trust-constr projection (called every iteration) with a fast cyclic projection onto the same constraints.



## 1) What the exploration notebook gives you

From `polymarket_exploration.ipynb` (and the same CSV logic in code):

- **Source:** `polymarket_markets_rich.csv` — in your run it is **500 rows × 92 columns** (one **cross-sectional** API snapshot).
- **Core raw fields:** `id`, `question`, `conditionId`, `slug`, `outcomes`, `outcomePrices`, `volume` / `volumeNum`, `liquidity` / `liquidityNum` (and CLOB variants), `bestBid`, `bestAsk`, `spread`, `competitive`, `active`, `closed`, `events` (JSON), `createdAt`, `endDate` / `endDateIso`, price-change columns, etc.
- **Derived in the notebook (reused in code):** `implied_0` from JSON `outcomePrices` (first price ≈ “Yes”), and event metadata from `events` (we use **`event_id`** / slug in the pipeline for grouping).
- **Empirical fact:** the saved snapshot is **all open** markets (`closed=False`), so there are **no resolutions** in-file — true resolution PnL is not available without extra data.

## 2) File structure (implemented)

| Path | Role |
|------|------|
| `src/polymarket/load.py` | `load_polymarket_snapshot`, `find_markets_csv`, `enrich_snapshot` (same enrichment as the notebook) |
| `src/polymarket/features.py` | Feature matrix + `StandardScaler` **fit on train only** per fold; `ref_time` = max train `createdAt` |
| `src/polymarket/execution.py` | `executable_yes_price` (Yes buy ≈ `bestAsk` / mid / implied+½spread), `liquidity_loadings` for the liquidity constraint |
| `src/polymarket/model_baseline.py` | Ridge on `logit(implied)` → `p_hat` |
| `src/polymarket/optimizer.py` | Projected **subgradient** ascent + **cyclic projection** (≤1 total weight, per-contract cap, per-event cap, liquidity loading) |
| `src/polymarket/backtest.py` | `MispricingConfig`, `run_walk_forward` (walk-forward by `createdAt` folds), outputs |
| `src/polymarket/io.py` | Writes `weights.csv`, `trades.csv`, `pnl_by_fold.csv`, `metrics_by_fold.json`, `summary.json`, `summary.txt` |
| `src/polymarket/diagnostics.py` | `diagnostics.png` |
| `scripts/run_polymarket_mispricing.py` | CLI |
| `docs/polymarket_mispricing_pipeline.md` | Assumptions + how to run |
| `notebooks/polymarket_mispricing_pipeline.ipynb` | Thin driver |

Existing `src/data_loader.py` / `src/features.py` (S&P) are **unchanged**; Polymarket lives under `src/polymarket/`.

## 3) Objective & method (as coded)

- **Maximize** \(\sum_i w_i ( \hat p_i - c^{\mathrm{exec}}_i ) - \gamma \sum_i w_i^2 \hat p_i(1-\hat p_i) - \kappa \|w - w_{\mathrm{prev}}\|_1\) with **\(w \ge 0\)**, **\(\sum_i w_i \le 1\)** (cash allowed), plus **per-contract**, **per-event**, and **liquidity** caps.
- **Optimizer:** subgradient on the smooth part minus an L1 subgradient term, then **projection** onto the feasible set (cyclic approximation).
- **No leakage:** within this snapshot, folds are by **`createdAt`**; scaling and Ridge are fit **only on prior folds**; `days_to_end` uses **`ref_time`** from the training set.

## 4) Outputs & run

- **Run (from repo root):**  
  `set PYTHONPATH=src` then  
  `python scripts/run_polymarket_mispricing.py --out outputs/polymarket_mispricing`  
  (or `uv run python ...` with the same `PYTHONPATH`).
- **Notebook:** `notebooks/polymarket_mispricing_pipeline.ipynb` — set `src` on the path as in the first cell.

**Assumptions** (including limits of a single snapshot and risk-neutral simulation) are spelled out in [`docs/polymarket_mispricing_pipeline.md`](docs/polymarket_mispricing_pipeline.md).

A test run completed successfully (fold 0 is skipped because there is no training set yet; later folds optimize non‑trivial weights where edge is positive). If you want, we can next add a **panel** of historical snapshots for real time-series backtesting, or tighten **execution** with fee/tick models.

