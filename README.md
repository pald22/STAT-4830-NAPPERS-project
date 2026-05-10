# STAT 4830 NAPPERS Project — Reproducibility Guide

This repository contains three related research tracks:

1. **Equity portfolio optimization** on the included S&P 500 monthly data.
2. **Polymarket mispricing optimization** using a leakage-safe walk-forward portfolio optimizer.
3. **Polymarket daily-panel experiments** covering predictability, liquidity, cross-sectional factors, unresolved-market risk, and validation design.

The original course-template README has been preserved as `README_course_template.md`. This top-level README is the practical reproduction guide for the project code and outputs.

---

## 1. Repository map

```text
STAT-4830-NAPPERS-project-main/
├── README.md                                      # this reproduction guide
├── README_course_template.md                      # original course template README
├── data/
│   ├── stock_market/
│   │   ├── README.md
│   │   └── sp500_monthly (1) (1).csv              # S&P 500 monthly panel
│   └── prediction_market/
│       ├── polymarket_daily_panel_60plus.csv      # daily Polymarket panel, 60+ day markets
│       ├── polymarket_markets_rich.csv            # rich static market snapshot
│       └── polymarket_time_data.zip
├── docs/
│   └── polymarket_panel/
│       ├── polymarket_mispricing_pipeline.md
│       ├── master_experiment_plan.md
│       ├── exp_01_persistent_predictability.md
│       ├── exp_02_liquidity_forecastability.md
│       ├── exp_03_cross_sectional_factors.md
│       ├── exp_04_unresolved_market_risk.md
│       ├── exp_05_validation_design.md
│       └── final_panel_time_series_report.md
├── notebooks/
│   ├── demo_01_optimization_equity_and_polymarket.ipynb
│   ├── demo_02_polymarket_panel_experiments.ipynb
│   └── Week*.ipynb                                # earlier development notebooks
├── scripts/
│   ├── run_polymarket_mispricing.py
│   └── run_panel_experiments.py
├── src/
│   ├── data_loader.py                             # equity data helpers
│   ├── features.py                                # equity feature helpers
│   ├── model.py                                   # equity moment estimator + optimizer
│   └── polymarket/
│       ├── load.py
│       ├── features.py
│       ├── execution.py
│       ├── model_baseline.py
│       ├── optimizer.py
│       ├── backtest.py
│       ├── diagnostics.py
│       ├── io.py
│       └── panel_experiments.py
├── tests/
│   └── test_basic.py
└── Outputs/
    ├── polymarket_mispricing/
    ├── polymarket_panel_experiments/
    ├── demo_optimization/
    └── demo_polymarket_panel_experiments/
```

---

## 2. Environment setup

Use Python 3.10 or newer. The notebooks and tests were smoke-tested in the container with Python 3.13, pandas, NumPy, scikit-learn, matplotlib, Jupyter, and pytest.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell/CMD equivalent

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn jupyter nbformat nbclient pytest
```

The older weekly notebooks used PyTorch directly. The reusable equity optimizer in `src/model.py` is now NumPy-based for portability, but if you want to run the old torch-heavy notebooks, also install PyTorch:

```bash
python -m pip install torch
```

Use `python -m pytest` rather than bare `pytest` if your environment has multiple Python installations:

```bash
python -m pytest -q
```

Expected result:

```text
3 passed
```

---

## 3. Important path conventions

Set `PYTHONPATH=src` when running the command-line scripts so Python can import the `polymarket` package:

```bash
export PYTHONPATH=src          # macOS/Linux
# set PYTHONPATH=src           # Windows CMD
# $env:PYTHONPATH="src"        # Windows PowerShell
```

Outputs in this repo use the capitalized folder `Outputs/`. This matters on Linux/macOS because paths are case-sensitive.

The two new demo notebooks find the repository root automatically and add `src/` to the import path, so they can be launched from either the repo root or the `notebooks/` directory.

---

## 4. New demo notebooks

### Demo 1: optimization pipelines

Path:

```text
notebooks/demo_01_optimization_equity_and_polymarket.ipynb
```

This notebook is the compact optimization demo. It runs two workflows:

1. **Equity optimizer** on `data/stock_market/sp500_monthly (1) (1).csv`.
2. **Polymarket mispricing optimizer** on a deterministic subset of `data/prediction_market/polymarket_daily_panel_60plus.csv`.

Launch it interactively:

```bash
jupyter notebook notebooks/demo_01_optimization_equity_and_polymarket.ipynb
```

What it shows:

- Rolling equity moment estimation with `src.model.estimate_moments`.
- Long-only simplex optimization with risk and turnover penalties via `src.model.optimize_weights`.
- Polymarket daily-panel enrichment, `p_hat` modeling, executable Yes-price estimation, projected-gradient portfolio optimization, and diagnostic outputs.
- Saved demo artifacts under `Outputs/demo_optimization/`.

Approximate quick-demo results on the included data:

| Demo section | Key result |
|---|---:|
| Equity optimized annual Sharpe | about `0.947` |
| Equity equal-weight same-universe annual Sharpe | about `0.925` |
| Equity optimized max drawdown | about `-0.192` |
| Polymarket demo periods | `55` |
| Polymarket mean objective | about `0.0284` |
| Polymarket mean MSE | about `0.0030` |
| Polymarket mean turnover | about `0.0849` |

These are quick smoke-demo numbers, not the final full-run report. The point of the notebook is to make the optimization mechanics easy to inspect and rerun.

### Demo 2: Polymarket panel experiments

Path:

```text
notebooks/demo_02_polymarket_panel_experiments.ipynb
```

Launch it interactively:

```bash
jupyter notebook notebooks/demo_02_polymarket_panel_experiments.ipynb
```

What it shows:

- Full daily-panel load from `polymarket_daily_panel_60plus.csv`.
- Feature construction from `src/polymarket/panel_experiments.py`.
- The 60% / 20% / 20% chronological split and no-overlap sanity checks.
- All five experiment functions.
- Generated markdown reports, figures, and tables under `Outputs/demo_polymarket_panel_experiments/`.

The notebook reproduces the same experiment family as the CLI script, while keeping the outputs separate from the checked-in production reports.

---

## 5. Equity optimization reproduction

### 5.1 Data

The equity data lives at:

```text
data/stock_market/sp500_monthly (1) (1).csv
```

Important columns used by the demo and old notebooks:

- `date`: monthly timestamp.
- `permno`: asset identifier.
- `ret`: monthly return.
- `prc`: price, used for market-cap screening.
- `shrout`: shares outstanding, used with price for market-cap screening.
- `shrcd`, `exchcd`: CRSP-style share-code and exchange filters.

### 5.2 Core objective

The equity optimizer solves a long-only rolling portfolio problem:

```text
maximize_w  mu_t' w - gamma * w' Sigma_t w - kappa * ||w - w_prev||_1
subject to  w >= 0, sum(w) = 1
```

Implementation:

- `src/model.py::estimate_moments(window_returns)` estimates `mu_t` and `Sigma_t` from a rolling window.
- `src/model.py::optimize_weights(mu, Sigma, w_prev, gamma, kappa, steps, lr)` applies projected subgradient ascent and projects back to the simplex.

### 5.3 Fast reproduction path

Run the new optimization demo notebook:

```bash
jupyter notebook notebooks/demo_01_optimization_equity_and_polymarket.ipynb
```

The equity section uses a deterministic reduced universe for speed:

- top market-cap names at each rebalance date,
- rolling covariance window,
- previous-weight turnover alignment,
- equal-weight same-universe benchmark.

### 5.4 Older development notebooks

Earlier equity development lives in the `notebooks/Week*.ipynb` files, especially the Week 10 notebooks. Some of those were built for Google Colab and expect manual file upload. The new demo notebook is preferable for a local, reproducible run because it uses the checked-in CSV directly and avoids manual upload.

---

## 6. Polymarket mispricing optimization reproduction

### 6.1 Architecture

The Polymarket optimization code is split into small modules:

| File | Responsibility |
|---|---|
| `src/polymarket/load.py` | Load CSVs, parse `outcomePrices`, parse event IDs/slugs, normalize timestamps. |
| `src/polymarket/features.py` | Build fold-safe features and fit `StandardScaler` on train only. |
| `src/polymarket/execution.py` | Approximate executable Yes price and liquidity loading. |
| `src/polymarket/model_baseline.py` | Ridge regression on `logit(implied_0)` to estimate `p_hat`. |
| `src/polymarket/optimizer.py` | Projected subgradient optimizer with contract, event, gross, and liquidity constraints. |
| `src/polymarket/backtest.py` | Walk-forward loop, train-only fitting, next-step realized proxy PnL. |
| `src/polymarket/io.py` | Save weights, trades, PnL, metrics, and summary files. |
| `src/polymarket/diagnostics.py` | Save diagnostic plots. |
| `scripts/run_polymarket_mispricing.py` | CLI entry point. |

### 6.2 Objective

For each decision date, the optimizer estimates edge:

```text
edge_i = p_hat_i - c_exec_i
```

and maximizes:

```text
F(w) = sum_i w_i * edge_i
       - gamma * sum_i w_i^2 * p_hat_i * (1 - p_hat_i)
       - kappa * ||w - w_prev||_1
```

Main constraints:

- gross weight `sum(w) <= 1`,
- nonnegative long-Yes weights,
- per-contract cap,
- per-event cap,
- liquidity-loading budget.

### 6.3 Fast notebook reproduction

The fastest reliable reproduction is the Polymarket section of:

```bash
jupyter notebook notebooks/demo_01_optimization_equity_and_polymarket.ipynb
```

This uses a deterministic subset of the daily panel and writes outputs to:

```text
Outputs/demo_optimization/polymarket_mispricing/
```

Expected files:

```text
diagnostics.png
metrics_by_fold.json
pnl_by_fold.csv
summary.json
summary.txt
trades.csv
weights.csv
```

### 6.4 CLI reproduction

Run from the repo root:

```bash
export PYTHONPATH=src
python scripts/run_polymarket_mispricing.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_mispricing \
  --folds 5 \
  --gamma 2.0 \
  --kappa 0.05 \
  --ridge-alpha 5.0
```

Notes:

- The daily-panel CLI refits a model for each decision date and can take several minutes on a full run.
- The rich snapshot file `polymarket_markets_rich.csv` is useful for schema exploration, but the current walk-forward backtest needs repeated price observations per market to compute `_next_implied`. Use the daily panel for optimization backtests.
- If you only need a fast sanity check, run the new demo notebook instead of the full CLI.

---

## 7. Polymarket panel experiments reproduction

### 7.1 Architecture

The five panel experiments are implemented in one module:

```text
src/polymarket/panel_experiments.py
```

Core functions:

| Function | Purpose |
|---|---|
| `load_panel(csv_path)` | Load and type the daily panel. |
| `add_features(df)` | Add lags, returns, rolling volatility, liquidity, maturity, and cross-sectional features. |
| `split_60_20_20(df)` | Make chronological train/validation/test split. |
| `sanity_check_no_overlap(train, valid, test)` | Verify no split overlap and chronological order. |
| `run_exp_01(df, out_dir)` | Persistent predictability net of costs. |
| `run_exp_02(df, out_dir)` | Liquidity and forecastability. |
| `run_exp_03(df, out_dir)` | Cross-sectional factors. |
| `run_exp_04(df, out_dir)` | Unresolved market risk/reliability. |
| `run_exp_05(df, out_dir)` | Validation design stress test. |
| `to_markdown(exp_name, result, out_path)` | Write per-experiment markdown report. |

The experiment designs are described in:

```text
docs/polymarket_panel/exp_01_persistent_predictability.md
docs/polymarket_panel/exp_02_liquidity_forecastability.md
docs/polymarket_panel/exp_03_cross_sectional_factors.md
docs/polymarket_panel/exp_04_unresolved_market_risk.md
docs/polymarket_panel/exp_05_validation_design.md
```

### 7.2 Fast notebook reproduction

Run:

```bash
jupyter notebook notebooks/demo_02_polymarket_panel_experiments.ipynb
```

Outputs are written to:

```text
Outputs/demo_polymarket_panel_experiments/
```

### 7.3 CLI reproduction

Run:

```bash
export PYTHONPATH=src
python scripts/run_panel_experiments.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_panel_experiments
```

Expected output tree:

```text
Outputs/polymarket_panel_experiments/
├── master_report.md
├── reports/
│   ├── exp_01_report.md
│   ├── exp_02_report.md
│   ├── exp_03_report.md
│   ├── exp_04_report.md
│   └── exp_05_report.md
├── exp_01/
│   ├── fig_exp01_cum_pnl.png
│   └── table_exp01_decile_returns.csv
├── exp_02/
│   ├── fig_exp02_rankic_bucket.png
│   └── table_exp02_bucket_metrics.csv
├── exp_03/
│   └── table_exp03_incremental.csv
├── exp_04/
│   └── table_exp04_reliability.csv
└── exp_05/
    ├── fig_exp05_val_vs_test.png
    └── table_exp05_splitter_compare.csv
```

### 7.4 Expected headline results

The checked-in report `docs/polymarket_panel/final_panel_time_series_report.md` summarizes the full panel run. The important reported values are:

| Experiment | Metric | Reported value |
|---|---|---:|
| Exp 1: Persistent predictability | `r2_test` | `-0.0187` |
| Exp 1: Persistent predictability | `mae_test` | `0.0944` |
| Exp 1: Persistent predictability | `directional_accuracy_test` | `0.5362` |
| Exp 1: Persistent predictability | `rank_ic_test` | `0.0769` |
| Exp 1: Persistent predictability | `net_sharpe_test` | `0.7695` |
| Exp 1: Persistent predictability | `turnover_test` | `0.2615` |
| Exp 2: Liquidity and forecastability | `mean_bucket_rank_ic` | `0.0915` |
| Exp 3: Cross-sectional factors | `delta_r2_full_minus_base` | `0.0080` |
| Exp 4: Unresolved market risk | `brier_directional` | about `0.258` |
| Exp 4: Unresolved market risk | `interval_coverage_1sigma` | about `0.947` |
| Exp 4: Unresolved market risk | `mae_adjusted` | about `0.0724` |
| Exp 5: Validation design | `val_test_corr` | `-0.6091` |

Small numerical differences can occur across scikit-learn versions, especially in the random-forest reliability experiment, but the qualitative conclusions should match.

### 7.5 Sanity checks

Each experiment runner returns a `checks` dictionary. The checked-in and demo runs enforce:

- no train/validation overlap,
- no train/test overlap,
- no validation/test overlap,
- chronological split order,
- validation-only hyperparameter selection where applicable,
- train-only liquidity bucket cutoffs,
- train-frozen factor definitions,
- no terminal-outcome peeking in unresolved-risk proxy labels.

The expected overall status is `PASS` for all five experiments.

---

## 8. End-to-end reproduction order

A clean full reproduction from the repo root is:

```bash
# 1. Create environment and install dependencies
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn jupyter nbformat nbclient pytest

# 2. Run tests
python -m pytest -q

# 3. Run full Polymarket panel experiments
export PYTHONPATH=src
python scripts/run_panel_experiments.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_panel_experiments

# 4. Run or inspect demo notebooks
jupyter notebook notebooks/demo_01_optimization_equity_and_polymarket.ipynb
jupyter notebook notebooks/demo_02_polymarket_panel_experiments.ipynb

# 5. Optional: run full Polymarket mispricing optimizer
python scripts/run_polymarket_mispricing.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_mispricing
```

For a shorter run, do steps 1, 2, and the two demo notebooks only.

---

## 9. Interpretation guide

### Equity optimizer

The equity result asks whether a rolling, risk-aware, turnover-aware portfolio can behave differently from equal weight on the same dynamic universe. The demo confirms the optimizer is not simply reproducing equal weights, and it reports the performance and turnover trade-off.

### Polymarket mispricing optimizer

The mispricing optimizer is a research baseline, not a live trading system. It demonstrates:

- fold-safe feature scaling,
- model-implied probability estimation,
- conservative executable price construction,
- constrained portfolio sizing,
- turnover accounting,
- diagnostic artifacts.

The daily-panel next-step PnL proxy is useful for pipeline testing, but it is not a terminal-resolution alpha claim.

### Polymarket panel experiments

The panel experiments provide the stronger research evidence. The final report’s synthesis is:

- Direction/rank signal exists even when absolute `R^2` is weak.
- Liquidity regime matters for forecastability.
- Cross-sectional features add modest incremental value.
- Unresolved-risk haircuts improve reliability.
- Validation design is fragile; a single holdout can mislead model selection.

---

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'polymarket'`

Run scripts with:

```bash
export PYTHONPATH=src
```

The new notebooks handle this automatically.

### `ModuleNotFoundError: No module named 'src'` in tests

Run tests as:

```bash
python -m pytest -q
```

This ensures the current repository root is on the Python import path.

### Case-sensitive output path confusion

Use `Outputs/`, not `outputs/`, on case-sensitive systems.

### Old notebooks ask for manual upload

Use the new demo notebooks for local reproduction. The old weekly notebooks document development history and may contain Colab-specific upload cells.

### Full mispricing run is slow

The full daily-panel mispricing CLI refits the baseline model at each decision date. Use the reduced demo in `demo_01_optimization_equity_and_polymarket.ipynb` for a quick functionality check.

---

## 11. Files changed for reproducibility

The following reproducibility updates were made:

- Added `notebooks/demo_01_optimization_equity_and_polymarket.ipynb`.
- Added `notebooks/demo_02_polymarket_panel_experiments.ipynb`.
- Added reusable equity optimizer implementations to `src/model.py`.
- Added `src/__init__.py` for robust package imports.
- Fixed the missing `Tuple` import in `src/features.py`.
- Updated script default paths to use checked-in data under `data/prediction_market/` and outputs under `Outputs/`.
- Updated `src/polymarket/load.py` to find `polymarket_markets_rich.csv` under `data/prediction_market/`.
- Preserved the original course README as `README_course_template.md`.

---

## 12. Research documentation index

For deeper context, read these documents in order:

1. `docs/polymarket_panel/polymarket_mispricing_pipeline.md`
2. `docs/polymarket_panel/master_experiment_plan.md`
3. `docs/polymarket_panel/exp_01_persistent_predictability.md`
4. `docs/polymarket_panel/exp_02_liquidity_forecastability.md`
5. `docs/polymarket_panel/exp_03_cross_sectional_factors.md`
6. `docs/polymarket_panel/exp_04_unresolved_market_risk.md`
7. `docs/polymarket_panel/exp_05_validation_design.md`
8. `docs/polymarket_panel/final_panel_time_series_report.md`

