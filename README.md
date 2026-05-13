# STAT 4830 NAPPERS Project

## Portfolio Optimization for Equities and Prediction Markets

This repository contains our final STAT 4830 project. The project began as a dynamic equity portfolio optimizer and later expanded into prediction markets using Polymarket and Kalshi-style market data. The common theme across both parts is portfolio construction under uncertainty: given noisy estimates, market prices, and realistic constraints, how should capital be allocated?

The equity portion provides the original optimization framework. The prediction-market portion is the main extension: we adapt the same allocation logic to contracts whose prices can be interpreted as market-implied probabilities, then study mispricing, liquidity, validation design, and unresolved-market risk.

## Main question

Can a constrained optimization framework be adapted from stock portfolios to prediction-market contracts, and what does the resulting pipeline reveal about predictability, liquidity, and validation in these markets?

We do not claim to have produced a live trading system or a reliably profitable strategy. The goal is to build a reproducible research pipeline and evaluate where the framework appears informative, fragile, or limited by the data.

---

## Repository structure

```text
STAT-4830-NAPPERS-project/
├── data/
│   ├── stock_market/
│   │   └── sp500_monthly (1) (1).csv
│   └── prediction_market/
│       ├── polymarket_daily_panel_60plus.csv
│       ├── polymarket_markets_rich.csv
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
│   └── Week*.ipynb
├── scripts/
│   ├── run_polymarket_mispricing.py
│   └── run_panel_experiments.py
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
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

The `Week*.ipynb` notebooks document earlier development work. For reproducibility, use the two demo notebooks and the scripts in `scripts/`.

---

## Environment setup

Use Python 3.10 or newer.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn jupyter nbformat nbclient pytest
```

Run the basic tests:

```bash
python -m pytest -q
```

Expected result:

```text
3 passed
```

When running scripts from the command line, set the import path:

```bash
export PYTHONPATH=src          # macOS/Linux
# $env:PYTHONPATH="src"        # Windows PowerShell
```

---

## Fast reproduction path

For a quick reproduction of the main project outputs, run these two notebooks:

```bash
jupyter notebook notebooks/demo_01_optimization_equity_and_polymarket.ipynb
jupyter notebook notebooks/demo_02_polymarket_panel_experiments.ipynb
```

### Demo 1: optimization framework

`notebooks/demo_01_optimization_equity_and_polymarket.ipynb`

This notebook shows both versions of the optimizer:

1. Equity portfolio optimization on the included S&P 500 monthly data.
2. Polymarket mispricing optimization on a deterministic subset of the daily prediction-market panel.

It writes demo outputs to:

```text
Outputs/demo_optimization/
```

The equity section demonstrates rolling moment estimation, long-only optimization, turnover penalties, and comparison to an equal-weight benchmark.

The Polymarket section demonstrates probability estimation, executable-price approximation, constrained position sizing, turnover accounting, and diagnostic output.

### Demo 2: prediction-market panel experiments

`notebooks/demo_02_polymarket_panel_experiments.ipynb`

This notebook reproduces the main prediction-market experiment family using:

```text
data/prediction_market/polymarket_daily_panel_60plus.csv
```

It writes outputs to:

```text
Outputs/demo_polymarket_panel_experiments/
```

The notebook runs the same five experiment designs as the command-line script, but keeps the demo outputs separate from the checked-in full-run outputs.

---

## Equity optimization

The equity optimizer is the original framework that motivated the project.

Data:

```text
data/stock_market/sp500_monthly (1) (1).csv
```

Core objective:

```text
maximize_w  mu_t' w - gamma * w' Sigma_t w - kappa * ||w - w_prev||_1
subject to  w >= 0, sum(w) = 1
```

Main implementation:

```text
src/model.py
```

Key functions:

```text
estimate_moments(window_returns)
optimize_weights(mu, Sigma, w_prev, gamma, kappa, steps, lr)
```

This part of the project asks whether a rolling, risk-aware, turnover-aware optimizer can improve on equal weighting over the same dynamic universe. It also exposed one of the main limitations of the project: portfolio optimization is very sensitive to noisy estimates of expected returns and covariances.

---

## Prediction-market mispricing optimizer

The prediction-market extension adapts the equity allocation framework to Polymarket-style contracts.

Instead of estimating stock returns, we estimate whether a contract is mispriced relative to an executable market price.

For each contract:

```text
edge_i = p_hat_i - c_exec_i
```

where:

- `p_hat_i` is the model-estimated probability,
- `c_exec_i` is the estimated executable Yes price.

The optimizer maximizes:

```text
F(w) = sum_i w_i * edge_i
       - gamma * sum_i w_i^2 * p_hat_i * (1 - p_hat_i)
       - kappa * ||w - w_prev||_1
```

Subject to:

- nonnegative long-Yes positions,
- gross exposure cap,
- per-contract cap,
- per-event cap,
- liquidity-based position limits,
- cash allowed.

Main implementation:

```text
src/polymarket/
```

Important files:

```text
src/polymarket/load.py              # load and clean prediction-market data
src/polymarket/features.py          # construct fold-safe features
src/polymarket/execution.py         # approximate executable prices
src/polymarket/model_baseline.py    # baseline probability model
src/polymarket/optimizer.py         # constrained optimizer
src/polymarket/backtest.py          # walk-forward backtest loop
src/polymarket/diagnostics.py       # plots and diagnostics
```

Run the full mispricing optimizer:

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

Expected output files include:

```text
Outputs/polymarket_mispricing/
├── diagnostics.png
├── metrics_by_fold.json
├── pnl_by_fold.csv
├── summary.json
├── summary.txt
├── trades.csv
└── weights.csv
```

Important limitation: this is a research baseline, not a live trading system. The next-step PnL proxy is useful for testing the pipeline, but it should not be interpreted as proof of terminal-resolution profitability.

---

## Prediction-market panel experiments

The strongest prediction-market evidence comes from the panel experiments. These experiments are implemented in:

```text
src/polymarket/panel_experiments.py
```

Run the full experiment suite:

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

The five experiments are:

1. **Persistent predictability**  
   Tests whether contract-level signals have directional or ranking value after costs.

2. **Liquidity and forecastability**  
   Studies whether signal quality differs across liquidity buckets.

3. **Cross-sectional factors**  
   Tests whether additional market-level features add predictive value beyond baseline price information.

4. **Unresolved-market risk**  
   Studies the reliability problem created by markets that have not yet resolved.

5. **Validation design**  
   Tests how sensitive conclusions are to the train/validation/test split.

The experiment design documents are in:

```text
docs/polymarket_panel/
```

The final panel report is:

```text
docs/polymarket_panel/final_panel_time_series_report.md
```

---

## Headline results from the prediction-market panel

The checked-in full panel run reports the following headline values:

| Experiment | Metric | Reported value |
|---|---:|---:|
| Exp 1: Persistent predictability | `r2_test` | -0.0187 |
| Exp 1: Persistent predictability | `mae_test` | 0.0944 |
| Exp 1: Persistent predictability | `directional_accuracy_test` | 0.5362 |
| Exp 1: Persistent predictability | `rank_ic_test` | 0.0769 |
| Exp 1: Persistent predictability | `net_sharpe_test` | 0.7695 |
| Exp 1: Persistent predictability | `turnover_test` | 0.2615 |
| Exp 2: Liquidity and forecastability | `mean_bucket_rank_ic` | 0.0915 |
| Exp 3: Cross-sectional factors | `delta_r2_full_minus_base` | 0.0080 |
| Exp 4: Unresolved market risk | `brier_directional` | about 0.258 |
| Exp 4: Unresolved market risk | `interval_coverage_1sigma` | about 0.947 |
| Exp 4: Unresolved market risk | `mae_adjusted` | about 0.0724 |
| Exp 5: Validation design | `val_test_corr` | -0.6091 |

Small numerical differences may occur across package versions, especially in models with randomness. The qualitative conclusions should be the same.

---

## Interpretation of results

The equity optimizer shows that the optimization framework works mechanically, but it is fragile because rolling estimates of returns and covariances are noisy.

The prediction-market results are more central to the final project. They suggest:

- directional and ranking signal can exist even when absolute R-squared is weak;
- liquidity matters for forecastability;
- cross-sectional features add modest incremental value;
- unresolved markets create reliability problems;
- validation design is fragile, so a single split can be misleading.

The main takeaway is not that we found a guaranteed trading strategy. The main takeaway is that prediction-market optimization requires careful treatment of probability estimates, executable prices, liquidity, time splits, and unresolved outcomes.

---

## Full reproduction order

A clean full reproduction from the repository root is:

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn jupyter nbformat nbclient pytest

# 3. Run tests
python -m pytest -q

# 4. Run full Polymarket panel experiments
export PYTHONPATH=src
python scripts/run_panel_experiments.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_panel_experiments

# 5. Run demo notebooks
jupyter notebook notebooks/demo_01_optimization_equity_and_polymarket.ipynb
jupyter notebook notebooks/demo_02_polymarket_panel_experiments.ipynb

# 6. Optional: run full Polymarket mispricing optimizer
python scripts/run_polymarket_mispricing.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_mispricing
```

For a shorter check, run only steps 1-3 and the two demo notebooks.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'polymarket'`

Set the Python path before running scripts:

```bash
export PYTHONPATH=src
```

### Tests cannot import local modules

Run:

```bash
python -m pytest -q
```

instead of bare `pytest`.

### Output folder issues

Use:

```text
Outputs/
```

not:

```text
outputs/
```

The capitalization matters on case-sensitive systems.

### Older notebooks ask for manual upload

Use the two demo notebooks for reproduction. The older weekly notebooks are kept as development history and may contain Colab-specific upload cells.

### Full mispricing run is slow

The full mispricing script refits the baseline model repeatedly across decision dates. Use the demo notebook for a faster functionality check.

---

## Suggested reading order

For the project logic and final results, read:

1. `docs/polymarket_panel/polymarket_mispricing_pipeline.md`
2. `docs/polymarket_panel/master_experiment_plan.md`
3. `docs/polymarket_panel/exp_01_persistent_predictability.md`
4. `docs/polymarket_panel/exp_02_liquidity_forecastability.md`
5. `docs/polymarket_panel/exp_03_cross_sectional_factors.md`
6. `docs/polymarket_panel/exp_04_unresolved_market_risk.md`
7. `docs/polymarket_panel/exp_05_validation_design.md`
8. `docs/polymarket_panel/final_panel_time_series_report.md`

---

## Project status

This repository contains a reproducible research pipeline for equity and prediction-market optimization. The equity portion provides the original optimization base. The prediction-market portion is the main final extension and includes the most important experiments, diagnostics, and interpretation.
