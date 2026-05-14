# STAT 4830 NAPPERS Project

## Portfolio Optimization for Equities and Prediction Markets

This repository contains our final STAT 4830 project, **Dynamic Portfolio Optimization in Equities and Prediction Markets**. The project began as a dynamic equity portfolio optimizer and later expanded into prediction markets using Polymarket and Kalshi-style market data. The common theme across both parts is portfolio construction under uncertainty: given noisy estimates, market prices, covariance risk, and realistic constraints, how should capital be allocated?

The equity portion provides the original optimization framework. The prediction-market portion is the main extension: we adapt the same allocation logic to contracts whose prices can be interpreted as market-implied probabilities, then study mispricing, liquidity, validation design, covariance-aware risk, and unresolved-market limitations.

## Main question

Can a constrained numerical optimization framework be adapted from stock portfolios to prediction-market contracts, and what does the resulting pipeline reveal about calibration, predictability, liquidity, execution assumptions, covariance risk, and validation design? The goal is to build a reproducible research pipeline and evaluate where the framework appears informative, fragile, or limited by the data.

---

## Final deliverable folder

The polished submission materials are located in:

```text
final_deliverable/
```

The folder is organized as follows:

```text
final_deliverable/
├── final_report/
│   └── Dynamic_Portfolio_Optimization_in_Equities_and_Prediction_Markets.pdf
├── implementation_code/
│   ├── STAT_4830_Portfolio_Optimization_Daily_Regime.ipynb
│   └── STAT_4830_Portfolio_Optimization_Monthly_Regime.ipynb
├── llm_exploration/
│   └── STAT_4830_LLM_Logs.pdf
├── readme/
│   └── README.tex
└── self_critiques/
    └── STAT_4830_Self_Critique_Edited.pdf
```

For the final assignment, `final_deliverable/` is the main folder to review. It contains the final report, implementation notebooks, LLM exploration logs, self-critique, and reproducibility guide.

---

## Repository structure

```text
STAT-4830-NAPPERS-project/
├── data/
│   ├── other/
│   ├── stock_market/
│   │   └── sp500_monthly (1) (1).csv
│   └── prediction_market/
│       ├── Kalshi_Data
│       ├── polymarket_daily_panel_60plus.csv
│       ├── polymarket_markets_rich.csv
│       └── polymarket_time_data.zip
├── docs/
│   ├── assignments/
│   ├── llm_exploration/
│   └── polymarket_panel/
│       ├── polymarket_mispricing_pipeline.md
│       ├── master_experiment_plan.md
│       ├── exp_01_persistent_predictability.md
│       ├── exp_02_liquidity_forecastability.md
│       ├── exp_03_cross_sectional_factors.md
│       ├── exp_04_unresolved_market_risk.md
│       ├── exp_05_validation_design.md
│       └── final_panel_time_series_report.md
├── figures/
│   └── Images/
├── final_deliverable/
├── notebooks/
│   ├── STAT_4830_Portfolio_Optimization_Daily_Regime.ipynb
│   ├── STAT_4830_Portfolio_Optimization_Monthly_Regime.ipynb
│   └── Week*.ipynb
├── other/
│   ├── scripts/
│   │   ├── run_panel_experiments.py
│   │   └── run_polymarket_mispricing.py
│   ├── demo_01_standalone_optimization_equity_and_polymarket.ipynb
│   ├── demo_02_standalone_polymarket_panel_experiments.ipynb
│   ├── polymarket_exploration.ipynb
│   ├── polymarket_exploration_60plus.ipynb
│   └── polymarket_time_data.py
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── utils.py
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
├── supplementary class setup material/
├── tests/
│   └── test_basic.py
├── README.md
├── pyproject.toml
├── uv.lock
└── Outputs/
    ├── polymarket_mispricing/
    ├── polymarket_mispricing_60d_cov/
    ├── polymarket_panel_experiments/
    ├── demo_optimization/
    └── demo_polymarket_panel_experiments/
```

The `Week*.ipynb` notebooks and other exploratory notebooks document earlier development work. For reproducibility, use the final deliverable notebooks, the two demo notebooks in `other/`, and the scripts in `other/scripts/`.

---

## Environment setup

Use Python 3.10 or newer.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn torch scipy jupyter nbformat nbclient pytest
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
jupyter notebook other/demo_01_standalone_optimization_equity_and_polymarket.ipynb
jupyter notebook other/demo_02_standalone_polymarket_panel_experiments.ipynb
```

The final deliverable versions are also available at:

```text
final_deliverable/implementation_code/
```

### Demo 1: optimization framework

`other/demo_01_standalone_optimization_equity_and_polymarket.ipynb`

This notebook shows both versions of the optimizer:

1. Equity portfolio optimization on the included S&P 500 monthly data.
2. Polymarket mispricing optimization on a deterministic subset of the daily prediction-market panel.

It writes demo outputs to:

```text
Outputs/demo_optimization/
```

The equity section demonstrates rolling moment estimation, long-only optimization, turnover penalties, and comparison to an equal-weight benchmark.

The Polymarket section demonstrates probability estimation, executable-price approximation, constrained position sizing, covariance-aware risk, turnover accounting, and diagnostic output.

### Demo 2: prediction-market panel experiments

`other/demo_02_standalone_polymarket_panel_experiments.ipynb`

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

The optimizer now supports a rolling empirical covariance penalty across active contracts. At each decision date, `build_contract_covariance` forms a 60-day history using only rows with `_decision_time` before the current decision date, pivots implied prices by contract, computes daily price changes, estimates the sample covariance, repairs invalid entries, symmetrizes the matrix, and adds a small ridge term.

With covariance enabled, the optimizer maximizes:

```text
F(w) = sum_i w_i * edge_i
       - gamma * w' Sigma w
       - kappa * ||w - w_prev||_1
```

where `Sigma` is the rolling contract covariance matrix.

If `Sigma=None`, the optimizer keeps the previous diagonal Bernoulli-risk objective:

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
src/polymarket/optimizer.py         # covariance builder and constrained optimizer
src/polymarket/backtest.py          # walk-forward backtest loop
src/polymarket/diagnostics.py       # plots and diagnostics
src/polymarket/io.py                # saved summaries and output helpers
```

Run the full mispricing optimizer from the repository root:

```bash
export PYTHONPATH=src

python other/scripts/run_polymarket_mispricing.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_mispricing \
  --folds 5 \
  --gamma 2.0 \
  --kappa 0.05 \
  --ridge-alpha 5.0
```

Run the covariance version from the repository root:

```bash
export PYTHONPATH=src

python other/scripts/run_polymarket_mispricing.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_mispricing_60d_cov
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

The covariance run also reports:

```text
mean_risk_cov
mean_risk_diag_equiv
```

Important limitation: this is a research baseline, not a live trading system. The next-step PnL proxy is useful for testing the pipeline, but it should not be interpreted as proof of terminal-resolution profitability.

---

## Prediction-market panel experiments

The strongest prediction-market evidence comes from the panel experiments. These experiments are implemented in:

```text
src/polymarket/panel_experiments.py
```

Run the full experiment suite from the repository root:

```bash
export PYTHONPATH=src

python other/scripts/run_panel_experiments.py \
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
- validation design is fragile, so a single split can be misleading;
- covariance-aware allocation is a natural extension of the original diagonal risk penalty, but it still depends on short and noisy historical price panels.

The main takeaway is not that we found a guaranteed trading strategy. The main takeaway is that prediction-market optimization requires careful treatment of probability estimates, executable prices, covariance risk, liquidity, time splits, and unresolved outcomes.

---

## Full reproduction order

A clean full reproduction from the repository root is:

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn torch scipy jupyter nbformat nbclient pytest

# 3. Run tests
python -m pytest -q

# 4. Run full Polymarket panel experiments
export PYTHONPATH=src
python other/scripts/run_panel_experiments.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_panel_experiments

# 5. Optional: run Polymarket mispricing optimizer with covariance risk
python other/scripts/run_polymarket_mispricing.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_mispricing_60d_cov

# 6. Run demo notebooks
jupyter notebook other/demo_01_standalone_optimization_equity_and_polymarket.ipynb
jupyter notebook other/demo_02_standalone_polymarket_panel_experiments.ipynb
```

For a shorter check, run only steps 1-3 and the two demo notebooks.

---

## Data

The project uses two broad classes of data.

### Equity data

The equity data are stored under:

```text
data/stock_market/
```

The monthly equity file supports the monthly equity portfolio optimization experiments. The daily equity return dataset used for the daily-regime extension is too large to include directly in the repository. Readers who would like access to the daily equity data for reproduction purposes should contact the authors at andar@sas.upenn.edu.

### Prediction-market data

The prediction-market data are stored under:

```text
data/prediction_market/
```

This folder includes:

```text
Kalshi_Data
polymarket_daily_panel_60plus.csv
polymarket_markets_rich.csv
polymarket_time_data.zip
```

The prediction-market pipeline relies primarily on the Polymarket-style data files. A key limitation is that the packaged artifact does not include final contract resolutions for all contracts, so prediction-market realized PnL is evaluated using next-step quote-to-quote changes rather than final settlement outcomes.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'polymarket'`

Set the Python path before running scripts:

```bash
export PYTHONPATH=src
```

Scripts are stored in `other/scripts/`, but they should be run from the repository root. For example:

```bash
python other/scripts/run_panel_experiments.py \
  --csv data/prediction_market/polymarket_daily_panel_60plus.csv \
  --out Outputs/polymarket_panel_experiments
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

Use the final deliverable notebooks or the two demo notebooks in `other/` for reproduction. The older weekly notebooks are kept as development history and may contain Colab-specific upload cells.

### Full mispricing run is slow

The full mispricing script refits the baseline model repeatedly across decision dates. Use the demo notebook for a faster functionality check.

---

## Suggested reading order

For the final assignment, we recommend reviewing the materials in this order:

1. `final_deliverable/final_report/Dynamic_Portfolio_Optimization_in_Equities_and_Prediction_Markets.pdf`
2. `other/demo_01_standalone_optimization_equity_and_polymarket.ipynb`
3. `final_deliverable/implementation_code/STAT_4830_Portfolio_Optimization_Monthly_Regime.ipynb`
4. `final_deliverable/implementation_code/STAT_4830_Portfolio_Optimization_Daily_Regime.ipynb`
5. `other/demo_02_standalone_polymarket_panel_experiments.ipynb`
6. `final_deliverable/llm_exploration/STAT_4830_LLM_Logs.pdf`
7. `final_deliverable/self_critiques/STAT_4830_Self_Critique_Edited.pdf`

For the prediction-market experiment logic and final results, also read:

1. `docs/polymarket_panel/polymarket_mispricing_pipeline.md`
2. `docs/polymarket_panel/master_experiment_plan.md`
3. `docs/polymarket_panel/exp_01_persistent_predictability.md`
4. `docs/polymarket_panel/exp_02_liquidity_forecastability.md`
5. `docs/polymarket_panel/exp_03_cross_sectional_factors.md`
6. `docs/polymarket_panel/exp_04_unresolved_market_risk.md`
7. `docs/polymarket_panel/exp_05_validation_design.md`
8. `docs/polymarket_panel/final_panel_time_series_report.md`

---

## Development process and LLM use

The file

```text
final_deliverable/llm_exploration/STAT_4830_LLM_Logs.pdf
```

documents our use of LLMs throughout the project. LLMs helped with brainstorming, code scaffolding, debugging, data inspection, explanation, writing, and presentation preparation. However, the team remained responsible for choosing the research direction, validating assumptions, running experiments, checking code, interpreting outputs, and deciding which claims were supported by the evidence.

---

## Self-critique and limitations

The file

```text
final_deliverable/self_critiques/STAT_4830_Self_Critique_Edited.pdf
```

provides a detailed reflection on the strengths and limitations of the project.

The main limitations are:

- The equity optimizer relies heavily on rolling sample means and covariance estimates, which are noisy in financial data.
- Hyperparameter exploration was informative, but future work with more date-level observations could impose an even cleaner development, validation, and test split.
- Equal-weight and broad index benchmarks are useful, but future work should include additional internal baselines such as market-cap weighting, minimum variance, risk parity, momentum, and volatility-scaled strategies.
- The prediction-market extension is limited by the available data. The packaged artifact does not include fully resolved outcomes for all contracts, so realized PnL is measured using next-step quote changes rather than final settlement values.
- The prediction-market model is intentionally simple and should be viewed as a baseline research framework rather than a final trading system.
- The covariance-risk update improves the risk model structure, but rolling covariance estimates are still noisy and should not be interpreted as a definitive model of contract dependence.

---

## Future research directions

Several natural extensions follow from the project:

- Use shrinkage covariance estimators, factor-model covariance matrices, Bayesian expected-return estimates, and stronger regularization in the equity optimizer.
- Add more portfolio-rule baselines, including market-cap weighting, minimum variance, risk parity, momentum, and volatility-scaled strategies.
- Build a resolved prediction-market dataset with contract histories, order-book data, liquidity measures, event categories, final resolutions, and settlement dates.
- Evaluate prediction-market performance using settlement-based PnL rather than only next-step quote-to-quote changes.
- Add richer prediction-market features, including text embeddings, event categories, price histories, volume changes, order-book depth, news signals, and external fundamentals.
- Make the prediction-market optimizer more execution-aware by modeling bid-ask spreads, market depth, partial fills, slippage, fees, and order timing.
- Extend the prediction-market covariance model with shrinkage, event-level factor structure, and more robust handling of sparse histories.
- Modularize the code further into a more reusable research package.

---

## Project status

This repository contains a reproducible research pipeline for dynamic portfolio optimization in equities and prediction markets. The equity portion provides the original optimization base. The prediction-market portion extends the same allocation logic to a newer and less standardized market setting. The project should be read as an applied numerical optimization study. Its main contribution is to show how optimization objectives, constraints, numerical solvers, regularization, data limitations, covariance risk, and evaluation design interact when theory is turned into an empirical allocation system.

---

## Authors and course information

- Naseebullah Andar
- Samantha Agisim
- Patrick Ledoit

Course: STAT 4830: Numerical Optimization for Data Science and Machine Learning  
Instructor: Dr. Damek Davis
