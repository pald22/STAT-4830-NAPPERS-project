# Final Report: Panel-Aware Polymarket Time-Series Modeling

## Executive summary

This report consolidates results from five panel experiments that upgraded the prior snapshot-only approach into a leakage-safe time-series pipeline for daily Polymarket data.

Main findings:

- A panel model can produce directional edge after costs (`directional_accuracy_test = 0.536`, `net_sharpe_test = 0.770`) in the current setup.
- Liquidity structure matters: average bucket-level rank correlation is positive (`mean_bucket_rank_ic = 0.091`), indicating forecastability varies by tradability regime.
- Cross-sectional features add incremental explanatory value (`delta_r2_full_minus_base = 0.0080`).
- Unresolved-risk handling improves reliability (`mae_adjusted = 0.0724`, `interval_coverage_1sigma = 0.947`).
- Validation design is a major risk: validation-to-test correlation is negative (`val_test_corr = -0.609`), implying model-selection instability and sensitivity to splitter choice.

All experiments passed documented split and sanity checks after implementation updates.

---

## 1) Problem statement and scope

We model one row per market-day from `polymarket_daily_panel_60plus.csv` to forecast short-horizon return behavior and evaluate cost-aware portfolio construction under sparse terminal labels.

Goals:

1. Move from static snapshot modeling to panel-aware temporal modeling.
2. Preserve strict chronology and leakage safety.
3. Test whether predictive structure survives execution costs.
4. Understand liquidity, cross-sectional factors, unresolved-market uncertainty, and validation design trade-offs.

---

## 2) Data and pipeline overview

Dataset fields used include `market_id`, `question`, `date`, `price`, `volume`, `liquidity`, `active`, `closed`, `end_date_iso`, and category metadata.

Core engineered features:

- **Market lag/time-series**: `price_lag_1`, `price_lag_7`, lagged returns, rolling momentum (`mom_3d`, `mom_7d`), and rolling volatility (`vol_7d`, `vol_21d`).
- **Cross-sectional**: within-date percentile ranks and median-relative mispricing.
- **Microstructure proxies**: log volume, log liquidity, and illiquidity proxy (`|ret_1d| / volume`).

Targets:

- `target_1d`: next-day market return (grouped by market).
- `target_7d`: next-7-day return (reserved for extension; core runs focused on `target_1d`).

---

## 3) Validation protocol and leakage controls

Default split inside each experiment:

- Train: earliest 60% of dates
- Validation: next 20%
- Test: final 20%

Critical rules:

- No overlap within an experiment across train/validation/test.
- Hyperparameter selection uses validation only.
- Test is untouched until final scoring.
- Features only use information available at or before decision time.

Sanity checks were programmatically enforced and logged in per-experiment reports.

---

## 4) Key terms (glossary)

- **Panel data**: repeated observations over time for each entity (here, each `market_id`).
- **Leakage**: unintended use of future information in training or feature construction.
- **Rank IC (Spearman)**: rank correlation between predictions and realized outcomes; useful for cross-sectional signal quality.
- **Brier score**: mean squared error of probabilistic directional forecasts; lower is better.
- **Sharpe ratio**: mean return divided by return volatility (annualized here).
- **Turnover**: amount of position change across rebalances; proxy for trading intensity/cost burden.
- **Cross-sectional factor**: feature describing relative behavior vs peers (e.g., within-date ranks).
- **Unresolved-market risk**: uncertainty from open markets without terminal outcomes in-sample.
- **Validation-to-test correlation**: how well validation ranking predicts true out-of-sample ranking; high is good.

---

## 5) Experiment-by-experiment results

## Experiment 1: Persistent predictability net of costs

Question: Do we observe persistent, tradable short-horizon predictability after cost assumptions?

Results:

- `r2_test = -0.0187`
- `mae_test = 0.0944`
- `directional_accuracy_test = 0.5362`
- `rank_ic_test = 0.0769`
- `net_sharpe_test = 0.7695`
- `turnover_test = 0.2615`
- Cost stress (+50%) sanity check remained positive.

Interpretation:

- Point-error fit is weak (`R^2 < 0`), but ranking/direction and cost-adjusted portfolio outcomes are positive.
- This is consistent with weak absolute prediction but usable relative ordering.

Artifacts:

- Figure: `outputs/polymarket_panel_experiments/exp_01/fig_exp01_cum_pnl.png`
- Table: `outputs/polymarket_panel_experiments/exp_01/table_exp01_decile_returns.csv`
- Report: `outputs/polymarket_panel_experiments/reports/exp_01_report.md`

## Experiment 2: Liquidity and forecastability

Question: Does liquidity dampen or amplify forecastability?

Results:

- `mean_bucket_rank_ic = 0.0915` across train-defined liquidity buckets.

Interpretation:

- Signal quality differs by liquidity regime; pooled modeling can hide this heterogeneity.
- Supports liquidity-aware weighting and/or regime interactions.

Artifacts:

- Figure: `outputs/polymarket_panel_experiments/exp_02/fig_exp02_rankic_bucket.png`
- Table: `outputs/polymarket_panel_experiments/exp_02/table_exp02_bucket_metrics.csv`
- Report: `outputs/polymarket_panel_experiments/reports/exp_02_report.md`

## Experiment 3: Stable cross-sectional factors

Question: Do cross-sectional factors add stable incremental value?

Results:

- `delta_r2_full_minus_base = 0.0080`

Interpretation:

- Adding cross-sectional features yields a measurable improvement over market-level-only baseline.
- Improvement is modest but directionally consistent with a multi-sleeve signal stack.

Artifacts:

- Table: `outputs/polymarket_panel_experiments/exp_03/table_exp03_incremental.csv`
- Report: `outputs/polymarket_panel_experiments/reports/exp_03_report.md`

## Experiment 4: Unresolved open-market risk

Question: How do we model risk when terminal outcomes are sparse?

Results:

- `brier_directional = 0.2584`
- `interval_coverage_1sigma = 0.9468`
- `mae_adjusted = 0.0724`

Interpretation:

- Uncertainty-adjusted predictions are better calibrated and relatively robust.
- Coverage is high, which may indicate conservative uncertainty estimates; calibration refinement is a next step.

Artifacts:

- Table: `outputs/polymarket_panel_experiments/exp_04/table_exp04_reliability.csv`
- Report: `outputs/polymarket_panel_experiments/reports/exp_04_report.md`

## Experiment 5: Validation design stress test

Question: Which validation design best reflects production deployment?

Results:

- `val_test_corr = -0.6091`

Interpretation:

- Splitter choice materially changes apparent model quality.
- Negative validation-test relationship is a warning sign of non-stationarity and selection risk.

Artifacts:

- Figure: `outputs/polymarket_panel_experiments/exp_05/fig_exp05_val_vs_test.png`
- Table: `outputs/polymarket_panel_experiments/exp_05/table_exp05_splitter_compare.csv`
- Report: `outputs/polymarket_panel_experiments/reports/exp_05_report.md`

---

## 6) Cross-experiment synthesis

What is working:

- Panel features + chronology-safe splits produce reproducible, cost-aware signal evidence.
- Liquidity and cross-sectional context improve signal interpretation and portfolio construction.
- Unresolved-risk discounting helps stabilize reliability under sparse terminal labels.

What is fragile:

- Absolute return prediction fit remains weak.
- Model-selection robustness across splitters is poor (largest risk surfaced by Experiment 5).

Production implications:

- Use ranked-signal framing (cross-sectional) rather than relying on absolute return point forecasts.
- Integrate liquidity-aware sizing and turnover control directly into optimizer inputs.
- Adopt robust multi-window validation as default governance, not a single split.

---

## 7) Recommended default deployment spec (v1)

- **Model layer**: Regularized linear baseline + tree ensemble challenger.
- **Signal format**: Predicted return + uncertainty band + rank percentile.
- **Portfolio layer**: Daily rebalance with liquidity and turnover penalties.
- **Risk layer**: Unresolved-market haircut linked to horizon and uncertainty.
- **Validation layer**: Expanding + rolling stress tests; require consistency before promotion.
- **Monitoring**: rank IC drift, calibration drift, turnover/cost drift, and selection-stability metrics.

---

## 8) Limitations

- Sparse true terminal labels limit direct terminal-outcome learning.
- Cost model is stylized and should be improved with richer execution assumptions.
- Current results are one full run snapshot; confidence intervals/bootstraps should be added for final claims.

---

## 9) Next actions

1. Add folded walk-forward evaluation with confidence bands around core metrics.
2. Add explicit optimizer decomposition (signal alpha vs cost drag vs allocation effects).
3. Expand unresolved-risk calibration (quantile calibration and regime-conditioned uncertainty).
4. Promote only models that pass both performance and validation-stability gates.

---

## 10) Report map (complete presentation assets)

- Master run status: `outputs/polymarket_panel_experiments/master_report.md`
- Experiment reports:
  - `outputs/polymarket_panel_experiments/reports/exp_01_report.md`
  - `outputs/polymarket_panel_experiments/reports/exp_02_report.md`
  - `outputs/polymarket_panel_experiments/reports/exp_03_report.md`
  - `outputs/polymarket_panel_experiments/reports/exp_04_report.md`
  - `outputs/polymarket_panel_experiments/reports/exp_05_report.md`
- This final synthesis:
  - `docs/polymarket_panel/final_panel_time_series_report.md`
