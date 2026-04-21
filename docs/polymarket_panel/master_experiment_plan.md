# Polymarket Panel Time-Series Master Plan

This document upgrades the snapshot-only architecture into a panel-aware daily modeling and portfolio system. It also defines split policies and links each research question to a standalone experiment report.

## 1) Panel-aware architecture (implementation blueprint)

## Folder structure changes

```text
src/polymarket/
  data/
    panel_loader.py
    schema.py
  features/
    panel_store.py
    lag_features.py
    cross_sectional.py
    microstructure.py
  models/
    targets.py
    baseline_linear.py
    panel_regularized.py
    tree_models.py
    sequence_models.py
    calibration.py
  portfolio/
    forecast_distribution.py
    optimizer_daily.py
    constraints.py
    transaction_costs.py
  evaluation/
    splitters.py
    metrics.py
    diagnostics.py
    reporting.py
```

## Module responsibilities and interface contracts

- `panel_loader.py`
  - Input: path to `polymarket_daily_panel_60plus.csv`.
  - Output: canonical daily panel with typed columns and no duplicate (`market_id`, `date`) keys.
  - Contract: `load_panel(cfg: DataConfig) -> pd.DataFrame`.
- `panel_store.py`
  - Input: panel DataFrame.
  - Output: feature matrix keyed by (`market_id`, `date`) and target table.
  - Contract: `build_feature_store(df: pd.DataFrame, cfg: FeatureConfig) -> FeatureStore`.
- `targets.py`
  - Input: panel with prices and metadata.
  - Output: aligned targets such as `ret_1d`, `ret_7d`, and volatility-adjusted variants.
  - Contract: `make_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame`.
- `splitters.py`
  - Input: panel date index and split config.
  - Output: train/validation/test index masks per fold with leakage checks.
  - Contract: `generate_splits(df: pd.DataFrame, cfg: SplitConfig) -> list[SplitSpec]`.
- `optimizer_daily.py`
  - Input: daily forecast distribution, covariance proxy, costs, constraints.
  - Output: next-day weights and rebalance trades.
  - Contract: `rebalance_day(state: PortfolioState, signal: ForecastPack, cfg: PortfolioConfig) -> TradeDecision`.
- `reporting.py`
  - Input: predictions, realized returns, trades, diagnostics.
  - Output: markdown-ready tables and figure specs.
  - Contract: `build_experiment_report(results: ExperimentResult, cfg: ReportConfig) -> ReportBundle`.

## 2) Leakage-safe split design (global policy)

Use date-based rolling/expanding windows. Keep chronology strict. Group by market for lag construction and sequence extraction.

- Within each experiment:
  - `train`: oldest block.
  - `validation`: next block (used for hyperparameter selection and threshold tuning only).
  - `test`: final forward block.
  - No overlap between train/validation/test dates.
- Across experiments:
  - Overlap is allowed, but each experiment must still preserve non-overlap internally.
- Integrity checks required before training:
  - No record where feature timestamp is after target timestamp.
  - No shared (`market_id`, `date`) rows across split partitions.
  - No forward-fill across split boundaries.
  - Category/bucket rank features are computed by date only from that day’s observable universe.

Recommended initial split calendar (adjust if data span differs):

- Train: first 60% of unique dates.
- Validation: next 20%.
- Test: final 20%.

Walk-forward extension:

- Fold `k`: expand train through date `T_k`, validate on `T_k+1..T_k+V`, test on `T_k+V+1..T_k+V+H`.

## 3) Core feature families (global)

- Market lag/time-series:
  - `price_lag_1`, `price_lag_7`, `ret_1d_lag_1`, `ret_7d_lag_1`.
  - Rolling realized volatility: `vol_7d`, `vol_21d`.
  - Momentum/reversal: cumulative return over 3/7/14 days and sign reversal flags.
- Cross-sectional:
  - Within-date z-score and percentile rank of return, spread proxy, and liquidity.
  - Peer-relative mispricing by event/category/liquidity bucket.
  - Horizon-to-expiry percentile and maturity bucket interactions.
- Microstructure proxies:
  - `dollar_volume`, `turnover_ratio`, liquidity depth proxy, spread proxy.
  - Price impact proxy: `|ret_1d| / max(volume, eps)`.
  - Illiquidity interaction terms with momentum and mean-reversion.

## 4) Model stack progression (global)

1. Baseline linear models:
   - Ridge/Lasso/Elastic Net for `ret_1d` and `ret_7d`.
   - Logistic model for direction-of-return.
2. Panel regularized models:
   - Group-regularized linear model (group by category/event).
   - Mixed-effects approximation or entity-fixed-effects controls.
3. Tree models:
   - LightGBM/XGBoost with monotonic and depth constraints.
   - Quantile objectives for forecast distributions.
4. Sequence models (only if justified by out-of-sample gains net of costs):
   - Temporal CNN/LSTM/Transformer over per-market windows.
   - Strict masking to avoid future leakage.

Promotion rule: only advance to next class if validation and test both show incremental, cost-adjusted gains and stable calibration.

## 5) Portfolio/optimizer upgrade (global)

Daily rebalance loop uses forecast distributions, not point forecasts only.

- Inputs:
  - Mean forecast `mu_i,t`, uncertainty `sigma_i,t`, and confidence interval/quantiles.
  - Liquidity capacity, estimated costs, and turnover state from previous weights.
- Objective:
  - Max expected risk-adjusted return:
  - `max_w  w' * mu - lambda_risk * w' * Sigma * w - lambda_tc * TC(w,w_prev) - lambda_to * |w-w_prev|`.
- Constraints:
  - Gross and net exposure caps.
  - Per-market and per-event concentration.
  - Liquidity utilization cap (position relative to expected tradable volume).
  - Optional unresolved-market risk haircut (downweight long-dated unresolved contracts).

## 6) Monitoring and diagnostics (global)

- Drift:
  - Population Stability Index on top predictors and target moments.
- Calibration:
  - Reliability curves by forecast bin; Brier score for directional models.
- Factor exposure:
  - Exposure to liquidity, maturity, and category factors.
- Regime sensitivity:
  - Slice metrics by liquidity regime, event density, and high-volatility windows.
- PnL decomposition:
  - Signal alpha, cost drag, allocation effect, interaction effect.

## 7) Staged migration plan

1. Data + split foundation
   - Build canonical panel loader and splitter with leakage tests.
2. Feature store
   - Add lag, cross-sectional, and microstructure features with registry pattern.
3. Baseline panel models
   - Reproduce baseline with new targets/splits and establish benchmark metrics.
4. Optimizer upgrade
   - Replace point-estimate objective with distribution-aware objective.
5. Diagnostics/reporting
   - Emit per-fold and aggregate markdown artifacts.
6. Advanced models
   - Add trees, then sequence only if justified.

## 8) Sub-experiment index

- [Experiment 1: Persistent Predictability Net of Costs](./exp_01_persistent_predictability.md)
- [Experiment 2: Liquidity and Forecastability](./exp_02_liquidity_forecastability.md)
- [Experiment 3: Stable Cross-Sectional Factors](./exp_03_cross_sectional_factors.md)
- [Experiment 4: Unresolved Open-Market Risk Modeling](./exp_04_unresolved_market_risk.md)
- [Experiment 5: Validation Design Stress Test](./exp_05_validation_design.md)
