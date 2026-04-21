# Experiment 1: Persistent Predictability Net of Costs

## Research question

Which markets show persistent short-horizon predictability after transaction costs and liquidity-aware execution assumptions?

## Split design (non-overlap within experiment)

- Train: earliest 60% of dates.
- Validation: next 20%.
- Test: final 20%.
- Optional walk-forward repeats with same no-overlap rule per fold.

## Method

- Target:
  - `ret_1d` and `ret_7d` (log return or simple return, fixed across all runs).
- Models:
  - Baseline ridge regression and logistic direction model.
  - Upgrade candidate: LightGBM regression with quantile outputs.
- Portfolio mapping:
  - Convert forecast to position via volatility-scaled score.
  - Apply turnover and liquidity penalties in daily optimizer.
- Cost model:
  - Effective spread + slippage proxy from liquidity/volume bucket.
  - Include fixed bps floor for low-liquidity names.

## Evaluation metrics

- Predictive quality:
  - Out-of-sample `R^2`, MAE, directional accuracy, rank IC (Spearman).
- Trading quality:
  - Net Sharpe, net information ratio, hit rate, max drawdown.
  - Turnover, average holding period, capacity utilization.
- Cost awareness:
  - Gross vs net return spread.
  - Cost as percentage of gross alpha.

## Sanity checks checklist

- [ ] No train/validation/test date overlap.
- [ ] No leakage from future lags (all features use `<= t` only).
- [ ] Net performance remains positive under +50% cost stress.
- [ ] Results are not driven by top 1% liquidity names only.
- [ ] Sign-flip placebo test collapses performance toward zero.

## Report artifacts (markdown-ready)

## Key figures

1. Cumulative gross vs net PnL on test.
2. Rolling 30-day net Sharpe and turnover.
3. Decile portfolio spread (top vs bottom signal bucket).

## Key tables

1. Predictive metrics by horizon (`1d`, `7d`) and model.
2. PnL decomposition (signal, costs, residual).
3. Performance by liquidity bucket and maturity bucket.

## Summary doc template

- Objective and hypothesis.
- Data and split windows.
- Best model and selected hyperparameters (chosen on validation only).
- Test metrics with confidence intervals or bootstrap bands.
- Failure modes and next iteration actions.
