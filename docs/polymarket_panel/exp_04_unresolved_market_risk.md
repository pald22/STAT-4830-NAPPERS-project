# Experiment 4: Unresolved Open-Market Risk Modeling

## Research question

How should unresolved/open-market risk be modeled when terminal outcome labels are sparse or unavailable?

## Split design (non-overlap within experiment)

- Train: earliest 60% of dates.
- Validation: next 20%.
- Test: final 20%.
- Any pseudo-labeling or proxy target generation must be fit on train only and frozen before validation/test scoring.

## Method

- Label strategy variants:
  1. Horizon-return proxy labels (`ret_1d`, `ret_7d`) only.
  2. Partial-resolution subset (markets that resolve during sample) with survival-style censor handling.
  3. Bayesian shrinkage toward market-implied probability for unresolved contracts.
- Risk haircut layer:
  - Penalize forecasts by uncertainty and days-to-expiry.
  - Apply unresolved-risk discount in optimizer expected return input.
- Compare:
  - No-risk-haircut baseline vs uncertainty-aware and censor-aware variants.

## Evaluation metrics

- Forecast reliability:
  - Calibration slope/intercept, Brier score for directional bins.
  - Prediction interval coverage (for quantile models).
- Portfolio robustness:
  - Net Sharpe and drawdown under unresolved-risk haircut scenarios.
  - Tail risk metrics (CVaR, worst 5% daily outcomes).
- Sensitivity:
  - Performance vs unresolved exposure share.

## Sanity checks checklist

- [ ] Proxy labels are generated without peeking into future outcomes.
- [ ] Resolved-only subset analysis is clearly separated from full-universe runs.
- [ ] Haircut severity tuned on validation only.
- [ ] Uncertainty estimates are calibrated, not just scaled scores.
- [ ] Portfolio does not become unintentionally concentrated in near-expiry names.

## Report artifacts (markdown-ready)

## Key figures

1. Calibration curves for each unresolved-risk method.
2. Net PnL and drawdown under haircut stress ladder.
3. Risk-return frontier with unresolved exposure on x-axis.

## Key tables

1. Method comparison across reliability and portfolio metrics.
2. Exposure decomposition by resolution status and maturity.
3. Tail-risk summary by method.

## Summary doc template

- Chosen unresolved-risk framework and rationale.
- Trade-off between calibration and portfolio performance.
- Recommended default haircut policy for production backtests.
