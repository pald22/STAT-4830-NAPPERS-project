# Experiment 2: Liquidity and Forecastability

## Research question

Does liquidity dampen or amplify short-horizon forecastability and realized tradable alpha?

## Split design (non-overlap within experiment)

- Train: earliest 60% of dates.
- Validation: next 20%.
- Test: final 20%.
- Build liquidity buckets from train distribution; apply bucket cutoffs unchanged to validation/test.

## Method

- Primary setup:
  - Train common model on full universe.
  - Evaluate by liquidity quintile and interaction effects.
- Secondary setup:
  - Train stratified models by liquidity regime (low/med/high).
  - Compare against pooled model with liquidity interaction terms.
- Features:
  - Baseline panel features plus liquidity proxies and interaction terms:
  - `signal * liquidity_rank`, `momentum * illiquidity`, `volatility * liquidity`.

## Evaluation metrics

- Predictive:
  - Rank IC by liquidity bucket.
  - Bucket-level `R^2` and directional accuracy.
- Trading:
  - Net Sharpe by bucket-constrained sub-portfolios.
  - Average implementation shortfall by bucket.
- Stability:
  - Dispersion of metrics across folds and regimes.

## Sanity checks checklist

- [ ] Liquidity bucket definitions are fixed from train only.
- [ ] Bucket sample sizes are adequate and balanced over time.
- [ ] Effect direction is robust to alternative liquidity proxy definitions.
- [ ] No bucket is dominated by a single event/category family.
- [ ] Results hold after excluding extreme volume outliers.

## Report artifacts (markdown-ready)

## Key figures

1. Rank IC vs liquidity bucket (bar + confidence interval).
2. Net Sharpe vs liquidity bucket.
3. Shortfall curve as function of liquidity percentile.

## Key tables

1. Interaction coefficient table for liquidity terms.
2. Pooled vs stratified model comparison.
3. Stress test: widened spread assumptions by bucket.

## Summary doc template

- Hypothesis on liquidity-forecastability relationship.
- Which liquidity regime carries most signal after costs.
- Whether interactions materially improve out-of-sample fit.
- Deployment implication: liquidity-aware sizing or universe filtering.
