# Experiment 3: Stable Cross-Sectional Factors

## Research question

Are there stable cross-sectional factors (event type/category, maturity horizon, crowding) that improve edge estimates over pure market-level time-series signals?

## Split design (non-overlap within experiment)

- Train: earliest 60% of dates.
- Validation: next 20%.
- Test: final 20%.
- Factor definitions and clustering learned on train only.

## Method

- Factor construction:
  - Category/event indicators from metadata.
  - Maturity horizon buckets from time-to-expiry.
  - Crowding proxy from volume concentration and question similarity density.
- Model comparison:
  - Base model: market lags + microstructure only.
  - Factor model: base + cross-sectional factors.
  - Hierarchical/regularized panel model with group penalties.
- Stability analysis:
  - Coefficient sign consistency across folds.
  - Factor return persistence using rolling estimates.

## Evaluation metrics

- Incremental prediction:
  - Delta `R^2`, delta rank IC, likelihood improvement.
- Economic value:
  - Incremental net Sharpe from adding factor sleeve.
  - Information ratio of factor-mimicking portfolios.
- Stability:
  - Sign consistency ratio and coefficient drift statistics.

## Sanity checks checklist

- [ ] Factor definitions do not use test-period information.
- [ ] Category labels are cleaned and stationary enough for grouping.
- [ ] Factor gains are not solely from one short date interval.
- [ ] Multicollinearity diagnostics are acceptable (VIF or condition checks).
- [ ] Removing top crowded events does not fully erase factor signal.

## Report artifacts (markdown-ready)

## Key figures

1. Rolling factor premia by factor family.
2. Cumulative return of factor-mimicking long-short sleeves.
3. Coefficient stability heatmap across folds.

## Key tables

1. Incremental model comparison (base vs factor-augmented).
2. Top factors by t-stat and economic contribution.
3. Factor exposure of final portfolio over test period.

## Summary doc template

- Which factors are stable versus transient.
- Quantified incremental edge from factors.
- Recommended factors for production signal stack.
