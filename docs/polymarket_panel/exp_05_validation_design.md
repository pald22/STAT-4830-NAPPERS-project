# Experiment 5: Validation Design Stress Test

## Research question

Which validation design best matches production deployment for daily rebalance on partially overlapping market lifecycles?

## Split design candidates (all non-overlap within candidate)

Candidate A (single holdout):

- Train 60% / Validation 20% / Test 20% by date.

Candidate B (expanding walk-forward):

- Multiple folds with expanding train, fixed validation horizon, fixed test horizon.

Candidate C (rolling window):

- Fixed-length train/validation/test windows shifted forward in time.

Rules:

- No date overlap among train/validation/test inside each fold.
- Same market may appear across folds over time, but never with overlapping dates inside a fold.

## Method

- Keep model class fixed (e.g., ridge + LightGBM).
- Run identical feature sets and optimizer settings across candidate splitters.
- Compare selected hyperparameters, model ranking stability, and test degradation.
- Include deployment simulation: choose model on validation, execute once on held-out test period.

## Evaluation metrics

- Selection quality:
  - Correlation between validation score and future test score.
  - Frequency of validation winner being test winner.
- Stability:
  - Variance of chosen hyperparameters across folds.
  - Drift in feature importance/ranking.
- Deployment realism:
  - Gap between in-validation and post-selection test performance.
  - Turnover and capacity stability under selected model.

## Sanity checks checklist

- [ ] Split boundaries respect chronology and no-overlap.
- [ ] Hyperparameter tuning uses validation only; test is untouched.
- [ ] Data transforms/scalers are fit per train block only.
- [ ] Cross-sectional ranks are computed independently for each date.
- [ ] Selection-performance gap is reported with uncertainty bands.

## Report artifacts (markdown-ready)

## Key figures

1. Validation vs test score scatter by splitter candidate.
2. Performance decay chart from validation to test.
3. Fold-wise turnover and drawdown distributions.

## Key tables

1. Candidate splitter comparison matrix.
2. Hyperparameter stability summary.
3. Leakage audit checklist results.

## Summary doc template

- Recommended default validation design.
- Why it best approximates production behavior.
- Residual risks and safeguards for future retraining cycles.
