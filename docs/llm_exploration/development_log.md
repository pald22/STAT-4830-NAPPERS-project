# Development Log – Week 3 (Portfolio Maximization, NAPPERS)

## Overview

Goal: build a minimal, end-to-end portfolio optimization pipeline that
constructs monthly portfolios on the `sp500_monthly` panel and compares their
out-of-sample performance to simple benchmarks (equal-weight, market-like).

The focus this week was on:
- getting a **clean data pipeline** from CSV → return panel,
- implementing a **mean–variance + turnover** optimizer in PyTorch,
- wiring up a **walk-forward backtest**,
- adding **basic unit tests** and documenting results.

---

## Timeline

### Day 1 – Problem framing and dataset

- Chose the project focus: **portfolio optimization** that tries to outperform
  standard benchmarks out of sample, rather than pure return prediction.
- Confirmed the starting dataset: `sp500_monthly` (CSV) with columns such as
  `date`, `permno`, `ret`, `prc`, `shrout`, `vol`, `shrcd`, and `exchcd`.
- Decided to begin with a **fixed universe** of S\&P-like stocks instead of
  allowing the asset set to change every month (simplifies dimension handling).
- With the help of an LLM, drafted the Week 3 **problem statement** and a
  high-level **technical approach** emphasizing:
  - mean–variance optimization with risk-aversion parameter $\gamma$,
  - long-only, fully-invested portfolios,
  - a turnover penalty to proxy transaction costs.

### Day 2 – Data processing and panel construction

- Loaded `sp500_monthly (1).csv` into a Colab notebook and inspected the
  columns and basic statistics.
- Applied standard CRSP-style filters:
  - keep `shrcd` in {10, 11} and `exchcd` in {1, 2, 3},
  - drop observations with missing `ret`.
- Built a **date × permno return panel** using `pivot(index='date', columns='permno', values='ret')`.
- Selected a **small, stable universe** by choosing the permnos with the
  longest history (e.g., top 30 by number of months).
- Dropped dates with missing returns within this universe to simplify the
  initial implementation.

### Day 3 – Optimization model and PyTorch implementation

- Implemented `estimate_moments(window_returns)` in `src/model.py` to compute
  rolling estimates of $\mu_t$ and $\Sigma_t$:
  - used sample mean and covariance,
  - enforced symmetry and added a small diagonal jitter for numerical stability.
- Implemented `optimize_weights(mu_t, Sigma_t, w_prev)` in `src/model.py`:
  - parameterized weights via logits `v_t` with `softmax(v_t)` to enforce
    long-only and full investment,
  - defined objective  
    $\mu_t^\top w_t - \gamma w_t^\top \Sigma_t w_t - \kappa \lVert w_t - w_{t-1}\rVert_1$,
  - optimized logits with SGD in PyTorch for a fixed number of steps.
- Built a simple **walk-forward backtest** in `notebooks/week3_implementation.ipynb`:
  - used a 24-month rolling window to estimate $(\mu_t,\Sigma_t)$,
  - at each date, solved for $w_t$ and held the portfolio for one month,
  - recorded portfolio and equal-weight benchmark returns.

### Day 4 – Metrics, diagnostics, and interpretation

- Computed performance metrics for the optimized portfolio and equal-weight
  benchmark:
  - mean monthly return,
  - annualized volatility,
  - annualized Sharpe ratio,
  - cumulative return,
  - maximum drawdown.
- Observed that the optimized portfolio and equal-weight benchmark had **very
  similar performance** (Sharpe $\approx 0.85$, cumulative return $\approx 17\times$).
- With LLM assistance, computed:
  - **average $\ell_1$ distance** from equal-weight: ~0.007,
  - **average turnover per rebalance**: ~0.00023.
- Interpreted these diagnostics as evidence that, under the current settings,
  the optimizer is effectively choosing an **almost equal-weight, almost buy-and-hold**
  strategy:
  - rolling-mean $\mu_t$ has weak cross-sectional signal,
  - risk and turnover penalties keep weights close to diversified, stable
    allocations.

### Day 5 – Tests and documentation

- Added `tests/test_basic.py` with basic unit tests:
  - `test_estimate_moments_shapes_and_symmetry`: checks shapes and symmetry of
    $\Sigma$ on a toy dataset.
  - `test_optimize_weights_simplex`: verifies that `optimize_weights` returns
    nonnegative weights summing to 1.
  - `test_turnover_penalty_influence`: confirms that increasing $\kappa$ reduces
    average turnover in a small synthetic example.
- Ran `pytest tests/test_basic.py` to confirm all tests pass.
- Completed the Week 3 **report draft**:
  - Problem Statement, Technical Approach, Initial Results, and Next Steps,
  - included quantitative metrics and the weight/turnover diagnostics.
- Logged how LLMs were used for problem framing, code scaffolding, and result
  interpretation in `docs/llm_exploration/week3_log.md`.

---

## Key Design Decisions

- **Fixed universe** of top-$K$ permnos by history length to avoid changing
  dimensionality and missing data headaches in Week 3.
- **Mean–variance + turnover objective** as a simple, interpretable starting
  point that directly matches the course’s optimization focus.
- **Softmax parameterization** of weights instead of explicit constraints,
  which keeps the optimization unconstrained in PyTorch and guarantees
  feasibility.
- **Rolling-window estimation** of $(\mu_t,\Sigma_t)$ for simplicity, with the
  understanding that more sophisticated estimators can be slotted in later.

---

## Issues Encountered / Lessons Learned

- The naive rolling-mean estimator for $\mu_t$ yields very weak cross-sectional
  signals, so the optimizer naturally hugs equal-weight when combined with a
  turnover penalty and diversification-oriented risk term.
- Hyperparameters (risk aversion $\gamma$, turnover penalty $\kappa$, learning
  rate, number of steps) strongly influence how far the solution moves from the
  equal-weight portfolio, but aggressive choices can easily lead to instability.
- Getting **time ordering** right (no look-ahead) is critical; using only past
  returns in the rolling window and reserving the next month solely for
  evaluation is an important design constraint.

---

## Planned Next Steps

- Experiment with **hyperparameter sweeps** over $\gamma$ and $\kappa$, including
  the case $\kappa = 0$, to map out when the optimizer meaningfully deviates
  from equal-weight and how that affects Sharpe, drawdowns, and turnover.
- Introduce simple **momentum-based or factor-based signals** for $\mu_t$ to
  give the optimizer stronger cross-sectional information.
- Apply **shrinkage** or other regularization to $\Sigma_t$ to reduce
  estimation error in the covariance.
- Extend tests to cover more edge cases (e.g., degenerate covariance matrices,
  extremely volatile assets) and to ensure that modifications still respect
  time ordering and feasibility constraints.
