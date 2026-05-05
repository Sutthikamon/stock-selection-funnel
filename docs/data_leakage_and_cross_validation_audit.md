# Data Leakage and Cross-Validation Audit

Audit date: 2026-05-05

Project files checked:

- `notebooks/01_prepare_sp500_data.ipynb`
- `notebooks/02_select_stocks_clustering_mst.ipynb`
- `notebooks/03_allocate_portfolios.ipynb`
- `notebooks/04_backtest_allocation_only.ipynb`
- `scripts/04_backtest_allocation_only.py`
- `notebooks/05_backtest_full_pipeline_walkforward.ipynb`
- `scripts/05_backtest_full_pipeline_walkforward.py`

## Executive Summary

The `notebooks/04_backtest_allocation_only.ipynb` / `scripts/04_backtest_allocation_only.py` files are valid as an allocation-only walk-forward test, because each rebalance computes portfolio weights using only returns available up to that rebalance date.

The `notebooks/05_backtest_full_pipeline_walkforward.ipynb` / `scripts/05_backtest_full_pipeline_walkforward.py` files reduce the major step-02 look-ahead problem by re-running clustering and stock selection at every rebalance date using only returns available up to that date.

The remaining major limitation is constituent bias: the investable universe still comes from the current S&P 500 list in the project data, not a point-in-time historical S&P 500 membership dataset. Results should be described as a current-constituent walk-forward simulation, not as a fully point-in-time historical S&P 500 backtest.

## Leakage Checklist

| Area | Current Status | Risk Level | Notes |
|---|---|---:|---|
| Current S&P 500 universe from Wikipedia | Uses current constituents | High for historical backtest | Creates survivorship/constituent bias if presented as a historical S&P 500 strategy |
| Step 02 stock selection | Uses full returns from `2019-01-03` to `2026-05-01` | High for full-pipeline backtest | Clustering and historical Sharpe ranking see future returns relative to the `2022-01-03` backtest start |
| Step 03 static allocation | Uses full available history | Not a backtest | Fine for "current allocation as of latest data", not for historical performance claims |
| Step 04 allocation weights | Uses `train_returns = returns_selected.loc[:rebalance_date]` | Low | No future returns are used when computing weights inside each rebalance |
| Step 04 realized performance | Uses only future holding returns after rebalance | Low | This is the correct out-of-sample measurement pattern |
| Step 05 full-pipeline selection | Re-selects stocks every rebalance using only past returns | Low for selection leakage | This fixes the fixed-25-stock look-ahead issue from Step 04 and records selected-stock audit metadata |
| Step 05 investable universe | Still uses current S&P 500 constituents | High for true historical S&P 500 claim | Requires point-in-time constituent data to fully fix |
| Markowitz-style delta selection in Step 04 | Picks max training Sharpe each rebalance | Medium | Not future leakage, but can overfit because hyperparameter selection and model fitting use the same train sample |
| CVaR tradeoff in Step 04 | Fixed at `1.0` | Low | No tuning leakage currently |
| Risk-free rate in Step 04 | Historical FRED `DGS3MO` aligned by date | Low/Medium | Good enough for research; for strict tradable tests, lag by one business day to account for publication timing |
| Dropping rows with selected-stock NaNs | Uses the fixed selected universe | Medium if delistings appear later | With current selected stocks all have full observations, but this can hide future availability issues in broader tests |

## What Is Currently Safe

`04_backtest_allocation_only` is safe for this question:

> Given the 25 selected stocks, which allocation method worked better out of sample?

Reason:

- First rebalance date: `2021-12-31`
- First holding date: `2022-01-03`
- Last holding date: `2026-05-01`
- Minimum training observations: `756`
- Training window: expanding
- Every rebalance uses only data up to the rebalance date
- Portfolio performance is measured with realized returns after the rebalance date

## What Is Not Yet Safe

The current backtest is not safe for this stronger claim:

> The full stock-selection plus allocation strategy would have produced these historical returns.

Reason:

- The 25 stocks were selected once using full history through `2026-05-01`.
- A true backtest starting on `2022-01-03` could not know the full 2022-2026 historical Sharpe rankings.
- The S&P 500 universe is the current Wikipedia list, not point-in-time historical constituents.

After adding Step 05, this stronger claim becomes closer but still needs careful wording:

> The full workflow worked on current S&P 500 constituents under walk-forward selection.

It still should not be described as a true historical S&P 500 constituent backtest unless point-in-time constituents are added.

## Recommended Cross-Validation Design

Do not use random k-fold cross-validation. Portfolio returns are time series, so random splits leak future regimes into the training data.

Use walk-forward validation instead.

### Level 1: Allocation-Only Walk-Forward

This is what Step 04 currently does.

```text
Fixed 25 selected stocks
For each month-end rebalance:
    train = returns up to rebalance date
    fit allocation weights
    hold next month
    record realized returns
```

Use this level to compare:

- Equal Weight
- Inverse Volatility
- Risk Parity
- Markowitz-style Mean-Volatility Optimization
- CVaR Bootstrap
- CVaR Monte Carlo

### Level 2: Nested Walk-Forward For Hyperparameters

Use this when tuning parameters such as:

- Markowitz-style `delta`
- CVaR `return_tradeoff`
- `MAX_WEIGHT`
- `N_CLUSTERS`
- minimum Sharpe observations

At each outer rebalance:

```text
outer_train = all data up to rebalance date

inside outer_train:
    split into smaller chronological train/validation folds
    test candidate hyperparameters on validation folds
    choose best hyperparameter by validation Sharpe/CAGR/drawdown rule

fit final weights on full outer_train using chosen hyperparameter
hold next month out of sample
```

This avoids selecting parameters using the same data used to score the final backtest.

### Level 3: Full-Pipeline Walk-Forward

Use this for the strongest historical claim.

At each rebalance:

```text
use only data available up to rebalance date
build/limit the investable universe
compute correlation
cluster stocks
select one stock per cluster by historical Sharpe ranking
allocate portfolio
hold next month
record realized returns
```

This removes the Step 02 selection leakage, but it is more expensive and ideally requires point-in-time S&P 500 membership data.

## Implemented Step 05

Added:

```text
notebooks/05_backtest_full_pipeline_walkforward.ipynb
scripts/05_backtest_full_pipeline_walkforward.py
```

Current Step 05 behavior:

- First rebalance date: `2021-12-31`
- First holding date: `2022-01-03`
- Last holding date: `2026-05-01`
- Rebalance count: `53`
- Selects 25 stocks at every rebalance
- Uses only returns up to each rebalance date for clustering, historical Sharpe ranking, and portfolio allocation
- Writes `train_start_date`, `train_end_date`, and `selection_mode = walk_forward_past_data_only` into `full_pipeline_selected_stocks_history.csv`
- Writes `full_pipeline_missing_holding_returns.csv` as a missing realized holding-period return audit file
- Uses current S&P 500 constituents from Step 01, not point-in-time constituents

Step 05 output interpretation:

- `full_pipeline_metrics.csv` answers how the dynamic full workflow performed.
- `full_pipeline_selected_stocks_history.csv` shows which stocks were selected at each rebalance and includes the training window and selection mode for auditability.
- `full_pipeline_missing_holding_returns.csv` records any missing realized holding-period returns; the latest run has zero rows.
- `full_pipeline_selection_frequency.csv` shows which stocks were repeatedly selected.
- `full_pipeline_selected_overlap.csv` shows selection stability over time.

## Recommended Next Fix

If higher historical rigor is required, add point-in-time S&P 500 membership data:

```text
historical_sp500_constituents_by_date
```

Purpose:

- At each rebalance, include only stocks that were actually S&P 500 constituents at that date.
- Avoid current-constituent survivorship/constituent bias.

If point-in-time S&P 500 constituent data is unavailable, label the result clearly as:

```text
current-constituent historical backtest
```

not:

```text
true historical S&P 500 constituent backtest
```
