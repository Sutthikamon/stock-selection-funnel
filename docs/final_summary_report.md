# Final Project Summary

## Executive Summary

This project builds a stock-selection and portfolio-allocation workflow for S&P 500 stocks. The most reliable result is Step 05, because it re-selects stocks and re-allocates the portfolio at each rebalance date using only past data.

The results should be interpreted as a current-constituent walk-forward simulation, not as a fully point-in-time historical S&P 500 backtest.

- Best full-pipeline CAGR: **Inverse Volatility** at **12.70%**.
- Best full-pipeline Sharpe: **CVaR Bootstrap** at **0.6357**.
- Shallowest full-pipeline max drawdown: **CVaR Bootstrap** at **-20.64%**.

Important limitation: Step 05 fixes the major look-ahead issue from using one fixed selected-stock list, but the universe still uses current S&P 500 constituents rather than point-in-time historical constituents.

## Data Summary

| Item                                 | Value                    |
| ------------------------------------ | ------------------------ |
| S&P 500 universe rows                | 503                      |
| Usable return tickers                | 502                      |
| Return observations                  | 1,842                    |
| Return date range                    | 2019-01-03 to 2026-05-01 |
| Benchmark                            | ^GSPC                    |
| Static selected stocks from Step 02  | 25                       |
| Full-pipeline rebalance count        | 53                       |
| Full-pipeline first holding date     | 2022-01-03               |
| Full-pipeline last holding date      | 2026-05-01               |
| Full-pipeline unique selected stocks | 114                      |
| Average monthly selection overlap    | 0.6075                   |
| Missing holding-return audit rows    | 0                        |

## Workflow Summary

| Step | File                                                  | Purpose                                                                                                                                                                      |
| ---- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 01   | notebooks/01_prepare_sp500_data.ipynb                 | Prepare S&P 500 universe, prices, returns, benchmark, and quality reports                                                                                                    |
| 02   | notebooks/02_select_stocks_clustering_mst.ipynb       | Cluster stocks and select one stock per cluster using historical Sharpe as a backward-looking ranking heuristic                                                              |
| 03   | notebooks/03_allocate_portfolios.ipynb                | Create current portfolio allocations using Equal Weight, Inverse Volatility, Markowitz-style Mean-Volatility Optimization, Risk Parity, CVaR Bootstrap, and CVaR Monte Carlo |
| 04   | notebooks/04_backtest_allocation_only.ipynb           | Allocation-only walk-forward test using the fixed Step 02 selected-stock list                                                                                                |
| 05   | notebooks/05_backtest_full_pipeline_walkforward.ipynb | Main full-pipeline walk-forward simulation with stock selection and allocation rerun each rebalance                                                                          |
| 06   | notebooks/06_final_summary_report.ipynb               | Final project summary and comparison report                                                                                                                                  |

## Step 05 Full-Pipeline Results

| Method                                       | Final Value | CAGR   | Sharpe | Max Drawdown | Volatility | Turnover |
| -------------------------------------------- | ----------- | ------ | ------ | ------------ | ---------- | -------- |
| Equal Weight                                 | 1.6381      | 12.09% | 0.5343 | -23.32%      | 16.34%     | 28.95    |
| Inverse Volatility                           | 1.6769      | 12.70% | 0.6144 | -21.78%      | 14.68%     | 29.46    |
| Risk Parity                                  | 1.6337      | 12.02% | 0.5772 | -21.68%      | 14.54%     | 27.34    |
| Markowitz-style Mean-Volatility Optimization | 1.5341      | 10.41% | 0.4013 | -30.50%      | 19.25%     | 30.61    |
| CVaR Bootstrap                               | 1.6721      | 12.63% | 0.6357 | -20.64%      | 13.91%     | 34.89    |
| CVaR Monte Carlo                             | 1.5025      | 9.87%  | 0.4644 | -23.67%      | 13.60%     | 32.20    |
| S&P 500                                      | 1.5231      | 10.22% | 0.4127 | -25.43%      | 17.55%     | 0.00     |

## Step 05 Audit Trail

Step 05 records each rebalance's selected stocks in `outputs/full_pipeline_selected_stocks_history.csv`, including the training window (`train_start_date` to `train_end_date`) and `selection_mode = walk_forward_past_data_only`.

The missing holding-return audit is saved in `outputs/full_pipeline_missing_holding_returns.csv`; the latest run has 0 missing-return audit rows.

These outputs support the claim that Step 05 reselects stocks using only past data at each rebalance, although the universe still uses current S&P 500 constituents rather than point-in-time historical membership.

## Step 04 vs Step 05: Leakage Impact

Step 04 used the fixed 25 stocks selected with full-history information. Step 05 reselects stocks every rebalance using only past data. The gap shows how much the fixed selected-stock list inflated the allocation-only backtest.

| Method                                       | 04 Allocation-only CAGR | 05 Full-pipeline CAGR | CAGR Gap |
| -------------------------------------------- | ----------------------- | --------------------- | -------- |
| Equal Weight                                 | 27.37%                  | 12.09%                | 15.28%   |
| Inverse Volatility                           | 25.48%                  | 12.70%                | 12.78%   |
| Risk Parity                                  | 23.43%                  | 12.02%                | 11.41%   |
| Markowitz-style Mean-Volatility Optimization | 26.48%                  | 10.41%                | 16.08%   |
| CVaR Bootstrap                               | 17.97%                  | 12.63%                | 5.34%    |
| CVaR Monte Carlo                             | 14.47%                  | 9.87%                 | 4.59%    |
| S&P 500                                      | 10.22%                  | 10.22%                | 0.00%    |

## Most Frequently Selected Stocks in Step 05

| ticker | selected_count | first_selected_date | last_selected_date | average_sharpe | sector                 |
| ------ | -------------- | ------------------- | ------------------ | -------------- | ---------------------- |
| MRNA   | 53             | 2021-12-31          | 2026-04-30         | 0.5691         | Health Care            |
| LLY    | 53             | 2021-12-31          | 2026-04-30         | 1.0910         | Health Care            |
| SW     | 53             | 2021-12-31          | 2026-04-30         | 0.2214         | Materials              |
| NEM    | 53             | 2021-12-31          | 2026-04-30         | 0.2503         | Materials              |
| TKO    | 49             | 2021-12-31          | 2026-04-30         | 0.0961         | Communication Services |
| COST   | 48             | 2021-12-31          | 2026-04-30         | 1.1044         | Consumer Staples       |
| MCK    | 47             | 2022-01-31          | 2026-04-30         | 1.0125         | Health Care            |
| PWR    | 44             | 2022-02-28          | 2026-04-30         | 1.2515         | Industrials            |
| DPZ    | 42             | 2021-12-31          | 2025-11-28         | 0.2578         | Consumer Discretionary |
| PGR    | 39             | 2022-06-30          | 2026-04-30         | 0.8590         | Financials             |
| PM     | 35             | 2022-01-31          | 2026-04-30         | 0.4829         | Consumer Staples       |
| IRM    | 33             | 2022-02-28          | 2026-04-30         | 0.6600         | Real Estate            |
| KR     | 32             | 2021-12-31          | 2026-04-30         | 0.4491         | Consumer Staples       |
| CBOE   | 30             | 2023-06-30          | 2026-04-30         | 0.3952         | Financials             |
| PCG    | 29             | 2021-12-31          | 2024-06-28         | -0.1823        | Utilities              |

## Latest Full-History Selected Stocks from Step 02

The static Step 02 selected-stock list is useful for the current allocation view, but it should not be treated as a leakage-free historical stock list because it uses full-history information.

The full table is available in `outputs/selected_stocks.csv`; Step 05 should be used for the main historical evaluation.

## Final Interpretation

- Use **Step 05** as the main historical evaluation of the workflow.
- **CVaR Bootstrap** has the best full-pipeline Sharpe and lowest drawdown among the tested models.
- **Inverse Volatility** has the highest full-pipeline CAGR and is simpler than the optimizer-based methods.
- **Markowitz-style Mean-Volatility Optimization** is sensitive to expected-return estimation and had the worst drawdown in the full-pipeline test.
- **CVaR Monte Carlo** had lower volatility but did not beat the `^GSPC` S&P 500 benchmark on CAGR in this current-constituent simulation.
- In this current-constituent walk-forward simulation, some strategy variants outperformed the `^GSPC` benchmark over the tested period. This should not be interpreted as evidence of a fully point-in-time historical trading edge.

## Limitations And Known Weaknesses

- The S&P 500 universe uses the current Wikipedia constituent list rather than point-in-time historical membership, so results can contain survivorship/current-constituent bias.
- Yahoo Finance data can contain missing values, ticker mapping changes, revisions, or adjusted-price methodology differences.
- The benchmark uses `^GSPC`, a price index, while stock returns use adjusted close prices, so the benchmark comparison is not fully total-return equivalent.
- Step 04 is allocation-only and uses the fixed Step 02 stock list selected with full-history data; it should not be treated as the main strategy backtest.
- Step 05 reselects stocks and reallocates monthly using only past returns, but it still does not solve point-in-time S&P 500 membership bias.
- Historical Sharpe ranking is backward-looking and should not be interpreted as a direct forecast of future winners.
- Markowitz-style optimization relies on noisy historical expected-return estimates, and its in-sample parameter selection can overfit without nested walk-forward validation.
- CVaR Bootstrap depends on historical sampled days, while CVaR Monte Carlo assumes multivariate normal returns that may understate fat tails and regime shifts.
- Transaction cost is modeled as a simple proportional cost of 0.001 per turnover and excludes bid-ask spread, slippage, market impact, taxes, liquidity constraints, and execution frictions.
- The strategy does not explicitly constrain sector exposure, and statistical significance or regime stress tests have not yet been added.
- Results are research evidence for this current-constituent simulation, not live investment advice.

## Recommended Improvements

1. Add point-in-time S&P 500 constituent history.
2. Add nested walk-forward validation for `N_CLUSTERS`, `MAX_WEIGHT`, Markowitz-style delta, and CVaR tradeoff.
3. Add transaction-cost sensitivity tests, such as 0.00%, 0.10%, 0.25%, and 0.50% per turnover.
4. Add alternative rebalance frequencies: monthly, quarterly, and semiannual.
5. Add robustness windows: start in 2021, 2022, 2023, and compare results.
6. Add benchmark-relative metrics: alpha, tracking error, information ratio, beta, and excess CAGR.
7. Compare portfolio sector exposures against the S&P 500 to determine whether performance comes from stock selection, diversification, or unintended sector tilts.
8. Test cluster-count sensitivity across multiple cluster counts such as 15, 20, 25, and 30.
9. Test alternative correlation-distance definitions, including `sqrt(2*(1-rho))`.

## Key Output Files

- `outputs/final_full_pipeline_metrics_table.csv`
- `outputs/final_04_vs_05_cagr_table.csv`
- `outputs/full_pipeline_selected_stocks_history.csv`
- `outputs/full_pipeline_missing_holding_returns.csv`
- `outputs/full_pipeline_config.csv`
- `outputs/final_full_pipeline_scorecard.png`
- `outputs/final_04_vs_05_cagr_comparison.png`
- `outputs/final_top_selected_stocks.png`
- `outputs/final_selection_stability_and_turnover.png`
- `outputs/full_pipeline_equity_curves.png`
- `outputs/full_pipeline_relative_wealth_vs_benchmark.png`
