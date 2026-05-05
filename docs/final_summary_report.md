# Final Project Summary

Generated: 2026-05-02 23:46:50

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
| Missing holding-return audit rows     | 0                        |

## Workflow Summary

| Step | File                                                  | Purpose                                                                               |
| ---- | ----------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 01   | notebooks/01_prepare_sp500_data.ipynb                 | Prepare S&P 500 universe, prices, returns, benchmark, and quality reports             |
| 02   | notebooks/02_select_stocks_clustering_mst.ipynb       | Cluster stocks and select one stock per cluster using historical Sharpe as a backward-looking ranking heuristic |
| 03   | notebooks/03_allocate_portfolios.ipynb                | Create current portfolio allocations using Equal Weight, Inverse Volatility, Markowitz-style Mean-Volatility Optimization, Risk Parity, CVaR Bootstrap, and CVaR Monte Carlo |
| 04   | notebooks/04_backtest_allocation_only.ipynb           | Allocation-only walk-forward test using the fixed Step 02 selected-stock list |
| 05   | notebooks/05_backtest_full_pipeline_walkforward.ipynb | Main full-pipeline walk-forward simulation with stock selection and allocation rerun each rebalance |
| 06   | notebooks/06_final_summary_report.ipynb               | Final project summary and comparison report                                           |

## Step 05 Full-Pipeline Results

| Method             | Final Value | CAGR   | Sharpe | Max Drawdown | Volatility | Turnover |
| ------------------ | ----------- | ------ | ------ | ------------ | ---------- | -------- |
| Equal Weight       | 1.6381      | 12.09% | 0.5343 | -23.32%      | 16.34%     | 28.95    |
| Inverse Volatility | 1.6769      | 12.70% | 0.6144 | -21.78%      | 14.68%     | 29.46    |
| Risk Parity        | 1.6337      | 12.02% | 0.5772 | -21.68%      | 14.54%     | 27.34    |
| Markowitz-style Mean-Volatility Optimization | 1.5341      | 10.41% | 0.4013 | -30.50%      | 19.25%     | 30.61    |
| CVaR Bootstrap     | 1.6721      | 12.63% | 0.6357 | -20.64%      | 13.91%     | 34.89    |
| CVaR Monte Carlo   | 1.5025      | 9.87%  | 0.4644 | -23.67%      | 13.60%     | 32.20    |
| S&P 500            | 1.5231      | 10.22% | 0.4127 | -25.43%      | 17.55%     | 0.00     |

## Step 05 Audit Outputs

Step 05 exports `outputs/full_pipeline_selected_stocks_history.csv` with audit metadata for every selected-stock record:

- `train_start_date`: first return date available in the expanding training window
- `train_end_date`: rebalance date used as the final training date
- `selection_mode`: `walk_forward_past_data_only`

This metadata makes the selected-stock history easier to audit because each record states the data window used for selection. Step 05 also exports `outputs/full_pipeline_missing_holding_returns.csv` to audit missing realized holding-period returns. The latest run has zero missing-return audit rows.

## Step 04 vs Step 05: Leakage Impact

Step 04 used the fixed 25 stocks selected with full-history information. Step 05 reselects stocks every rebalance using only past data. The gap shows how much the fixed selected-stock list inflated the allocation-only backtest.

| Method             | 04 Allocation-only CAGR | 05 Full-pipeline CAGR | CAGR Gap |
| ------------------ | ----------------------- | --------------------- | -------- |
| Equal Weight       | 27.37%                  | 12.09%                | 15.28%   |
| Inverse Volatility | 25.48%                  | 12.70%                | 12.78%   |
| Risk Parity        | 23.43%                  | 12.02%                | 11.41%   |
| Markowitz-style Mean-Volatility Optimization | 26.48%                  | 10.41%                | 16.08%   |
| CVaR Bootstrap     | 17.97%                  | 12.63%                | 5.34%    |
| CVaR Monte Carlo   | 14.47%                  | 9.87%                 | 4.59%    |
| S&P 500            | 10.22%                  | 10.22%                | 0.00%    |

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

These are useful for the current allocation view, but should not be treated as a leakage-free historical stock list.

| ticker | cluster_id | sector                 | annual_return | annual_volatility | sharpe_ratio |
| ------ | ---------- | ---------------------- | ------------- | ----------------- | ------------ |
| PM     | 1          | Consumer Staples       | 19.11%        | 25.40%            | 0.6075       |
| TGT    | 2          | Consumer Staples       | 12.67%        | 35.06%            | 0.2563       |
| COST   | 3          | Consumer Staples       | 26.37%        | 22.67%            | 1.0006       |
| KR     | 4          | Consumer Staples       | 15.78%        | 28.06%            | 0.4311       |
| IRM    | 5          | Real Estate            | 26.94%        | 31.83%            | 0.7308       |
| MCD    | 6          | Consumer Discretionary | 9.75%         | 21.55%            | 0.2818       |
| TMUS   | 7          | Communication Services | 16.88%        | 26.67%            | 0.4951       |
| CBOE   | 8          | Financials             | 19.60%        | 26.27%            | 0.6059       |
| PGR    | 9          | Financials             | 22.15%        | 26.17%            | 0.7059       |
| STX    | 10         | Information Technology | 54.50%        | 42.60%            | 1.1930       |
| NEM    | 11         | Materials              | 20.89%        | 36.89%            | 0.4665       |
| NVDA   | 12         | Information Technology | 74.73%        | 50.96%            | 1.3941       |
| VRT    | 13         | Industrials            | 62.01%        | 56.07%            | 1.0403       |
| TKO    | 14         | Communication Services | 14.96%        | 37.28%            | 0.3026       |
| SW     | 15         | Materials              | 9.91%         | 45.93%            | 0.1357       |
| PWR    | 16         | Industrials            | 55.05%        | 36.19%            | 1.4194       |
| HWM    | 17         | Industrials            | 47.56%        | 40.50%            | 1.0833       |
| MRNA   | 18         | Health Care            | 15.73%        | 72.01%            | 0.1673       |
| AAPL   | 19         | Information Technology | 31.89%        | 30.85%            | 0.9145       |
| EA     | 20         | Communication Services | 13.90%        | 28.15%            | 0.3630       |
| STLD   | 21         | Materials              | 34.42%        | 41.68%            | 0.7376       |
| MCK    | 22         | Health Care            | 32.05%        | 28.54%            | 0.9942       |
| LLY    | 23         | Health Care            | 35.87%        | 32.98%            | 0.9760       |
| MSI    | 24         | Information Technology | 21.67%        | 26.67%            | 0.6746       |
| HCA    | 25         | Health Care            | 20.20%        | 35.17%            | 0.4698       |

## Final Interpretation

- Use **Step 05** as the main historical evaluation of the workflow.
- **CVaR Bootstrap** has the best full-pipeline Sharpe and lowest drawdown among the tested models.
- **Inverse Volatility** has the highest full-pipeline CAGR and is simpler than the optimizer-based methods.
- **Markowitz-style Mean-Volatility Optimization** is sensitive to expected-return estimation and had the worst drawdown in the full-pipeline test.
- **CVaR Monte Carlo** had lower volatility but did not beat the `^GSPC` S&P 500 benchmark on CAGR in this current-constituent simulation.
- In this current-constituent walk-forward simulation, some strategy variants outperformed the `^GSPC` benchmark over the tested period. This should not be interpreted as evidence of a fully point-in-time historical trading edge.

## Limitations And Known Weaknesses

### Data Limitations

- The S&P 500 universe is based on the current Wikipedia constituent list, not point-in-time historical membership. This can create survivorship and constituent bias. The results should be interpreted as a current-constituent walk-forward simulation, not as a fully point-in-time historical S&P 500 backtest.
- Yahoo Finance data can contain revisions, ticker mapping changes, missing fields, delisting gaps, or adjusted-price methodology changes.
- Missing return handling can affect results, especially for delisted, suspended, or sparsely traded securities. Any fill rules should be interpreted as simplifying data assumptions rather than true tradable outcomes.
- When realized holding-period returns are missing, the current full-pipeline implementation may fill missing values with `0.0`. This is a simplifying assumption and may understate losses or distort performance for delisted or suspended securities.
- The benchmark uses `^GSPC`, which is a price index and may not include reinvested dividends. Since stock returns are calculated from adjusted close prices, benchmark comparisons may not be fully total-return equivalent.
- For a stricter no-lookahead implementation, the risk-free rate series should be lagged by one business day because some macro or rate data may not be available before portfolio decisions are made.
- The backtest period starts holding on 2022-01-03 and ends on 2026-05-01, so the test covers only about 4.3 years of realized holding history.
- The 2026 calendar-year result is YTD only, not a full-year result.

### Methodology Limitations

- Step 04 is intentionally allocation-only and uses the fixed 25 stocks from Step 02; it should not be used as the main historical strategy result.
- Step 05 fixes the fixed-stock look-ahead issue by reselecting each rebalance and now records `train_start_date`, `train_end_date`, and `selection_mode` in the selected-stock history, but it still does not fix point-in-time S&P 500 membership bias.
- The within-cluster stock selection uses historical Sharpe ratio as a backward-looking ranking heuristic, not as a predictive model of future returns. High past Sharpe may not persist out of sample.
- The clustering distance uses 1 minus Spearman correlation as a practical similarity-to-distance transformation. Alternative correlation-distance definitions such as `sqrt(2*(1-rho))` could be tested in future robustness checks.
- The number of clusters is fixed at 25 as a design choice. Different cluster counts may lead to different diversification and performance results. Future work should test sensitivity across multiple cluster counts such as 15, 20, 25, and 30.
- Transaction cost is modeled as a simple proportional turnover cost of 0.001. It does not include bid-ask spread, market impact, tax, liquidity, borrow constraints, or execution slippage.
- Rebalancing is monthly. Results may change materially with weekly, quarterly, or threshold-based rebalancing.
- The strategy assumes fractional shares and immediate execution at return-series prices.

### Model Limitations

- The Markowitz implementation is Markowitz-style mean-risk optimization using volatility as the risk penalty, rather than the classic mean-variance objective using variance directly.
- Markowitz-style Mean-Volatility Optimization depends heavily on historical expected-return estimates, which are noisy and unstable for equities.
- The Markowitz-style allocation selects optimization settings using in-sample training data at each rebalance. This does not create direct look-ahead bias, but it may overfit the training window. A stricter design would use nested walk-forward validation.
- CVaR Bootstrap depends on historical sampled days and may miss unseen future regimes.
- The Monte Carlo CVaR method depends on a multivariate normal return assumption. Because equity returns can exhibit fat tails, skewness, and regime shifts, this assumption may underestimate tail risk. The bootstrap CVaR method is less parametric because it samples from historical return observations.
- Risk Parity and Inverse Volatility reduce risk exposure but do not directly forecast future returns.
- The strategy does not explicitly constrain sector exposure. Selected stocks may create unintended sector tilts relative to the S&P 500 benchmark.
- The 10% max-weight cap is a project risk-control assumption, not a universal optimal setting.

### Validation Limitations

- Hyperparameters such as `N_CLUSTERS`, `MAX_WEIGHT`, Markowitz-style delta grid, CVaR tradeoff, and scenario count are not selected through nested walk-forward validation.
- The same broad research period influenced model design choices, so there is still research/iteration overfitting risk.
- Even without direct future-return leakage, repeated experimentation with model design, parameters, and reporting choices can create research overfitting. Results should be interpreted as research findings rather than confirmed trading edge.
- Statistical significance is not tested. Differences such as 12.70% vs 12.63% CAGR should not be treated as conclusive without robustness checks.
- No stress test by market regime, sector exposure, liquidity bucket, or alternative start date is included yet.

### Interpretation Warnings

- A positive Sharpe ratio means return exceeded the risk-free rate per unit volatility; it does not necessarily mean the model beat the S&P 500 benchmark.
- Historical Sharpe ranking is backward-looking and should not be interpreted as a direct prediction of future winners.
- Best CAGR, best Sharpe, and lowest drawdown can point to different models. The final choice depends on the investor's objective.
- Current results are research evidence, not a live trading recommendation.

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

## Key Output Charts

- `outputs/final_full_pipeline_scorecard.png`
- `outputs/final_04_vs_05_cagr_comparison.png`
- `outputs/final_top_selected_stocks.png`
- `outputs/final_selection_stability_and_turnover.png`
- `outputs/full_pipeline_equity_curves.png`
- `outputs/full_pipeline_relative_wealth_vs_benchmark.png`
- `outputs/portfolio_mean_cvar_frontier.png`
- `outputs/portfolio_absolute_risk_contribution_heatmap.png`
