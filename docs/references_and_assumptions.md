# References and Assumptions

Last updated: 2026-05-03

This file records where each part of the S&P 500 stock-selection and portfolio-allocation workflow came from, and which values are project assumptions rather than external defaults.

## Project Workflow Files

| Step | File | Purpose |
|---|---|---|
| 1 | `notebooks/01_prepare_sp500_data.ipynb` | Build S&P 500 universe, download price data, create returns matrix and benchmark returns. |
| 2 | `notebooks/02_select_stocks_clustering_mst.ipynb` | Use Spearman correlation, hierarchical clustering, and historical excess-Sharpe ranking to pick one stock per cluster. |
| 3 | `notebooks/03_allocate_portfolios.ipynb` | Allocate portfolio weights with Equal Weight, Inverse Volatility, Markowitz-style Mean-Volatility Optimization, Risk Parity, CVaR Bootstrap, and CVaR Monte Carlo. |
| 4 | `notebooks/04_backtest_allocation_only.ipynb` / `scripts/04_backtest_allocation_only.py` | Walk-forward backtest of allocation methods using the fixed Step 02 stock list. |
| 5 | `notebooks/05_backtest_full_pipeline_walkforward.ipynb` / `scripts/05_backtest_full_pipeline_walkforward.py` | Walk-forward backtest that re-runs stock selection and allocation at every rebalance. |
| 6 | `notebooks/06_final_summary_report.ipynb` / `scripts/06_final_summary_report.py` | Build the final summary report, comparison tables, and charts. |

## External Data Sources

| Source | Used For | Where Used | Link |
|---|---|---|---|
| Wikipedia list of S&P 500 companies | Current S&P 500 universe, sectors, sub-industries, ticker symbols | `notebooks/01_prepare_sp500_data.ipynb` | https://en.wikipedia.org/wiki/List_of_S%26P_500_companies |
| Yahoo Finance data through `yfinance` | Adjusted close prices, close prices, volume, benchmark price data | `notebooks/01_prepare_sp500_data.ipynb` | https://pypi.org/project/yfinance/ |
| FRED `DGS3MO` | Real short-term U.S. Treasury proxy for risk-free rate | `notebooks/02_select_stocks_clustering_mst.ipynb`, `notebooks/03_allocate_portfolios.ipynb` | https://fred.stlouisfed.org/series/DGS3MO |
| Yahoo Finance `^IRX` chart endpoint | Fallback risk-free proxy if FRED fetch fails | `notebooks/02_select_stocks_clustering_mst.ipynb`, `notebooks/03_allocate_portfolios.ipynb` | https://finance.yahoo.com/quote/%5EIRX/ |

## Source Repository Reference

| Source | Used For | Link |
|---|---|---|
| `investment-funnel` by VanekPetr | Overall idea of investment funnel workflow: feature selection, scenario generation, Markowitz/CVaR optimization, and backtesting separation | https://github.com/VanekPetr/investment-funnel |

Also inspected a local clone of `investment-funnel` and the `ifunnel==0.0.6` package source during development to compare the model structure.

Important adaptation: the original repo works with ETFs/funds. This project adapts the idea to individual S&P 500 stocks.

This project uses the current S&P 500 constituent universe as the stock universe. Therefore, the results should be interpreted as a current-constituent walk-forward simulation, not as a fully point-in-time historical S&P 500 backtest.

## Portfolio Model References

| Model / Concept | What We Took | Our Implementation | Reference |
|---|---|---|---|
| Markowitz-style Mean-Volatility Optimization | Efficient-frontier idea and mean-risk tradeoff | MOSEK-style frontier: maximize `annual_return - delta * annual_volatility`; 10 log-spaced delta values over MOSEK's example range | https://docs.mosek.com/portfolio-cookbook/markowitz.html |
| Markowitz constraints | Budget and long-only style constraints, plus diversification limits | Sum weights = 1, long-only, max 10% per stock | https://docs.mosek.com/portfolio-cookbook/markowitz.html |
| CVaR risk measure | Rockafellar-Uryasev style linear programming form using scenarios, VaR threshold, and tail excess variables | Minimize `CVaR(loss) - return_tradeoff * expected_return` with Bootstrap and Monte Carlo scenarios | https://docs.mosek.com/portfolio-cookbook/riskmeasures.html |
| Risk Parity / Risk Budgeting | Equal risk contribution idea | Solve for weights whose normalized risk contributions are close to `1/N` | https://docs.mosek.com/portfolio-cookbook/risk_parity.html |
| Bootstrap scenarios | Resampling observations with replacement | Sample whole historical daily-return rows to preserve cross-asset co-movement | https://en.wikipedia.org/wiki/Bootstrapping_(statistics) |
| Monte Carlo scenarios | Random simulation from an assumed distribution | Sample daily stock-return vectors from multivariate normal fitted to historical mean/covariance | https://en.wikipedia.org/wiki/Monte_Carlo_method |

## What Came From `investment-funnel`

| Item | Source Idea | Our Adaptation |
|---|---|---|
| Investment funnel structure | Separate universe preparation, feature selection, optimization, and backtest/evaluation | `notebooks/01` through `notebooks/06` |
| Hierarchical clustering | Use correlation structure to group similar assets | Use Spearman correlation for S&P 500 stocks and select one stock per cluster |
| MST | Used in the repo as a graph-based feature-selection method | Used here only as a relationship visualization, not the primary stock selector |
| Scenario generation | Bootstrap and Monte Carlo scenarios for CVaR | Generate scenarios only for CVaR optimization |
| Markowitz/CVaR optimization | Optimization model families in the repo | Implemented in SciPy instead of MOSEK/CVXPY for local reproducibility |

The Markowitz implementation is Markowitz-style mean-risk optimization using volatility as the risk penalty, rather than the classic mean-variance objective using variance directly.

## Project-Specific Assumptions

These values are not fixed by MOSEK, Wikipedia, FRED, Yahoo Finance, or the original repo. They are project choices.

| Assumption | Current Value | Reason |
|---|---:|---|
| Number of clusters | `25` | Target about 25 selected stocks, one per cluster |
| Stocks per cluster | `1` | Reduce holding multiple highly correlated names from the same cluster |
| Correlation method | Spearman | More robust to nonlinear monotonic relationships and outliers than Pearson |
| Minimum overlap for pairwise correlation | `252` trading days | Allow newer stocks while requiring at least about one trading year of overlap |
| Minimum observations for Sharpe selection | `756` trading days | Require about three trading years before a stock can be ranked by historical Sharpe |
| Portfolio max weight | `10%` | Concentration control for individual stocks |
| Portfolio min weight | `0%` | Long-only portfolio; no short selling |
| CVaR alpha | `95%` | Focus on worst 5% daily-return scenarios |
| Number of scenarios | `3000` | Balance stability and runtime |
| CVaR return tradeoff grid | `[0.1, 0.3, 1.0, 3.0, 10.0]` | Conservative-to-aggressive comparison |
| Markowitz-style delta grid | `np.logspace(-1, 1.5, 10)[::-1]` | 10 values over the same range used in the MOSEK example |
| Markowitz-style headline representative | Max-training-Sharpe point from Markowitz-style frontier | Avoid presenting any single delta as theoretically fixed |
| Training window for selection and allocation | Full available history, currently `2019-01-03` to `2026-05-01` | Keep Step 2 and Step 3 consistent |
| Benchmark | `^GSPC` | S&P 500 price-index benchmark; may not be total-return equivalent to adjusted-close stock returns |
| Random seed | `42` | Reproducibility for scenario generation |

## Backtest Setup

Step 4 is an allocation-only walk-forward backtest inspired by the original repo's custom `algo.backtest()` flow.

| Item | Current Setting | Notes |
|---|---:|---|
| Backtest file | `notebooks/04_backtest_allocation_only.ipynb` / `scripts/04_backtest_allocation_only.py` | The `.py` file uses VS Code notebook cell markers |
| Universe | Fixed 25 stocks from `outputs/selected_stocks.csv` | Tests allocation methods, not the full stock-selection pipeline |
| First rebalance date | `2021-12-31` | First date with at least 756 prior trading observations |
| First holding date | `2022-01-03` | First out-of-sample return date |
| Last holding date | `2026-05-01` | Latest available return date in the current dataset |
| Rebalance frequency | Month-end trading day | Similar spirit to the source repo's rolling rebalance approach |
| Training window | Expanding window | Each rebalance uses only data available up to that rebalance date |
| Transaction cost | `0.001` | Simple proportional turnover cost; excludes bid-ask spreads, market impact, taxes, borrow costs, liquidity constraints, and execution slippage |
| Realized performance data | Actual daily returns only | Scenarios are not used to simulate realized performance |
| CVaR Bootstrap scenarios | Historical daily-return rows sampled with replacement | Used only to optimize CVaR weights |
| CVaR Monte Carlo scenarios | Multivariate normal scenarios fitted from historical mean/covariance | Used only to optimize CVaR weights; this assumption may underestimate fat tails, skewness, and regime shifts |

## Current Output Interpretation

| Output | Meaning |
|---|---|
| `outputs/selected_stocks.csv` | Final 25 selected stocks after clustering and excess-Sharpe selection |
| `outputs/stock_selection_risk_free_rate.csv` | Risk-free rate used in Step 2 Sharpe selection |
| `outputs/risk_free_rate.csv` | Risk-free rate used in Step 3 portfolio metrics |
| `outputs/portfolio_markowitz_frontier_summary.csv` | Main Markowitz-style Mean-Volatility Optimization result: 10 delta points on the efficient frontier |
| `outputs/portfolio_weights_markowitz_best_sharpe_default.csv` | Representative Markowitz-style point used in the headline comparison |
| `outputs/portfolio_cvar_frontier_summary.csv` | CVaR frontier points for Bootstrap and Monte Carlo scenarios |
| `outputs/portfolio_allocation_summary.csv` | Headline comparison across portfolio allocation methods |
| `outputs/portfolio_allocation_risk_return_scatter.png` | Step 3 risk-return view of the allocation methods |
| `outputs/portfolio_allocation_metric_bars.png` | Step 3 metric dashboard: return, volatility, Sharpe, and CVaR |
| `outputs/portfolio_top10_weights_by_method.png` | Step 3 top holdings for each allocation method |
| `outputs/portfolio_risk_contribution_heatmap.png` | Step 3 risk contribution by ticker and method |
| `outputs/portfolio_cvar_scenario_return_distributions.png` | Step 3 Bootstrap vs Monte Carlo scenario distributions for CVaR portfolios |
| `outputs/backtest_metrics.csv` | Out-of-sample performance metrics for each allocation method and benchmark |
| `outputs/backtest_equity_curves.csv` | Growth of 1.0 through the backtest period |
| `outputs/backtest_daily_returns.csv` | Net daily portfolio returns after transaction cost |
| `outputs/backtest_weights_history.csv` | Target weights at each rebalance date |
| `outputs/backtest_turnover.csv` | Turnover and transaction-cost records for each rebalance |
| `outputs/backtest_risk_return_scatter.png` | Step 4 realized risk-return view from the backtest |
| `outputs/backtest_metric_dashboard.png` | Step 4 final-value, CAGR, Sharpe, and drawdown dashboard |
| `outputs/backtest_relative_wealth_vs_benchmark.png` | Step 4 portfolio value divided by S&P 500 benchmark value |
| `outputs/backtest_calendar_year_returns.png` | Step 4 calendar-year returns by method |
| `outputs/backtest_rolling_252d_return.png` | Step 4 rolling one-year return |
| `outputs/backtest_rolling_252d_sharpe.png` | Step 4 rolling one-year Sharpe ratio |
| `outputs/full_pipeline_metrics.csv` | Step 5 full-pipeline walk-forward performance metrics |
| `outputs/full_pipeline_selected_stocks_history.csv` | Step 5 selected stocks at every rebalance date |
| `outputs/full_pipeline_selection_frequency.csv` | Step 5 frequency of each stock being selected across rebalance dates |
| `outputs/full_pipeline_selected_overlap.csv` | Step 5 month-to-month selection stability |
| `outputs/full_pipeline_weights_history.csv` | Step 5 allocation weights after dynamic stock selection |
| `outputs/full_pipeline_equity_curves.png` | Step 5 growth of 1.0 for the full pipeline |
| `outputs/full_pipeline_relative_wealth_vs_benchmark.png` | Step 5 portfolio value divided by S&P 500 benchmark value |
| `outputs/full_pipeline_calendar_year_return_details.csv` | Step 5 year-by-year period start/end markers, including whether the latest year is YTD/partial |
| `outputs/full_pipeline_selected_stock_frequency.png` | Step 5 most frequently selected stocks |
| `outputs/full_pipeline_selection_stability.png` | Step 5 stability of the selected 25-stock set over time |

## Important Limitations

1. The S&P 500 universe comes from the current Wikipedia page, so historical backtests using this universe may have survivorship/constituent bias. Results should be described as a current-constituent walk-forward simulation, not as a fully point-in-time historical S&P 500 backtest.
2. Yahoo Finance data availability and ticker mapping can change over time.
3. Missing return handling can affect results, especially for delisted, suspended, or sparsely traded securities. Any fill rules should be interpreted as simplifying data assumptions rather than true tradable outcomes.
4. When realized holding-period returns are missing, the current full-pipeline implementation may fill missing values with `0.0`. This is a simplifying assumption and may understate losses or distort performance for delisted or suspended securities.
5. For a stricter no-lookahead implementation, the risk-free rate series should be lagged by one business day because some macro or rate data may not be available before portfolio decisions are made.
6. The benchmark uses `^GSPC`, which is a price index and may not include reinvested dividends. Since stock returns are calculated from adjusted close prices, benchmark comparisons may not be fully total-return equivalent.
7. The within-cluster stock selection uses historical Sharpe ratio as a backward-looking ranking heuristic, not as a predictive model of future returns. High past Sharpe may not persist out of sample.
8. The clustering distance uses 1 minus Spearman correlation as a practical similarity-to-distance transformation. Alternative correlation-distance definitions such as `sqrt(2*(1-rho))` could be tested in future robustness checks.
9. The number of clusters is fixed at 25 as a design choice. Different cluster counts may lead to different diversification and performance results.
10. Markowitz-style expected returns are estimated from historical returns and can be noisy.
11. The Markowitz-style allocation selects optimization settings using in-sample training data at each rebalance. This does not create direct look-ahead bias, but it may overfit the training window. A stricter design would use nested walk-forward validation.
12. Monte Carlo CVaR scenarios assume multivariate normal returns, which may understate fat tails, skewness, and regime shifts. Bootstrap CVaR is less parametric because it samples from historical return observations.
13. Bootstrap scenarios preserve same-day cross-asset co-movement but do not preserve serial dependence unless block bootstrap is added later.
14. The strategy does not explicitly constrain sector exposure. Selected stocks may create unintended sector tilts relative to the S&P 500 benchmark.
15. The 10% max-weight cap is a risk-control assumption, not a requirement from MOSEK or the original repo.
16. Step 5 removes look-ahead leakage from stock selection by reselecting stocks at every rebalance with past data only, but it still does not solve point-in-time S&P 500 membership bias.
17. Even without direct future-return leakage, repeated experimentation with model design, parameters, and reporting choices can create research overfitting. Results should be interpreted as research findings rather than confirmed trading edge.

## Recommended Robustness Checks

Future work should:

1. Add point-in-time S&P 500 constituent history.
2. Test cluster-count sensitivity across multiple cluster counts such as 15, 20, 25, and 30.
3. Compare portfolio sector exposures against the S&P 500 to determine whether performance comes from stock selection, diversification, or unintended sector tilts.
4. Test alternative correlation-distance definitions, including `sqrt(2*(1-rho))`.
5. Add nested walk-forward validation for `N_CLUSTERS`, `MAX_WEIGHT`, Markowitz-style delta, and CVaR return tradeoff.
