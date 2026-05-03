# S&P 500 Stock Selection Funnel

This project adapts the idea of an investment-funnel workflow from ETF/fund selection to individual S&P 500 stocks.

The workflow builds a current S&P 500 constituent universe, downloads real price data, selects stocks with correlation-based clustering and backward-looking historical Sharpe ranking, allocates portfolios with several methods, and evaluates the full process with walk-forward simulation.

This project uses the current S&P 500 constituent universe as the stock universe. Therefore, the results should be interpreted as a current-constituent walk-forward simulation, not as a fully point-in-time historical S&P 500 backtest.

## Workflow

| Step | File | Purpose |
|---|---|---|
| 01 | `notebooks/01_prepare_sp500_data.ipynb` | Prepare the current S&P 500 constituent universe, prices, returns, benchmark, and quality reports |
| 02 | `notebooks/02_select_stocks_clustering_mst.ipynb` | Select one stock per correlation cluster using historical Sharpe ratio as a backward-looking ranking heuristic, and create MST diagnostics |
| 03 | `notebooks/03_allocate_portfolios.ipynb` | Allocate the latest selected stocks using Equal Weight, Inverse Volatility, Markowitz-style Mean-Volatility Optimization, Risk Parity, CVaR Monte Carlo, and CVaR Bootstrap |
| 04 | `notebooks/04_backtest_allocation_only.ipynb` | Run an allocation-only walk-forward test using the fixed Step 02 selected-stock list |
| 05 | `notebooks/05_backtest_full_pipeline_walkforward.ipynb` | Run the main full-pipeline walk-forward simulation by re-running stock selection and allocation at each rebalance |
| 06 | `notebooks/06_final_summary_report.ipynb` | Build the final summary report and comparison charts |

Script versions of Steps 04-06 are available in `scripts/` for repeatable execution outside notebooks.

## Portfolio Models

- Equal Weight baseline
- Inverse Volatility baseline
- Markowitz-style Mean-Volatility Optimization
- Risk Parity / equal risk contribution
- CVaR optimization with Bootstrap scenarios
- CVaR optimization with Monte Carlo scenarios

The Markowitz implementation is Markowitz-style mean-risk optimization using volatility as the risk penalty, rather than the classic mean-variance objective using variance directly.

## Repository Structure

```text
stock-selection-funnel/
├── data/       # Generated input datasets from Step 01
├── docs/       # Method notes, audit notes, final report
├── notebooks/  # Main notebook workflow, Steps 01-06
├── outputs/    # Generated tables and charts
├── scripts/    # Script versions of long-running/report steps
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducibility

Tested with Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in numerical order from Step 01 to Step 06. Do not treat later notebooks as independent if the expected files from earlier steps have not been created.

For script-based reruns of later steps:

```bash
python scripts/04_backtest_allocation_only.py
python scripts/05_backtest_full_pipeline_walkforward.py
python scripts/06_final_summary_report.py
```

## Current Main Result

The main historical evaluation is Step 05 because it re-selects stocks and re-allocates at every rebalance using only past returns. In this current-constituent walk-forward simulation, several allocation methods outperformed the `^GSPC` benchmark over the tested period, but this should not be interpreted as proof of a fully point-in-time historical S&P 500 trading edge.

Latest full-pipeline backtest window:

- First holding date: `2022-01-03`
- Last holding date: `2026-05-01`
- Rebalance frequency: month-end
- Transaction cost: `0.10%` per turnover
- Universe limitation: current S&P 500 constituents, not point-in-time historical constituents

| Method | CAGR | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| Equal Weight | 12.09% | 0.5343 | -23.32% |
| Inverse Volatility | 12.70% | 0.6144 | -21.78% |
| Risk Parity | 12.02% | 0.5772 | -21.68% |
| Markowitz-style Mean-Volatility Optimization | 10.41% | 0.4013 | -30.50% |
| CVaR Bootstrap | 12.63% | 0.6357 | -20.64% |
| CVaR Monte Carlo | 9.87% | 0.4644 | -23.67% |
| S&P 500 Benchmark | 10.22% | 0.4127 | -25.43% |

## Important Limitations

- The universe is based on the current Wikipedia S&P 500 list, so historical tests still have current-constituent bias.
- This is a current-constituent walk-forward simulation, not a fully point-in-time historical S&P 500 backtest.
- Yahoo Finance data can include revisions, missing data, and adjusted-price methodology changes.
- Missing return handling can affect results, especially for delisted, suspended, or sparsely traded securities. When realized holding-period returns are missing, the current full-pipeline implementation may fill missing values with `0.0`, which is a simplifying assumption rather than a true tradable outcome.
- The within-cluster stock selection uses historical Sharpe ratio as a backward-looking ranking heuristic, not as a predictive model of future returns. High past Sharpe may not persist out of sample.
- The benchmark uses `^GSPC`, which is a price index and may not include reinvested dividends. Since stock returns are calculated from adjusted close prices, benchmark comparisons may not be fully total-return equivalent.
- Hyperparameters such as cluster count, max weight, Markowitz-style delta grid, and CVaR tradeoff are project assumptions, not universally optimal values.
- Transaction costs are modeled as a simple proportional turnover cost. The model does not include bid-ask spreads, market impact, taxes, borrow costs, liquidity constraints, or execution slippage.
- The strategy does not explicitly constrain sector exposure. Selected stocks may create unintended sector tilts relative to the S&P 500 benchmark.
- Even without direct future-return leakage, repeated experimentation with model design, parameters, and reporting choices can create research overfitting.
- The project is research code and not investment advice.

See `docs/final_summary_report.md`, `docs/data_leakage_and_cross_validation_audit.md`, and `docs/references_and_assumptions.md` for details.
