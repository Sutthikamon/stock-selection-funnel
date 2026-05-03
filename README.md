# S&P 500 Stock Selection Funnel

This project adapts the idea of an investment-funnel workflow from ETF/fund selection to individual S&P 500 stocks.

The workflow builds a current S&P 500 universe, downloads real price data, selects stocks with correlation-based clustering, allocates portfolios with several models, and evaluates the full process with walk-forward backtesting.

## Workflow

| Step | File | Purpose |
|---|---|---|
| 01 | `notebooks/01_prepare_sp500_data.ipynb` | Download S&P 500 universe, prices, returns, benchmark, and quality reports |
| 02 | `notebooks/02_select_stocks_clustering_mst.ipynb` | Select one high-Sharpe stock per correlation cluster and create MST diagnostics |
| 03 | `notebooks/03_allocate_portfolios.ipynb` | Allocate the latest selected stocks with Equal Weight, Inverse Volatility, Markowitz, Risk Parity, and CVaR |
| 04 | `notebooks/04_backtest_allocation_only.ipynb` | Walk-forward backtest of allocation methods on the fixed Step 02 stock list |
| 05 | `notebooks/05_backtest_full_pipeline_walkforward.ipynb` | Walk-forward backtest that re-runs stock selection and allocation each rebalance |
| 06 | `notebooks/06_final_summary_report.ipynb` | Build the final summary report and comparison charts |

Script versions of Steps 04-06 are available in `scripts/` for repeatable execution outside notebooks.

## Portfolio Models

- Equal Weight baseline
- Inverse Volatility baseline
- Markowitz / Mean-Variance efficient frontier
- Risk Parity / equal risk contribution
- CVaR optimization with Bootstrap scenarios
- CVaR optimization with Monte Carlo scenarios

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

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order from `01` to `06`.

For script-based reruns of later steps:

```bash
python scripts/04_backtest_allocation_only.py
python scripts/05_backtest_full_pipeline_walkforward.py
python scripts/06_final_summary_report.py
```

## Current Main Result

The main historical evaluation is Step 05 because it re-selects stocks and re-allocates at every rebalance using only past returns.

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
| Markowitz | 10.41% | 0.4013 | -30.50% |
| CVaR Bootstrap | 12.63% | 0.6357 | -20.64% |
| CVaR Monte Carlo | 9.87% | 0.4644 | -23.67% |
| S&P 500 Benchmark | 10.22% | 0.4127 | -25.43% |

## Important Limitations

- The universe is based on the current Wikipedia S&P 500 list, so historical tests still have current-constituent bias.
- Yahoo Finance data can include revisions, missing data, and adjusted-price methodology changes.
- Hyperparameters such as cluster count, max weight, Markowitz delta grid, and CVaR tradeoff are project assumptions, not universally optimal values.
- The project is research code and not investment advice.

See `docs/final_summary_report.md`, `docs/data_leakage_and_cross_validation_audit.md`, and `docs/references_and_assumptions.md` for details.
