# File List For Review

Last updated: 2026-05-05

This list contains the main files to inspect for an academic or GitHub review of the project. It intentionally excludes temporary files, local virtual environments, and editor-specific files.

## Main Workflow

| Step | File | Purpose |
|---|---|---|
| 01 | `notebooks/01_prepare_sp500_data.ipynb` | Build current S&P 500 universe, price data, return matrix, benchmark, and data-quality outputs. |
| 02 | `notebooks/02_select_stocks_clustering_mst.ipynb` | Cluster stocks by Spearman correlation and select one historical-Sharpe-ranked stock per cluster. |
| 03 | `notebooks/03_allocate_portfolios.ipynb` | Allocate latest selected stocks and produce Markowitz-style, CVaR, risk contribution, and allocation comparison diagnostics. |
| 04 | `notebooks/04_backtest_allocation_only.ipynb` | Allocation-only walk-forward test using the fixed Step 02 selected-stock list. |
| 05 | `notebooks/05_backtest_full_pipeline_walkforward.ipynb` | Main full-pipeline walk-forward test with stock selection and allocation rerun at each rebalance. |
| 06 | `notebooks/06_final_summary_report.ipynb` | Final summary report and comparison charts. |

## Documentation

- `README.md`
- `docs/final_summary_report.md`
- `docs/data_leakage_and_cross_validation_audit.md`
- `docs/references_and_assumptions.md`
- `FILE_LIST_FOR_REVIEW.md`

## Script Versions

- `scripts/04_backtest_allocation_only.py`
- `scripts/05_backtest_full_pipeline_walkforward.py`
- `scripts/06_final_summary_report.py`

## Key Data Outputs

- `data/sp500_universe.csv`
- `data/returns_matrix.parquet`
- `data/benchmark_returns.parquet`
- `data/data_quality_report.csv`
- `outputs/selected_stocks.csv`
- `outputs/stock_sharpe_selection_table.csv`
- `outputs/cluster_assignments.csv`

## Step 03 Portfolio Outputs

- `outputs/portfolio_allocation_summary.csv`
- `outputs/portfolio_weights_all.csv`
- `outputs/portfolio_markowitz_frontier_summary.csv`
- `outputs/portfolio_cvar_frontier_summary.csv`
- `outputs/portfolio_markowitz_efficient_frontier.png`
- `outputs/portfolio_cvar_efficient_frontier.png`
- `outputs/portfolio_mean_cvar_frontier.png`
- `outputs/portfolio_allocation_risk_return_scatter.png`
- `outputs/portfolio_absolute_risk_contribution_heatmap.png`
- `outputs/portfolio_cvar_scenario_return_distributions.png`

## Step 04 Backtest Outputs

- `outputs/backtest_metrics.csv`
- `outputs/backtest_equity_curves.csv`
- `outputs/backtest_daily_returns.csv`
- `outputs/backtest_turnover.csv`
- `outputs/backtest_risk_return_scatter.png`
- `outputs/backtest_relative_wealth_vs_benchmark.png`

## Step 05 Full-Pipeline Outputs

- `outputs/full_pipeline_metrics.csv`
- `outputs/full_pipeline_selected_stocks_history.csv`
- `outputs/full_pipeline_missing_holding_returns.csv`
- `outputs/full_pipeline_selection_frequency.csv`
- `outputs/full_pipeline_selected_overlap.csv`
- `outputs/full_pipeline_equity_curves.png`
- `outputs/full_pipeline_relative_wealth_vs_benchmark.png`
- `outputs/full_pipeline_selection_stability.png`

## Final Report Outputs

- `outputs/final_full_pipeline_scorecard.png`
- `outputs/final_04_vs_05_cagr_comparison.png`
- `outputs/final_top_selected_stocks.png`
- `outputs/final_selection_stability_and_turnover.png`
