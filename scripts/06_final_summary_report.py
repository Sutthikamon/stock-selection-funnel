# %% [markdown]
# # 06 Final Summary Report
#
# This final notebook summarizes the whole project from data preparation through full-pipeline
# walk-forward backtesting. It reads the real outputs from steps 01-05 and writes a compact
# final report plus summary charts.

# %%
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------
def find_project_root() -> Path:
    """Find the repository root from either a script run or notebook run."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd())

    for start in dict.fromkeys(candidates):
        for candidate in [start, *start.parents]:
            if (candidate / "data" / "returns_matrix.parquet").exists():
                return candidate

    raise FileNotFoundError("Could not find project root containing data/returns_matrix.parquet.")


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Outputs will be written to: {OUTPUT_DIR}")

# %% [markdown]
# ## Load Final Outputs

# %%
returns_matrix = pd.read_parquet(DATA_DIR / "returns_matrix.parquet").sort_index()
benchmark_returns = pd.read_parquet(DATA_DIR / "benchmark_returns.parquet").sort_index()
sp500_universe = pd.read_csv(DATA_DIR / "sp500_universe.csv")

selected_latest = pd.read_csv(OUTPUT_DIR / "selected_stocks.csv")
portfolio_summary = pd.read_csv(OUTPUT_DIR / "portfolio_allocation_summary.csv").set_index("method")
allocation_backtest = pd.read_csv(OUTPUT_DIR / "backtest_metrics.csv").set_index("method")
full_pipeline = pd.read_csv(OUTPUT_DIR / "full_pipeline_metrics.csv").set_index("method")
full_pipeline_config = pd.read_csv(OUTPUT_DIR / "full_pipeline_config.csv")
selection_frequency = pd.read_csv(OUTPUT_DIR / "full_pipeline_selection_frequency.csv")
selection_overlap = pd.read_csv(OUTPUT_DIR / "full_pipeline_selected_overlap.csv")
selection_history = pd.read_csv(OUTPUT_DIR / "full_pipeline_selected_stocks_history.csv")
annual_returns = pd.read_csv(OUTPUT_DIR / "full_pipeline_calendar_year_returns.csv")
annual_return_details = pd.read_csv(OUTPUT_DIR / "full_pipeline_calendar_year_return_details.csv")

returns_matrix.index = pd.to_datetime(returns_matrix.index)
benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

config = dict(zip(full_pipeline_config["setting"], full_pipeline_config["value"]))

METHOD_LABELS = {
    "equal": "Equal Weight",
    "inverse_volatility": "Inverse Volatility",
    "risk_parity": "Risk Parity",
    "markowitz_best_sharpe_default": "Markowitz",
    "cvar_bootstrap": "CVaR Bootstrap",
    "cvar_montecarlo": "CVaR Monte Carlo",
    "benchmark_sp500": "S&P 500",
}

METHOD_ORDER = [
    "equal",
    "inverse_volatility",
    "risk_parity",
    "markowitz_best_sharpe_default",
    "cvar_bootstrap",
    "cvar_montecarlo",
    "benchmark_sp500",
]

print("Data window:", returns_matrix.index.min().date(), "to", returns_matrix.index.max().date())
print("Returns matrix:", returns_matrix.shape)
print("Latest selected stocks:", len(selected_latest))
print("Full-pipeline methods:", full_pipeline.shape[0])

# %% [markdown]
# ## Helper Functions

# %%
def pct(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x * 100:.{digits}f}%"


def num(x, digits=4):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    df = df.copy()
    df = df.reset_index(drop=True)
    headers = list(df.columns)
    rows = df.astype(str).values.tolist()
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows)) if rows else len(str(headers[i]))
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def make_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        if method not in metrics.index:
            continue
        row = metrics.loc[method]
        rows.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Final Value": num(row["final_value"], 4),
                "CAGR": pct(row["cagr"]),
                "Sharpe": num(row["sharpe_ratio"], 4),
                "Max Drawdown": pct(row["max_drawdown"]),
                "Volatility": pct(row["annualized_volatility"]),
                "Turnover": num(row.get("total_turnover", 0), 2),
            }
        )
    return pd.DataFrame(rows)


def show_or_close(fig):
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


def label_method(method):
    return METHOD_LABELS.get(method, method)


full_pipeline_table = make_metric_table(full_pipeline)
allocation_backtest_table = make_metric_table(allocation_backtest)

display_cols = ["final_value", "cagr", "sharpe_ratio", "max_drawdown", "annualized_volatility"]
try:
    display(full_pipeline[display_cols].round(4))
except NameError:
    print(full_pipeline[display_cols].round(4))

# %% [markdown]
# ## Final Summary Charts

# %%
plt.style.use("seaborn-v0_8-whitegrid")

plot_methods = [m for m in METHOD_ORDER if m in full_pipeline.index]
method_names = [label_method(m) for m in plot_methods]

# 1) Main full-pipeline scorecard.
score_cols = ["cagr", "sharpe_ratio", "max_drawdown"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, title in zip(axes, score_cols, ["CAGR", "Sharpe Ratio", "Max Drawdown"]):
    values = full_pipeline.loc[plot_methods, col]
    colors = ["#4C78A8" if m != "benchmark_sp500" else "#F58518" for m in plot_methods]
    ax.bar(method_names, values.values, color=colors)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    if col == "max_drawdown":
        ax.axhline(0, color="black", linewidth=0.8)
fig.suptitle("Final Full-Pipeline Backtest Scorecard", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "final_full_pipeline_scorecard.png", dpi=160)
show_or_close(fig)

# 2) Allocation-only vs full-pipeline leakage impact.
comparison_rows = []
for method in METHOD_ORDER:
    if method in allocation_backtest.index and method in full_pipeline.index:
        comparison_rows.append(
            {
                "method": method,
                "Method": label_method(method),
                "allocation_only_cagr": allocation_backtest.loc[method, "cagr"],
                "full_pipeline_cagr": full_pipeline.loc[method, "cagr"],
                "cagr_gap": allocation_backtest.loc[method, "cagr"] - full_pipeline.loc[method, "cagr"],
                "allocation_only_sharpe": allocation_backtest.loc[method, "sharpe_ratio"],
                "full_pipeline_sharpe": full_pipeline.loc[method, "sharpe_ratio"],
            }
        )
leakage_comparison = pd.DataFrame(comparison_rows)
leakage_comparison.to_csv(OUTPUT_DIR / "final_04_vs_05_comparison.csv", index=False)

x = np.arange(len(leakage_comparison))
width = 0.38
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width / 2, leakage_comparison["allocation_only_cagr"], width, label="04 Allocation-only", color="#59A14F")
ax.bar(x + width / 2, leakage_comparison["full_pipeline_cagr"], width, label="05 Full-pipeline", color="#4C78A8")
ax.set_xticks(x)
ax.set_xticklabels(leakage_comparison["Method"], rotation=35, ha="right")
ax.set_ylabel("CAGR")
ax.set_title("Leakage Impact: Allocation-only vs Full-pipeline CAGR")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "final_04_vs_05_cagr_comparison.png", dpi=160)
show_or_close(fig)

# 3) Selection frequency top names.
top_selection = selection_frequency.head(20).copy()
top_selection.to_csv(OUTPUT_DIR / "final_top_selected_stocks.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 7))
top_plot = top_selection.sort_values("selected_count")
ax.barh(top_plot["ticker"], top_plot["selected_count"], color="#59A14F")
ax.set_xlabel("Times Selected Across 53 Rebalances")
ax.set_title("Most Frequently Selected Stocks in Full-Pipeline Backtest")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "final_top_selected_stocks.png", dpi=160)
show_or_close(fig)

# 4) Selection stability and turnover together.
selection_stability = selection_overlap["jaccard_vs_previous"].dropna()
turnover_by_method = full_pipeline.loc[[m for m in METHOD_ORDER if m in full_pipeline.index and m != "benchmark_sp500"], "average_turnover_per_rebalance"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(pd.to_datetime(selection_overlap["rebalance_date"]), selection_overlap["jaccard_vs_previous"], color="#4C78A8", linewidth=1.8)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Stock Selection Stability")
axes[0].set_ylabel("Jaccard Similarity vs Previous Month")
axes[1].bar([label_method(m) for m in turnover_by_method.index], turnover_by_method.values, color="#E15759")
axes[1].set_title("Average Turnover per Rebalance")
axes[1].tick_params(axis="x", rotation=35)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "final_selection_stability_and_turnover.png", dpi=160)
show_or_close(fig)

print("Saved final summary charts.")

# %% [markdown]
# ## Write Final Markdown Report

# %%
data_summary = pd.DataFrame(
    [
        {"Item": "S&P 500 universe rows", "Value": f"{len(sp500_universe):,}"},
        {"Item": "Usable return tickers", "Value": f"{returns_matrix.shape[1]:,}"},
        {"Item": "Return observations", "Value": f"{returns_matrix.shape[0]:,}"},
        {"Item": "Return date range", "Value": f"{returns_matrix.index.min().date()} to {returns_matrix.index.max().date()}"},
        {"Item": "Benchmark", "Value": "^GSPC"},
        {"Item": "Static selected stocks from Step 02", "Value": f"{len(selected_latest):,}"},
        {"Item": "Full-pipeline rebalance count", "Value": str(config.get("rebalance_count", ""))},
        {"Item": "Full-pipeline first holding date", "Value": str(config.get("first_holding_date", ""))},
        {"Item": "Full-pipeline last holding date", "Value": str(config.get("last_holding_date", ""))},
        {"Item": "Full-pipeline unique selected stocks", "Value": f"{selection_frequency.shape[0]:,}"},
        {"Item": "Average monthly selection overlap", "Value": num(selection_stability.mean(), 4)},
    ]
)

best_cagr_method = full_pipeline.drop(index=["benchmark_sp500"], errors="ignore")["cagr"].idxmax()
best_sharpe_method = full_pipeline.drop(index=["benchmark_sp500"], errors="ignore")["sharpe_ratio"].idxmax()
best_drawdown_method = full_pipeline.drop(index=["benchmark_sp500"], errors="ignore")["max_drawdown"].idxmax()

top_selected_table = top_selection[["ticker", "selected_count", "first_selected_date", "last_selected_date", "average_sharpe", "sector"]].head(15).copy()
top_selected_table["average_sharpe"] = top_selected_table["average_sharpe"].map(lambda x: num(x, 4))

latest_selected_table = selected_latest[["ticker", "cluster_id", "sector", "annual_return", "annual_volatility", "sharpe_ratio"]].copy()
latest_selected_table["annual_return"] = latest_selected_table["annual_return"].map(pct)
latest_selected_table["annual_volatility"] = latest_selected_table["annual_volatility"].map(pct)
latest_selected_table["sharpe_ratio"] = latest_selected_table["sharpe_ratio"].map(lambda x: num(x, 4))

leakage_table = leakage_comparison[["Method", "allocation_only_cagr", "full_pipeline_cagr", "cagr_gap"]].copy()
for col in ["allocation_only_cagr", "full_pipeline_cagr", "cagr_gap"]:
    leakage_table[col] = leakage_table[col].map(pct)
leakage_table = leakage_table.rename(
    columns={
        "allocation_only_cagr": "04 Allocation-only CAGR",
        "full_pipeline_cagr": "05 Full-pipeline CAGR",
        "cagr_gap": "CAGR Gap",
    }
)

report_lines = [
    "# Final Project Summary",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Executive Summary",
    "",
    "This project builds a stock-selection and portfolio-allocation workflow for S&P 500 stocks. "
    "The most reliable result is Step 05, because it re-selects stocks and re-allocates the portfolio at each rebalance date using only past data.",
    "",
    f"- Best full-pipeline CAGR: **{label_method(best_cagr_method)}** at **{pct(full_pipeline.loc[best_cagr_method, 'cagr'])}**.",
    f"- Best full-pipeline Sharpe: **{label_method(best_sharpe_method)}** at **{num(full_pipeline.loc[best_sharpe_method, 'sharpe_ratio'], 4)}**.",
    f"- Shallowest full-pipeline max drawdown: **{label_method(best_drawdown_method)}** at **{pct(full_pipeline.loc[best_drawdown_method, 'max_drawdown'])}**.",
    "",
    "Important limitation: Step 05 fixes the major look-ahead issue from using one fixed selected-stock list, "
    "but the universe still uses current S&P 500 constituents rather than point-in-time historical constituents.",
    "",
    "## Data Summary",
    "",
    markdown_table(data_summary),
    "",
    "## Workflow Summary",
    "",
    markdown_table(
        pd.DataFrame(
            [
                {"Step": "01", "File": "notebooks/01_prepare_sp500_data.ipynb", "Purpose": "Prepare S&P 500 universe, prices, returns, benchmark, and quality reports"},
                {"Step": "02", "File": "notebooks/02_select_stocks_clustering_mst.ipynb", "Purpose": "Cluster stocks and select one high-Sharpe stock per cluster using latest full history"},
                {"Step": "03", "File": "notebooks/03_allocate_portfolios.ipynb", "Purpose": "Create current portfolio allocations and efficient-frontier diagnostics"},
                {"Step": "04", "File": "notebooks/04_backtest_allocation_only.ipynb", "Purpose": "Allocation-only walk-forward test on the fixed 25 stocks from Step 02"},
                {"Step": "05", "File": "notebooks/05_backtest_full_pipeline_walkforward.ipynb", "Purpose": "Full-pipeline walk-forward test with stock selection rerun each rebalance"},
                {"Step": "06", "File": "notebooks/06_final_summary_report.ipynb", "Purpose": "Final project summary and comparison report"},
            ]
        )
    ),
    "",
    "## Step 05 Full-Pipeline Results",
    "",
    markdown_table(full_pipeline_table),
    "",
    "## Step 04 vs Step 05: Leakage Impact",
    "",
    "Step 04 used the fixed 25 stocks selected with full-history information. Step 05 reselects stocks every rebalance using only past data. "
    "The gap shows how much the fixed selected-stock list inflated the allocation-only backtest.",
    "",
    markdown_table(leakage_table),
    "",
    "## Most Frequently Selected Stocks in Step 05",
    "",
    markdown_table(top_selected_table),
    "",
    "## Latest Full-History Selected Stocks from Step 02",
    "",
    "These are useful for the current allocation view, but should not be treated as a leakage-free historical stock list.",
    "",
    markdown_table(latest_selected_table),
    "",
    "## Final Interpretation",
    "",
    "- Use **Step 05** as the main historical evaluation of the workflow.",
    "- **CVaR Bootstrap** has the best full-pipeline Sharpe and lowest drawdown among the tested models.",
    "- **Inverse Volatility** has the highest full-pipeline CAGR and is simpler than the optimizer-based methods.",
    "- **Markowitz** is sensitive to expected-return estimation and had the worst drawdown in the full-pipeline test.",
    "- **CVaR Monte Carlo** had lower volatility but did not beat the S&P 500 on CAGR in this run.",
    "",
    "## Limitations And Known Weaknesses",
    "",
    "### Data Limitations",
    "",
    "- The S&P 500 universe is based on the current Wikipedia constituent list, not point-in-time historical membership. "
    "This can create survivorship and constituent bias.",
    "- Yahoo Finance data can contain revisions, ticker mapping changes, missing fields, delisting gaps, or adjusted-price methodology changes.",
    "- The backtest period starts holding on 2022-01-03 and ends on 2026-05-01, so the test covers only about 4.3 years of realized holding history.",
    "- The 2026 calendar-year result is YTD only, not a full-year result.",
    "",
    "### Methodology Limitations",
    "",
    "- Step 04 is intentionally allocation-only and uses the fixed 25 stocks from Step 02; it should not be used as the main historical strategy result.",
    "- Step 05 fixes the fixed-stock look-ahead issue by reselecting each rebalance, but it still does not fix point-in-time S&P 500 membership bias.",
    "- Transaction cost is modeled as a simple proportional turnover cost of 0.001. It does not include bid-ask spread, market impact, tax, liquidity, borrow constraints, or execution slippage.",
    "- Rebalancing is monthly. Results may change materially with weekly, quarterly, or threshold-based rebalancing.",
    "- The strategy assumes fractional shares and immediate execution at return-series prices.",
    "",
    "### Model Limitations",
    "",
    "- Markowitz depends heavily on historical expected-return estimates, which are noisy and unstable for equities.",
    "- CVaR Bootstrap depends on historical sampled days and may miss unseen future regimes.",
    "- CVaR Monte Carlo assumes multivariate normal daily returns, which can understate fat tails and crash behavior.",
    "- Risk Parity and Inverse Volatility reduce risk exposure but do not directly forecast future returns.",
    "- The 10% max-weight cap is a project risk-control assumption, not a universal optimal setting.",
    "",
    "### Validation Limitations",
    "",
    "- Hyperparameters such as `N_CLUSTERS`, `MAX_WEIGHT`, Markowitz delta grid, CVaR tradeoff, and scenario count are not selected through nested walk-forward validation.",
    "- The same broad research period influenced model design choices, so there is still research/iteration overfitting risk.",
    "- Statistical significance is not tested. Differences such as 12.70% vs 12.63% CAGR should not be treated as conclusive without robustness checks.",
    "- No stress test by market regime, sector exposure, liquidity bucket, or alternative start date is included yet.",
    "",
    "### Interpretation Warnings",
    "",
    "- A positive Sharpe ratio means return exceeded the risk-free rate per unit volatility; it does not necessarily mean the model beat the S&P 500.",
    "- Best CAGR, best Sharpe, and lowest drawdown can point to different models. The final choice depends on the investor's objective.",
    "- Current results are research evidence, not a live trading recommendation.",
    "",
    "## Recommended Improvements",
    "",
    "1. Add point-in-time S&P 500 constituent history.",
    "2. Add nested walk-forward validation for `N_CLUSTERS`, `MAX_WEIGHT`, Markowitz delta, and CVaR tradeoff.",
    "3. Add transaction-cost sensitivity tests, such as 0.00%, 0.10%, 0.25%, and 0.50% per turnover.",
    "4. Add alternative rebalance frequencies: monthly, quarterly, and semiannual.",
    "5. Add robustness windows: start in 2021, 2022, 2023, and compare results.",
    "6. Add benchmark-relative metrics: alpha, tracking error, information ratio, beta, and excess CAGR.",
    "7. Add sector exposure and concentration diagnostics over time.",
    "",
    "## Key Output Charts",
    "",
    "- `outputs/final_full_pipeline_scorecard.png`",
    "- `outputs/final_04_vs_05_cagr_comparison.png`",
    "- `outputs/final_top_selected_stocks.png`",
    "- `outputs/final_selection_stability_and_turnover.png`",
    "- `outputs/full_pipeline_equity_curves.png`",
    "- `outputs/full_pipeline_relative_wealth_vs_benchmark.png`",
    "",
]

report_path = DOCS_DIR / "final_summary_report.md"
report_path.write_text("\n".join(report_lines), encoding="utf-8")

# Also save the key tables as CSV files for easy reuse.
data_summary.to_csv(OUTPUT_DIR / "final_data_summary.csv", index=False)
full_pipeline_table.to_csv(OUTPUT_DIR / "final_full_pipeline_metrics_table.csv", index=False)
allocation_backtest_table.to_csv(OUTPUT_DIR / "final_allocation_only_metrics_table.csv", index=False)
leakage_table.to_csv(OUTPUT_DIR / "final_04_vs_05_cagr_table.csv", index=False)

print("Saved final markdown report:", report_path)
print("Saved final summary tables and charts to:", OUTPUT_DIR)

# %% [markdown]
# ## Final Recommendation
#
# Use Step 05 as the main evidence for the project. Treat Step 04 as a diagnostic that explains how allocation
# methods behave after the stock list is already known, not as a leakage-free historical strategy result.
