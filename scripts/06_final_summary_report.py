# %% [markdown]
# # 6. Final Summary Report
#
# This final notebook summarizes the whole project from data preparation through full-pipeline
# walk-forward backtesting. It reads the real outputs from steps 01-05 and writes a compact
# final report plus summary charts.

# %% [markdown]
# ## Workflow
#
# 1. Set project paths and expected output files.
# 2. Load the Step 01-05 outputs needed for the final report.
# 3. Define method labels, formatting helpers, and report tables.
# 4. Create final comparison charts.
# 5. Build and write the final Markdown report.
# 6. Save final summary tables and verify the final report output manifest.

# %% [markdown]
# ## 1. Setup and Paths

# %%
from pathlib import Path
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
# ## 2. Load Final Inputs

# %% [markdown]
# ### 2.1 Required File Check

# %%
required_input_files = [
    DATA_DIR / "returns_matrix.parquet",
    DATA_DIR / "sp500_universe.csv",
    OUTPUT_DIR / "selected_stocks.csv",
    OUTPUT_DIR / "backtest_metrics.csv",
    OUTPUT_DIR / "full_pipeline_metrics.csv",
    OUTPUT_DIR / "full_pipeline_config.csv",
    OUTPUT_DIR / "full_pipeline_selection_frequency.csv",
    OUTPUT_DIR / "full_pipeline_selected_overlap.csv",
    OUTPUT_DIR / "full_pipeline_missing_holding_returns.csv",
]

missing_inputs = [path for path in required_input_files if not path.exists()]
if missing_inputs:
    raise FileNotFoundError("Missing required final-report inputs: " + ", ".join(str(path) for path in missing_inputs))

print(f"All {len(required_input_files)} required final-report inputs are available.")

# %% [markdown]
# ### 2.2 Load Data and Result Tables

# %%
returns_matrix = pd.read_parquet(DATA_DIR / "returns_matrix.parquet").sort_index()
sp500_universe = pd.read_csv(DATA_DIR / "sp500_universe.csv")

selected_latest = pd.read_csv(OUTPUT_DIR / "selected_stocks.csv")
allocation_backtest = pd.read_csv(OUTPUT_DIR / "backtest_metrics.csv").set_index("method")
full_pipeline = pd.read_csv(OUTPUT_DIR / "full_pipeline_metrics.csv").set_index("method")
full_pipeline_config = pd.read_csv(OUTPUT_DIR / "full_pipeline_config.csv")
selection_frequency = pd.read_csv(OUTPUT_DIR / "full_pipeline_selection_frequency.csv")
selection_overlap = pd.read_csv(OUTPUT_DIR / "full_pipeline_selected_overlap.csv")
missing_holding_returns = pd.read_csv(OUTPUT_DIR / "full_pipeline_missing_holding_returns.csv")

returns_matrix.index = pd.to_datetime(returns_matrix.index)
config = dict(zip(full_pipeline_config["setting"], full_pipeline_config["value"]))

print("Data window:", returns_matrix.index.min().date(), "to", returns_matrix.index.max().date())
print("Returns matrix:", returns_matrix.shape)
print("Latest selected stocks:", len(selected_latest))
print("Full-pipeline methods:", full_pipeline.shape[0])

# %% [markdown]
# ### 2.3 Method Labels and Ordering

# %%
METHOD_LABELS = {
    "equal": "Equal Weight",
    "inverse_volatility": "Inverse Volatility",
    "risk_parity": "Risk Parity",
    "markowitz_best_sharpe_default": "Markowitz-style Mean-Volatility Optimization",
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

# %% [markdown]
# ## 3. Helper Functions

# %% [markdown]
# ### 3.1 Formatting Helpers

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

# %% [markdown]
# ### 3.2 Metric Table Helper

# %%
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

# %% [markdown]
# ### 3.3 Preview Key Metrics

# %%
display_cols = ["final_value", "cagr", "sharpe_ratio", "max_drawdown", "annualized_volatility"]
try:
    display(full_pipeline[display_cols].round(4))
except NameError:
    print(full_pipeline[display_cols].round(4))

# %% [markdown]
# ## 4. Final Summary Charts

# %% [markdown]
# ### 4.1 Final Chart Setup

# %%
plt.style.use("seaborn-v0_8-whitegrid")

plot_methods = [m for m in METHOD_ORDER if m in full_pipeline.index]
def chart_label_method(method):
    return {"markowitz_best_sharpe_default": "Mean-Vol"}.get(method, label_method(method))


method_names = [chart_label_method(m) for m in plot_methods]

# %% [markdown]
# ### 4.2 Final Full-Pipeline Scorecard

# %%
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

# %% [markdown]
# ### 4.3 Allocation-Only vs Full-Pipeline Comparison

# %%
comparison_rows = []
for method in METHOD_ORDER:
    if method in allocation_backtest.index and method in full_pipeline.index:
        comparison_rows.append(
            {
                "method": method,
                "Method": label_method(method),
                "Chart Label": chart_label_method(method),
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
ax.set_xticklabels(leakage_comparison["Chart Label"], rotation=35, ha="right")
ax.set_ylabel("CAGR")
ax.set_title("Leakage Impact: Allocation-only vs Full-pipeline CAGR")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "final_04_vs_05_cagr_comparison.png", dpi=160)
show_or_close(fig)

# %% [markdown]
# ### 4.4 Top Selected Stocks

# %%
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

# %% [markdown]
# ### 4.5 Selection Stability and Turnover

# %%
selection_stability = selection_overlap["jaccard_vs_previous"].dropna()
turnover_by_method = full_pipeline.loc[[m for m in METHOD_ORDER if m in full_pipeline.index and m != "benchmark_sp500"], "average_turnover_per_rebalance"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(pd.to_datetime(selection_overlap["rebalance_date"]), selection_overlap["jaccard_vs_previous"], color="#4C78A8", linewidth=1.8)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Stock Selection Stability")
axes[0].set_ylabel("Jaccard Similarity vs Previous Month")
axes[1].bar([chart_label_method(m) for m in turnover_by_method.index], turnover_by_method.values, color="#E15759")
axes[1].set_title("Average Turnover per Rebalance")
axes[1].tick_params(axis="x", rotation=35)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "final_selection_stability_and_turnover.png", dpi=160)
show_or_close(fig)

print("Saved final summary charts.")

# %% [markdown]
# ## 5. Build Final Markdown Report

# %% [markdown]
# ### 5.1 Build Report Tables

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
        {"Item": "Missing holding-return audit rows", "Value": f"{len(missing_holding_returns):,}"},
    ]
)

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

# %% [markdown]
# ### 5.2 Identify Headline Methods

# %%
strategy_metrics = full_pipeline.drop(index=["benchmark_sp500"], errors="ignore")
best_cagr_method = strategy_metrics["cagr"].idxmax()
best_sharpe_method = strategy_metrics["sharpe_ratio"].idxmax()
best_drawdown_method = strategy_metrics["max_drawdown"].idxmax()

headline_summary = pd.DataFrame(
    [
        {"Metric": "Best full-pipeline CAGR", "Method": label_method(best_cagr_method), "Value": pct(full_pipeline.loc[best_cagr_method, "cagr"])},
        {"Metric": "Best full-pipeline Sharpe", "Method": label_method(best_sharpe_method), "Value": num(full_pipeline.loc[best_sharpe_method, "sharpe_ratio"], 4)},
        {"Metric": "Shallowest full-pipeline max drawdown", "Method": label_method(best_drawdown_method), "Value": pct(full_pipeline.loc[best_drawdown_method, "max_drawdown"])},
    ]
)

try:
    display(headline_summary)
except NameError:
    print(headline_summary)

# %% [markdown]
# ### 5.3 Build Executive and Results Sections

# %%
workflow_summary_table = pd.DataFrame(
    [
        {"Step": "01", "File": "notebooks/01_prepare_sp500_data.ipynb", "Purpose": "Prepare S&P 500 universe, prices, returns, benchmark, and quality reports"},
        {"Step": "02", "File": "notebooks/02_select_stocks_clustering_mst.ipynb", "Purpose": "Cluster stocks and select one stock per cluster using historical Sharpe as a backward-looking ranking heuristic"},
        {"Step": "03", "File": "notebooks/03_allocate_portfolios.ipynb", "Purpose": "Create current portfolio allocations using Equal Weight, Inverse Volatility, Markowitz-style Mean-Volatility Optimization, Risk Parity, CVaR Bootstrap, and CVaR Monte Carlo"},
        {"Step": "04", "File": "notebooks/04_backtest_allocation_only.ipynb", "Purpose": "Allocation-only walk-forward test using the fixed Step 02 selected-stock list"},
        {"Step": "05", "File": "notebooks/05_backtest_full_pipeline_walkforward.ipynb", "Purpose": "Main full-pipeline walk-forward simulation with stock selection and allocation rerun each rebalance"},
        {"Step": "06", "File": "notebooks/06_final_summary_report.ipynb", "Purpose": "Final project summary and comparison report"},
    ]
)

executive_and_results_lines = [
    "# Final Project Summary",
    "",
    "## Executive Summary",
    "",
    "This project builds a stock-selection and portfolio-allocation workflow for S&P 500 stocks. "
    "The most reliable result is Step 05, because it re-selects stocks and re-allocates the portfolio at each rebalance date using only past data.",
    "",
    "The results should be interpreted as a current-constituent walk-forward simulation, not as a fully point-in-time historical S&P 500 backtest.",
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
    markdown_table(workflow_summary_table),
    "",
    "## Step 05 Full-Pipeline Results",
    "",
    markdown_table(full_pipeline_table),
    "",
    "## Step 05 Audit Trail",
    "",
    "Step 05 records each rebalance's selected stocks in `outputs/full_pipeline_selected_stocks_history.csv`, including the training window (`train_start_date` to `train_end_date`) and `selection_mode = walk_forward_past_data_only`.",
    "",
    f"The missing holding-return audit is saved in `outputs/full_pipeline_missing_holding_returns.csv`; the latest run has {len(missing_holding_returns):,} missing-return audit rows.",
    "",
    "These outputs support the claim that Step 05 reselects stocks using only past data at each rebalance, although the universe still uses current S&P 500 constituents rather than point-in-time historical membership.",
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
    "The static Step 02 selected-stock list is useful for the current allocation view, but it should not be treated as a leakage-free historical stock list because it uses full-history information.",
    "",
    "The full table is available in `outputs/selected_stocks.csv`; Step 05 should be used for the main historical evaluation.",
    "",

]

# %% [markdown]
# ### 5.4 Build Interpretation and Limitation Sections

# %%
interpretation_and_limitations_lines = [
    "## Final Interpretation",
    "",
    "- Use **Step 05** as the main historical evaluation of the workflow.",
    "- **CVaR Bootstrap** has the best full-pipeline Sharpe and lowest drawdown among the tested models.",
    "- **Inverse Volatility** has the highest full-pipeline CAGR and is simpler than the optimizer-based methods.",
    "- **Markowitz-style Mean-Volatility Optimization** is sensitive to expected-return estimation and had the worst drawdown in the full-pipeline test.",
    "- **CVaR Monte Carlo** had lower volatility but did not beat the `^GSPC` S&P 500 benchmark on CAGR in this current-constituent simulation.",
    "- In this current-constituent walk-forward simulation, some strategy variants outperformed the `^GSPC` benchmark over the tested period. This should not be interpreted as evidence of a fully point-in-time historical trading edge.",
    "",
    "## Limitations And Known Weaknesses",
    "",
    "- The S&P 500 universe uses the current Wikipedia constituent list rather than point-in-time historical membership, so results can contain survivorship/current-constituent bias.",
    "- Yahoo Finance data can contain missing values, ticker mapping changes, revisions, or adjusted-price methodology differences.",
    "- The benchmark uses `^GSPC`, a price index, while stock returns use adjusted close prices, so the benchmark comparison is not fully total-return equivalent.",
    "- Step 04 is allocation-only and uses the fixed Step 02 stock list selected with full-history data; it should not be treated as the main strategy backtest.",
    "- Step 05 reselects stocks and reallocates monthly using only past returns, but it still does not solve point-in-time S&P 500 membership bias.",
    "- Historical Sharpe ranking is backward-looking and should not be interpreted as a direct forecast of future winners.",
    "- Markowitz-style optimization relies on noisy historical expected-return estimates, and its in-sample parameter selection can overfit without nested walk-forward validation.",
    "- CVaR Bootstrap depends on historical sampled days, while CVaR Monte Carlo assumes multivariate normal returns that may understate fat tails and regime shifts.",
    "- Transaction cost is modeled as a simple proportional cost of 0.001 per turnover and excludes bid-ask spread, slippage, market impact, taxes, liquidity constraints, and execution frictions.",
    "- The strategy does not explicitly constrain sector exposure, and statistical significance or regime stress tests have not yet been added.",
    "- Results are research evidence for this current-constituent simulation, not live investment advice.",
    "",

]

# %% [markdown]
# ### 5.5 Build Improvement and Output Sections

# %%
improvement_and_output_lines = [
    "## Recommended Improvements",
    "",
    "1. Add point-in-time S&P 500 constituent history.",
    "2. Add nested walk-forward validation for `N_CLUSTERS`, `MAX_WEIGHT`, Markowitz-style delta, and CVaR tradeoff.",
    "3. Add transaction-cost sensitivity tests, such as 0.00%, 0.10%, 0.25%, and 0.50% per turnover.",
    "4. Add alternative rebalance frequencies: monthly, quarterly, and semiannual.",
    "5. Add robustness windows: start in 2021, 2022, 2023, and compare results.",
    "6. Add benchmark-relative metrics: alpha, tracking error, information ratio, beta, and excess CAGR.",
    "7. Compare portfolio sector exposures against the S&P 500 to determine whether performance comes from stock selection, diversification, or unintended sector tilts.",
    "8. Test cluster-count sensitivity across multiple cluster counts such as 15, 20, 25, and 30.",
    "9. Test alternative correlation-distance definitions, including `sqrt(2*(1-rho))`.",
    "",
    "## Key Output Files",
    "",
    "- `outputs/final_full_pipeline_metrics_table.csv`",
    "- `outputs/final_04_vs_05_cagr_table.csv`",
    "- `outputs/full_pipeline_selected_stocks_history.csv`",
    "- `outputs/full_pipeline_missing_holding_returns.csv`",
    "- `outputs/full_pipeline_config.csv`",
    "- `outputs/final_full_pipeline_scorecard.png`",
    "- `outputs/final_04_vs_05_cagr_comparison.png`",
    "- `outputs/final_top_selected_stocks.png`",
    "- `outputs/final_selection_stability_and_turnover.png`",
    "- `outputs/full_pipeline_equity_curves.png`",
    "- `outputs/full_pipeline_relative_wealth_vs_benchmark.png`",
    "",
]

report_lines = executive_and_results_lines + interpretation_and_limitations_lines + improvement_and_output_lines

# %% [markdown]
# ### 5.6 Write Report and Summary Tables

# %%
report_path = DOCS_DIR / "final_summary_report.md"
report_path.write_text("\n".join(report_lines), encoding="utf-8")

# Also save the key tables as CSV files for easy reuse.
final_summary_table_files = [
    "final_data_summary.csv",
    "final_full_pipeline_metrics_table.csv",
    "final_allocation_only_metrics_table.csv",
    "final_04_vs_05_cagr_table.csv",
]

data_summary.to_csv(OUTPUT_DIR / "final_data_summary.csv", index=False)
full_pipeline_table.to_csv(OUTPUT_DIR / "final_full_pipeline_metrics_table.csv", index=False)
allocation_backtest_table.to_csv(OUTPUT_DIR / "final_allocation_only_metrics_table.csv", index=False)
leakage_table.to_csv(OUTPUT_DIR / "final_04_vs_05_cagr_table.csv", index=False)

print("Saved final markdown report:", report_path)
print("Saved final summary tables and charts to:", OUTPUT_DIR)

# %% [markdown]
# ## 6. Output Manifest

# %%
final_chart_output_files = [
    "final_full_pipeline_scorecard.png",
    "final_04_vs_05_comparison.csv",
    "final_04_vs_05_cagr_comparison.png",
    "final_top_selected_stocks.csv",
    "final_top_selected_stocks.png",
    "final_selection_stability_and_turnover.png",
]

final_report_output_files = [
    report_path.relative_to(PROJECT_ROOT).as_posix(),
    *[f"outputs/{file}" for file in final_summary_table_files],
    *[f"outputs/{file}" for file in final_chart_output_files],
]

final_output_manifest = pd.DataFrame(
    {
        "file": final_report_output_files,
        "exists": [(PROJECT_ROOT / file).exists() for file in final_report_output_files],
        "size_bytes": [
            (PROJECT_ROOT / file).stat().st_size if (PROJECT_ROOT / file).exists() else np.nan
            for file in final_report_output_files
        ],
    }
)
final_output_manifest.to_csv(OUTPUT_DIR / "final_output_manifest.csv", index=False)

try:
    display(final_output_manifest)
except NameError:
    print(final_output_manifest)

missing_final_outputs = final_output_manifest.loc[~final_output_manifest["exists"], "file"].tolist()
if missing_final_outputs:
    print("Missing expected final outputs:", missing_final_outputs)
else:
    print(f"All {len(final_output_manifest)} expected final outputs are present.")

# %% [markdown]
# ## 7. Interpretation Note
#
# Use Step 05 as the main evidence for the project. Treat Step 04 as a diagnostic that explains how allocation
# methods behave after the stock list is already known, not as a leakage-free historical strategy result.
