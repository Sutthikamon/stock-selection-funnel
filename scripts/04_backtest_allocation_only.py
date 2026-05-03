# %% [markdown]
# # 04 Backtest / Evaluation
#
# This notebook-style script tests the portfolio allocation models out of sample.
#
# Important scope:
# - This is an allocation-only backtest.
# - It uses the 25 selected stocks from step 02 as a fixed universe.
# - At every rebalance date, weights are computed only from past returns available at that date.
# - Realized performance is measured only with actual historical returns after the rebalance date.
# - Bootstrap and Monte Carlo scenarios are used only inside the CVaR optimizers, not to simulate performance.

# %%
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog


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
OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Backtest configuration
# ----------------------------------------------------------------------
MIN_TRAINING_DAYS = 756       # about 3 trading years before the first out-of-sample holding period
REBALANCE_FREQUENCY = "month_end"
TRAIN_WINDOW = "expanding"    # use all past data available at each rebalance date

MAX_WEIGHT = 0.10             # project concentration cap per stock
MIN_WEIGHT = 0.00             # long-only
TRANSACTION_COST = 0.001      # same convention as the source repo: 0.10% per turnover
COUNT_INITIAL_COST = True

MARKOWITZ_DELTA_GRID = np.logspace(start=-1, stop=1.5, num=10)[::-1].tolist()

N_SCENARIOS = 3000
CVAR_ALPHA = 0.95
CVAR_RETURN_TRADEOFF = 1.0
RANDOM_SEED = 42

METHOD_ORDER = [
    "equal",
    "inverse_volatility",
    "risk_parity",
    "markowitz_best_sharpe_default",
    "cvar_bootstrap",
    "cvar_montecarlo",
]

BENCHMARK_NAME = "benchmark_sp500"

print(f"Project root: {PROJECT_ROOT}")
print(f"Outputs will be written to: {OUTPUT_DIR}")

# %% [markdown]
# ## Load Real Project Data
#
# The backtest uses existing real files from steps 01 and 02:
#
# - `data/returns_matrix.parquet`
# - `data/benchmark_returns.parquet`
# - `outputs/selected_stocks.csv`
#
# The selected 25 stocks are fixed in this file so that the test focuses on allocation models.

# %%
def estimate_trading_days_per_year(index: pd.Index) -> int:
    dates = pd.to_datetime(index)
    counts = pd.Series(1, index=dates).groupby(dates.year).sum()
    full_years = counts[(counts.index > dates.min().year) & (counts.index < dates.max().year)]
    base = full_years if len(full_years) else counts
    return int(round(base.median()))


selected = pd.read_csv(OUTPUT_DIR / "selected_stocks.csv")
returns_all = pd.read_parquet(DATA_DIR / "returns_matrix.parquet").sort_index()
benchmark = pd.read_parquet(DATA_DIR / "benchmark_returns.parquet").sort_index()

if "ticker" not in selected.columns:
    raise ValueError("selected_stocks.csv must contain a 'ticker' column.")

selected_tickers = selected["ticker"].astype(str).tolist()
missing = sorted(set(selected_tickers) - set(returns_all.columns))
if missing:
    raise ValueError(f"Selected tickers missing from returns matrix: {missing}")

returns_selected = returns_all[selected_tickers].copy()
returns_selected.index = pd.to_datetime(returns_selected.index)
benchmark.index = pd.to_datetime(benchmark.index)

benchmark_col = "benchmark_return" if "benchmark_return" in benchmark.columns else benchmark.columns[0]
benchmark_returns = benchmark[benchmark_col].copy()

common_index = returns_selected.dropna(how="any").index.intersection(benchmark_returns.dropna().index)
returns_selected = returns_selected.loc[common_index].sort_index()
benchmark_returns = benchmark_returns.loc[common_index].sort_index()

TRADING_DAYS = estimate_trading_days_per_year(returns_selected.index)

print("Selected tickers:", len(selected_tickers))
print("Return rows:", len(returns_selected))
print("Return window:", returns_selected.index.min().date(), "to", returns_selected.index.max().date())
print("Estimated trading days/year:", TRADING_DAYS)

if len(returns_selected) <= MIN_TRAINING_DAYS:
    raise ValueError("Not enough data to create an out-of-sample backtest after the minimum training window.")

# %% [markdown]
# ## Risk-Free Rate For Metrics
#
# The script tries to fetch historical FRED `DGS3MO` rates. If that is unavailable, it falls back to the
# real risk-free rate already saved by step 03 in `outputs/risk_free_rate.csv`.

# %%
def fetch_fred_dgs3mo_history(start_date, end_date):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
    fred = pd.read_csv(url)
    date_col = "DATE" if "DATE" in fred.columns else "observation_date"
    value_col = "DGS3MO" if "DGS3MO" in fred.columns else fred.columns[-1]
    fred = fred.rename(columns={date_col: "date", value_col: "annual_rate_percent"})
    fred["date"] = pd.to_datetime(fred["date"])
    fred["annual_rate_decimal"] = pd.to_numeric(fred["annual_rate_percent"], errors="coerce") / 100.0
    fred = fred.dropna(subset=["annual_rate_decimal"])
    fred = fred[(fred["date"] <= pd.Timestamp(end_date)) & (fred["date"] >= pd.Timestamp(start_date) - pd.Timedelta(days=10))]
    if fred.empty:
        raise ValueError("No FRED DGS3MO observations available for the backtest range.")
    return fred[["date", "annual_rate_decimal"]].drop_duplicates("date").set_index("date").sort_index()


def load_saved_risk_free_rate():
    path = OUTPUT_DIR / "risk_free_rate.csv"
    if not path.exists():
        raise FileNotFoundError("outputs/risk_free_rate.csv does not exist.")
    saved = pd.read_csv(path)
    if "annual_rate_decimal" not in saved.columns:
        raise ValueError("risk_free_rate.csv must contain annual_rate_decimal.")
    row = saved.iloc[0]
    return {
        "annual_rate_decimal": float(row["annual_rate_decimal"]),
        "daily_rate_decimal": float(row.get("daily_rate_decimal", (1 + row["annual_rate_decimal"]) ** (1 / TRADING_DAYS) - 1)),
        "source": str(row.get("source", "saved")),
        "series_id": str(row.get("series_id", "unknown")),
        "observation_date": str(row.get("observation_date", "unknown")),
    }


def build_risk_free_series(index: pd.Index):
    index = pd.DatetimeIndex(index)
    try:
        annual = fetch_fred_dgs3mo_history(index.min(), index.max())
        annual = annual.reindex(index, method="ffill").bfill()
        daily = (1.0 + annual["annual_rate_decimal"]) ** (1.0 / TRADING_DAYS) - 1.0
        out = pd.DataFrame(
            {
                "annual_rate_decimal": annual["annual_rate_decimal"],
                "daily_rate_decimal": daily,
                "source": "FRED",
                "series_id": "DGS3MO",
            },
            index=index,
        )
        out.index.name = "date"
        return out
    except Exception as error:
        warnings.warn(
            "Could not fetch historical FRED DGS3MO rates. "
            f"Falling back to saved step-03 risk-free rate. Error: {error}"
        )
        saved = load_saved_risk_free_rate()
        out = pd.DataFrame(
            {
                "annual_rate_decimal": saved["annual_rate_decimal"],
                "daily_rate_decimal": saved["daily_rate_decimal"],
                "source": f"saved_{saved['source']}",
                "series_id": saved["series_id"],
                "observation_date": saved["observation_date"],
            },
            index=index,
        )
        out.index.name = "date"
        return out


risk_free = build_risk_free_series(returns_selected.index)
risk_free.to_csv(OUTPUT_DIR / "backtest_risk_free_rates.csv")

print("Risk-free source used:", risk_free["source"].iloc[0], risk_free["series_id"].iloc[0])
print("Risk-free annual range:", risk_free["annual_rate_decimal"].min(), "to", risk_free["annual_rate_decimal"].max())

# %% [markdown]
# ## Allocation Helpers

# %%
def annualized_mean(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * TRADING_DAYS


def annualized_cov(returns: pd.DataFrame) -> pd.DataFrame:
    cov = returns.cov() * TRADING_DAYS
    return cov + np.eye(cov.shape[0]) * 1e-10


def cap_and_redistribute(weights, max_weight=MAX_WEIGHT, min_weight=MIN_WEIGHT, tol=1e-12):
    w = np.asarray(weights, dtype=float).copy()
    n = len(w)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, min_weight)

    if w.sum() <= tol:
        w = np.ones(n) / n
    else:
        w = w / w.sum()

    if max_weight is None:
        return w

    if n * max_weight < 1 - 1e-10:
        raise ValueError("MAX_WEIGHT is too low to allocate 100% across the selected stocks.")

    for _ in range(100):
        over = w > max_weight + tol
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        under = w < max_weight - tol
        if not under.any():
            break
        capacity = max_weight - w[under]
        if capacity.sum() <= tol:
            break
        w[under] += np.minimum(capacity, excess * capacity / capacity.sum())

    return w / w.sum()


def portfolio_training_metrics(weights, returns: pd.DataFrame, annual_risk_free: float, alpha=CVAR_ALPHA):
    w = pd.Series(weights, index=returns.columns)
    port_ret = returns @ w
    ann_ret = port_ret.mean() * TRADING_DAYS
    ann_vol = port_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = np.nan if ann_vol == 0 else (ann_ret - annual_risk_free) / ann_vol
    var = -port_ret.quantile(1 - alpha)
    tail = port_ret[port_ret <= port_ret.quantile(1 - alpha)]
    cvar = np.nan if tail.empty else -tail.mean()
    return {
        "annual_return": float(ann_ret),
        "annual_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "historical_var_95_daily": float(var),
        "historical_cvar_95_daily": float(cvar),
        "max_weight": float(w.max()),
        "min_weight": float(w.min()),
        "effective_number_of_holdings": float(1 / np.sum(np.square(w.values))),
    }


def delta_label(value):
    return f"d{value:.4g}".replace(".", "_")

# %% [markdown]
# ## Allocation Models

# %%
def allocate_equal_weight(returns: pd.DataFrame):
    n = returns.shape[1]
    return np.ones(n) / n


def allocate_inverse_volatility(returns: pd.DataFrame):
    vol = returns.std(ddof=1).replace(0, np.nan)
    inv_vol = 1 / vol
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0)
    return cap_and_redistribute(inv_vol.to_numpy())


def allocate_markowitz(returns: pd.DataFrame, delta: float):
    mu = annualized_mean(returns).to_numpy()
    cov = annualized_cov(returns).to_numpy()
    n = len(mu)

    def objective(w):
        portfolio_vol = np.sqrt(max(w @ cov @ w, 0.0))
        return -(mu @ w - delta * portfolio_vol)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        warnings.warn(f"Markowitz optimizer failed: {result.message}. Falling back to equal weight.")
        return allocate_equal_weight(returns)
    return cap_and_redistribute(result.x)


def allocate_risk_parity(returns: pd.DataFrame):
    cov = annualized_cov(returns).to_numpy()
    n = cov.shape[0]
    target_budget = np.ones(n) / n

    def risk_contributions(w):
        portfolio_var = w @ cov @ w
        if portfolio_var <= 0:
            return np.ones(n) / n
        marginal_risk = cov @ w
        return w * marginal_risk / portfolio_var

    def objective(w):
        rc = risk_contributions(w)
        return np.sum((rc - target_budget) ** 2)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in range(n)]
    x0 = allocate_inverse_volatility(returns)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        warnings.warn(f"Risk Parity optimizer failed: {result.message}. Falling back to inverse volatility.")
        return allocate_inverse_volatility(returns)
    return cap_and_redistribute(result.x)


def generate_bootstrap_scenarios(returns: pd.DataFrame, n_scenarios=N_SCENARIOS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    idx = rng.integers(0, len(values), size=n_scenarios)
    return values[idx]


def generate_monte_carlo_scenarios(returns: pd.DataFrame, n_scenarios=N_SCENARIOS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy() + np.eye(returns.shape[1]) * 1e-10
    return rng.multivariate_normal(mean=mu, cov=cov, size=n_scenarios)


def allocate_cvar(scenarios: np.ndarray, fallback_returns: pd.DataFrame, alpha=CVAR_ALPHA, return_tradeoff=CVAR_RETURN_TRADEOFF):
    scenarios = np.asarray(scenarios, dtype=float)
    n_scenarios, n_assets = scenarios.shape
    scenario_mean = scenarios.mean(axis=0)

    n_vars = n_assets + 1 + n_scenarios
    t_idx = n_assets
    u_start = n_assets + 1

    c = np.zeros(n_vars)
    c[:n_assets] = -return_tradeoff * scenario_mean
    c[t_idx] = 1.0
    c[u_start:] = 1.0 / ((1.0 - alpha) * n_scenarios)

    a_ub = np.zeros((n_scenarios, n_vars))
    a_ub[:, :n_assets] = -scenarios
    a_ub[:, t_idx] = -1.0
    a_ub[np.arange(n_scenarios), u_start + np.arange(n_scenarios)] = -1.0
    b_ub = np.zeros(n_scenarios)

    a_eq = np.zeros((1, n_vars))
    a_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])

    bounds = (
        [(MIN_WEIGHT, MAX_WEIGHT) for _ in range(n_assets)]
        + [(None, None)]
        + [(0.0, None) for _ in range(n_scenarios)]
    )

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        warnings.warn(f"CVaR optimizer failed: {result.message}. Falling back to inverse volatility.")
        return allocate_inverse_volatility(fallback_returns)

    return cap_and_redistribute(result.x[:n_assets])

# %% [markdown]
# ## Backtest Engine

# %%
def make_month_end_rebalance_dates(index: pd.Index, min_training_days: int):
    dates = pd.DatetimeIndex(index).sort_values()
    by_month = pd.DataFrame(index=dates)
    rebalance_dates = by_month.groupby([by_month.index.year, by_month.index.month]).tail(1).index
    positions = pd.Series(np.arange(len(dates)), index=dates)
    rebalance_dates = pd.DatetimeIndex([d for d in rebalance_dates if positions.loc[d] >= min_training_days - 1])
    if len(rebalance_dates) < 2:
        raise ValueError("Not enough rebalance dates after applying the minimum training window.")
    return rebalance_dates


def compute_rebalance_weights(train_returns: pd.DataFrame, rebalance_date: pd.Timestamp, period_number: int):
    annual_rf = float(risk_free.loc[rebalance_date, "annual_rate_decimal"])

    weights = {
        "equal": allocate_equal_weight(train_returns),
        "inverse_volatility": allocate_inverse_volatility(train_returns),
        "risk_parity": allocate_risk_parity(train_returns),
    }

    frontier_rows = []
    frontier_weights = {}
    for delta in MARKOWITZ_DELTA_GRID:
        method = f"markowitz_{delta_label(delta)}"
        w = allocate_markowitz(train_returns, delta=delta)
        metrics = portfolio_training_metrics(w, train_returns, annual_rf)
        metrics.update(
            {
                "rebalance_date": rebalance_date,
                "method": method,
                "delta": delta,
                "penalty": "annual_volatility",
                "weight_sum": float(np.sum(w)),
            }
        )
        frontier_rows.append(metrics)
        frontier_weights[method] = w

    frontier = pd.DataFrame(frontier_rows)
    best_method = frontier.sort_values("sharpe_ratio", ascending=False).iloc[0]["method"]
    weights["markowitz_best_sharpe_default"] = frontier_weights[best_method]

    bootstrap_scenarios = generate_bootstrap_scenarios(
        train_returns,
        n_scenarios=N_SCENARIOS,
        seed=RANDOM_SEED + period_number,
    )
    monte_carlo_scenarios = generate_monte_carlo_scenarios(
        train_returns,
        n_scenarios=N_SCENARIOS,
        seed=RANDOM_SEED + 10_000 + period_number,
    )

    weights["cvar_bootstrap"] = allocate_cvar(
        bootstrap_scenarios,
        fallback_returns=train_returns,
        return_tradeoff=CVAR_RETURN_TRADEOFF,
    )
    weights["cvar_montecarlo"] = allocate_cvar(
        monte_carlo_scenarios,
        fallback_returns=train_returns,
        return_tradeoff=CVAR_RETURN_TRADEOFF,
    )

    for method, w in weights.items():
        weights[method] = cap_and_redistribute(w)

    return weights, frontier


def update_drifted_weights(weights: np.ndarray, asset_returns: np.ndarray):
    gross_return = float(weights @ asset_returns)
    denom = 1.0 + gross_return
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(weights) / len(weights), gross_return
    new_weights = weights * (1.0 + asset_returns) / denom
    new_weights = np.where(np.isfinite(new_weights), new_weights, 0.0)
    if new_weights.sum() <= 0:
        new_weights = np.ones_like(weights) / len(weights)
    else:
        new_weights = new_weights / new_weights.sum()
    return new_weights, gross_return


def run_backtest():
    rebalance_dates = make_month_end_rebalance_dates(returns_selected.index, MIN_TRAINING_DAYS)
    first_rebalance_date = rebalance_dates[0]
    first_holding_date = returns_selected.index[returns_selected.index.get_loc(first_rebalance_date) + 1]

    current_weights = {method: np.zeros(len(selected_tickers)) for method in METHOD_ORDER}
    equity = {method: 1.0 for method in METHOD_ORDER}
    benchmark_equity = 1.0

    daily_return_records = []
    equity_records = [{"date": first_rebalance_date, **{method: 1.0 for method in METHOD_ORDER}, BENCHMARK_NAME: 1.0}]
    turnover_records = []
    weight_records = []
    frontier_records = []

    for period_number, rebalance_date in enumerate(rebalance_dates[:-1], start=1):
        next_rebalance_date = rebalance_dates[period_number]
        train_returns = returns_selected.loc[:rebalance_date].dropna(how="any")
        if len(train_returns) < MIN_TRAINING_DAYS:
            continue

        target_weights, frontier = compute_rebalance_weights(train_returns, rebalance_date, period_number)
        frontier_records.append(frontier)

        holding_returns = returns_selected.loc[
            (returns_selected.index > rebalance_date) & (returns_selected.index <= next_rebalance_date)
        ]
        if holding_returns.empty:
            continue

        pending_cost = {}
        for method in METHOD_ORDER:
            target = target_weights[method]
            previous = current_weights[method]
            if not COUNT_INITIAL_COST and np.isclose(previous.sum(), 0.0):
                turnover = 0.0
            else:
                turnover = float(np.abs(target - previous).sum())
            cost_rate = turnover * TRANSACTION_COST
            pending_cost[method] = cost_rate

            turnover_records.append(
                {
                    "rebalance_date": rebalance_date,
                    "method": method,
                    "turnover": turnover,
                    "transaction_cost_rate": cost_rate,
                    "equity_before_cost": equity[method],
                    "transaction_cost_value": equity[method] * cost_rate,
                    "train_start_date": train_returns.index.min(),
                    "train_end_date": train_returns.index.max(),
                    "train_observations": len(train_returns),
                    "holding_start_date": holding_returns.index.min(),
                    "holding_end_date": holding_returns.index.max(),
                    "holding_observations": len(holding_returns),
                }
            )

            for ticker, weight in zip(selected_tickers, target):
                weight_records.append(
                    {
                        "rebalance_date": rebalance_date,
                        "method": method,
                        "ticker": ticker,
                        "weight": float(weight),
                    }
                )

            current_weights[method] = target.copy()

        for day_number, (date, asset_return_row) in enumerate(holding_returns.iterrows()):
            asset_returns = asset_return_row.to_numpy(dtype=float)
            return_record = {"date": date}
            equity_record = {"date": date}

            for method in METHOD_ORDER:
                starting_equity = equity[method]
                cost_rate = pending_cost[method] if day_number == 0 else 0.0
                if cost_rate:
                    equity[method] *= max(1.0 - cost_rate, 0.0)

                drifted, gross_return = update_drifted_weights(current_weights[method], asset_returns)
                equity[method] *= 1.0 + gross_return
                current_weights[method] = drifted

                net_return = equity[method] / starting_equity - 1.0
                return_record[method] = float(net_return)
                equity_record[method] = float(equity[method])

            benchmark_return = float(benchmark_returns.loc[date])
            benchmark_equity *= 1.0 + benchmark_return
            return_record[BENCHMARK_NAME] = benchmark_return
            equity_record[BENCHMARK_NAME] = float(benchmark_equity)

            daily_return_records.append(return_record)
            equity_records.append(equity_record)

        print(
            f"{period_number:02d}/{len(rebalance_dates)-1}: "
            f"rebalance {rebalance_date.date()} -> hold to {next_rebalance_date.date()}"
        )

    daily_returns = pd.DataFrame(daily_return_records).set_index("date").sort_index()
    equity_curves = pd.DataFrame(equity_records).set_index("date").sort_index()
    turnover = pd.DataFrame(turnover_records)
    weights_history = pd.DataFrame(weight_records)
    markowitz_frontier_history = pd.concat(frontier_records, ignore_index=True) if frontier_records else pd.DataFrame()

    metadata = {
        "first_rebalance_date": first_rebalance_date,
        "first_holding_date": first_holding_date,
        "last_holding_date": daily_returns.index.max(),
        "rebalance_count": len(turnover["rebalance_date"].drop_duplicates()) if not turnover.empty else 0,
    }
    return daily_returns, equity_curves, turnover, weights_history, markowitz_frontier_history, metadata


daily_returns, equity_curves, turnover, weights_history, markowitz_frontier_history, backtest_meta = run_backtest()

print("First rebalance date:", backtest_meta["first_rebalance_date"].date())
print("First holding date:", backtest_meta["first_holding_date"].date())
print("Last holding date:", backtest_meta["last_holding_date"].date())
print("Rebalance count:", backtest_meta["rebalance_count"])

# %% [markdown]
# ## Evaluate Backtest Performance

# %%
def max_drawdown(equity_curve: pd.Series):
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min(), drawdown


def summarize_performance(daily_returns: pd.DataFrame, equity_curves: pd.DataFrame, turnover: pd.DataFrame):
    rows = []
    rf_daily = risk_free["daily_rate_decimal"].reindex(daily_returns.index).ffill().bfill()

    turnover_summary = pd.DataFrame()
    if not turnover.empty:
        turnover_summary = turnover.groupby("method").agg(
            total_turnover=("turnover", "sum"),
            average_turnover_per_rebalance=("turnover", "mean"),
            total_transaction_cost_value=("transaction_cost_value", "sum"),
            total_transaction_cost_rate=("transaction_cost_rate", "sum"),
        )

    for method in list(METHOD_ORDER) + [BENCHMARK_NAME]:
        ret = daily_returns[method].dropna()
        eq = equity_curves[method].dropna()
        if ret.empty or eq.empty:
            continue

        first_return_date = ret.index.min()
        last_return_date = ret.index.max()
        years = max((last_return_date - first_return_date).days / 365.25, 1 / TRADING_DAYS)
        final_value = float(eq.iloc[-1])
        total_return = final_value - 1.0
        cagr = final_value ** (1.0 / years) - 1.0
        annual_return = ret.mean() * TRADING_DAYS
        annual_vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
        excess = ret - rf_daily.loc[ret.index]
        sharpe = np.nan if annual_vol == 0 else (excess.mean() * TRADING_DAYS) / annual_vol
        downside = ret[ret < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
        sortino = np.nan if downside == 0 or np.isnan(downside) else (excess.mean() * TRADING_DAYS) / downside
        mdd, _ = max_drawdown(eq)
        calmar = np.nan if mdd == 0 else cagr / abs(mdd)
        hit_rate = float((ret > 0).mean())

        row = {
            "method": method,
            "first_return_date": first_return_date,
            "last_return_date": last_return_date,
            "observations": len(ret),
            "years": years,
            "final_value": final_value,
            "total_return": total_return,
            "cagr": cagr,
            "annualized_arithmetic_return": annual_return,
            "annualized_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": float(mdd),
            "calmar_ratio": calmar,
            "positive_day_rate": hit_rate,
        }

        if method in turnover_summary.index:
            row.update(turnover_summary.loc[method].to_dict())
        else:
            row.update(
                {
                    "total_turnover": 0.0,
                    "average_turnover_per_rebalance": 0.0,
                    "total_transaction_cost_value": 0.0,
                    "total_transaction_cost_rate": 0.0,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows).set_index("method")


drawdowns = equity_curves.apply(lambda col: col / col.cummax() - 1.0)
metrics = summarize_performance(daily_returns, equity_curves, turnover)

metrics_display = metrics.copy()
numeric_metric_cols = metrics_display.select_dtypes(include=[np.number]).columns
metrics_display[numeric_metric_cols] = metrics_display[numeric_metric_cols].round(4)
try:
    display(metrics_display)
except NameError:
    print(metrics_display)

# %% [markdown]
# ## Save Outputs

# %%
daily_returns.to_csv(OUTPUT_DIR / "backtest_daily_returns.csv")
equity_curves.to_csv(OUTPUT_DIR / "backtest_equity_curves.csv")
drawdowns.to_csv(OUTPUT_DIR / "backtest_drawdowns.csv")
metrics.to_csv(OUTPUT_DIR / "backtest_metrics.csv")
turnover.to_csv(OUTPUT_DIR / "backtest_turnover.csv", index=False)
weights_history.to_csv(OUTPUT_DIR / "backtest_weights_history.csv", index=False)
markowitz_frontier_history.to_csv(OUTPUT_DIR / "backtest_markowitz_frontier_history.csv", index=False)

config_rows = [
    ("project_root", "."),
    ("universe_mode", "fixed_selected_stocks_from_step_02"),
    ("selected_stock_count", len(selected_tickers)),
    ("first_available_return_date", returns_selected.index.min().date().isoformat()),
    ("last_available_return_date", returns_selected.index.max().date().isoformat()),
    ("first_rebalance_date", backtest_meta["first_rebalance_date"].date().isoformat()),
    ("first_holding_date", backtest_meta["first_holding_date"].date().isoformat()),
    ("last_holding_date", backtest_meta["last_holding_date"].date().isoformat()),
    ("minimum_training_days", MIN_TRAINING_DAYS),
    ("rebalance_frequency", REBALANCE_FREQUENCY),
    ("train_window", TRAIN_WINDOW),
    ("trading_days_per_year", TRADING_DAYS),
    ("max_weight", MAX_WEIGHT),
    ("min_weight", MIN_WEIGHT),
    ("transaction_cost", TRANSACTION_COST),
    ("count_initial_cost", COUNT_INITIAL_COST),
    ("markowitz_delta_grid", json.dumps(MARKOWITZ_DELTA_GRID)),
    ("cvar_alpha", CVAR_ALPHA),
    ("cvar_return_tradeoff", CVAR_RETURN_TRADEOFF),
    ("cvar_scenarios_per_rebalance", N_SCENARIOS),
    ("risk_free_source", str(risk_free["source"].iloc[0])),
    ("risk_free_series_id", str(risk_free["series_id"].iloc[0])),
]
pd.DataFrame(config_rows, columns=["setting", "value"]).to_csv(OUTPUT_DIR / "backtest_config.csv", index=False)

print("Saved backtest outputs to:", OUTPUT_DIR)
print(metrics[["final_value", "cagr", "annualized_volatility", "sharpe_ratio", "max_drawdown"]].round(4))

# %% [markdown]
# ## Charts

# %%
plt.style.use("seaborn-v0_8-whitegrid")


def show_or_close(fig):
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


def short_method_name(method):
    return {
        "equal": "Equal",
        "inverse_volatility": "Inv Vol",
        "risk_parity": "Risk Parity",
        "markowitz_best_sharpe_default": "Markowitz",
        "cvar_bootstrap": "CVaR Boot",
        "cvar_montecarlo": "CVaR MC",
        BENCHMARK_NAME: "S&P 500",
    }.get(method, method)


fig, ax = plt.subplots(figsize=(12, 7))
for col in list(METHOD_ORDER) + [BENCHMARK_NAME]:
    ax.plot(equity_curves.index, equity_curves[col], label=col, linewidth=1.7)
ax.set_title("Backtest Equity Curves")
ax.set_ylabel("Growth of $1")
ax.legend(loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "backtest_equity_curves.png", dpi=160)
show_or_close(fig)

fig, ax = plt.subplots(figsize=(12, 7))
for col in list(METHOD_ORDER) + [BENCHMARK_NAME]:
    ax.plot(drawdowns.index, drawdowns[col], label=col, linewidth=1.4)
ax.set_title("Backtest Drawdowns")
ax.set_ylabel("Drawdown")
ax.legend(loc="lower left", ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "backtest_drawdowns.png", dpi=160)
show_or_close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
metrics.loc[METHOD_ORDER, "average_turnover_per_rebalance"].plot(kind="bar", ax=ax, color="#4C78A8")
ax.set_title("Average Turnover per Rebalance")
ax.set_ylabel("Turnover")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "backtest_turnover.png", dpi=160)
show_or_close(fig)


def plot_weights_over_time(method: str):
    method_weights = weights_history[weights_history["method"] == method]
    pivot = method_weights.pivot(index="rebalance_date", columns="ticker", values="weight").fillna(0.0)
    top = pivot.mean().sort_values(ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot[top].plot.area(ax=ax, linewidth=0)
    ax.set_title(f"Top Average Weights Over Time: {method}")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"backtest_weights_over_time_{method}.png"
    fig.savefig(out_path, dpi=160)
    show_or_close(fig)


for method in METHOD_ORDER:
    plot_weights_over_time(method)

print("Saved backtest charts.")

# %% [markdown]
# ## Additional Backtest Diagnostic Charts
#
# These charts help explain how each allocation behaves through time, not just the final score.

# %%
# 1) Realized risk-return scatter from the actual backtest.
plot_methods = list(METHOD_ORDER) + [BENCHMARK_NAME]
fig, ax = plt.subplots(figsize=(9, 6))
points = ax.scatter(
    metrics.loc[plot_methods, "annualized_volatility"],
    metrics.loc[plot_methods, "cagr"],
    c=metrics.loc[plot_methods, "sharpe_ratio"],
    s=170,
    cmap="viridis",
    edgecolor="black",
    linewidth=0.7,
)
for method in plot_methods:
    row = metrics.loc[method]
    ax.annotate(
        short_method_name(method),
        (row["annualized_volatility"], row["cagr"]),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=9,
    )
ax.set_title("Backtest Realized Risk vs CAGR")
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("CAGR")
fig.colorbar(points, ax=ax, label="Sharpe Ratio")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "backtest_risk_return_scatter.png", dpi=160)
show_or_close(fig)

# 2) Metric dashboard for quick comparison.
metric_specs = [
    ("final_value", "Final Value", False),
    ("cagr", "CAGR", False),
    ("sharpe_ratio", "Sharpe Ratio", False),
    ("max_drawdown", "Max Drawdown", True),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (col, title, draw_zero_line) in zip(axes.ravel(), metric_specs):
    values = metrics.loc[plot_methods, col]
    colors = ["#4C78A8" if method != BENCHMARK_NAME else "#F58518" for method in values.index]
    ax.bar([short_method_name(x) for x in values.index], values.values, color=colors)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    if draw_zero_line:
        ax.axhline(0.0, color="black", linewidth=0.8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "backtest_metric_dashboard.png", dpi=160)
show_or_close(fig)

# 3) Relative wealth versus benchmark. Above 1.0 means outperforming S&P 500 since the start.
relative_wealth = equity_curves[METHOD_ORDER].div(equity_curves[BENCHMARK_NAME], axis=0)
relative_wealth.to_csv(OUTPUT_DIR / "backtest_relative_wealth_vs_benchmark.csv")

fig, ax = plt.subplots(figsize=(12, 7))
for method in METHOD_ORDER:
    ax.plot(relative_wealth.index, relative_wealth[method], label=short_method_name(method), linewidth=1.7)
ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
ax.set_title("Relative Wealth vs S&P 500 Benchmark")
ax.set_ylabel("Portfolio Value / Benchmark Value")
ax.legend(loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "backtest_relative_wealth_vs_benchmark.png", dpi=160)
show_or_close(fig)

# 4) Calendar-year realized returns.
annual_returns = daily_returns.groupby(daily_returns.index.year).apply(lambda frame: (1.0 + frame).prod() - 1.0)
annual_returns.index.name = "year"
annual_returns.to_csv(OUTPUT_DIR / "backtest_calendar_year_returns.csv")

annual_plot = annual_returns[plot_methods].T
max_abs = float(np.nanmax(np.abs(annual_plot.to_numpy())))
fig, ax = plt.subplots(figsize=(10, 5.8))
im = ax.imshow(annual_plot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-max_abs, vmax=max_abs)
ax.set_yticks(np.arange(len(annual_plot.index)))
ax.set_yticklabels([short_method_name(x) for x in annual_plot.index])
ax.set_xticks(np.arange(len(annual_plot.columns)))
ax.set_xticklabels(annual_plot.columns.astype(str))
ax.set_title("Calendar-Year Returns")
for i in range(annual_plot.shape[0]):
    for j in range(annual_plot.shape[1]):
        value = annual_plot.iat[i, j]
        ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8, color="black")
fig.colorbar(im, ax=ax, label="Annual Return")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "backtest_calendar_year_returns.png", dpi=160)
show_or_close(fig)

# 5) Rolling one-year return, volatility, and Sharpe.
rolling_window = TRADING_DAYS
rf_daily = risk_free["daily_rate_decimal"].reindex(daily_returns.index).ffill().bfill()

rolling_return = (1.0 + daily_returns[plot_methods]).rolling(rolling_window).apply(np.prod, raw=True) - 1.0
rolling_volatility = daily_returns[plot_methods].rolling(rolling_window).std(ddof=1) * np.sqrt(TRADING_DAYS)
rolling_sharpe = (
    (daily_returns[plot_methods].sub(rf_daily, axis=0)).rolling(rolling_window).mean() * TRADING_DAYS
) / rolling_volatility

rolling_return.to_csv(OUTPUT_DIR / "backtest_rolling_252d_return.csv")
rolling_volatility.to_csv(OUTPUT_DIR / "backtest_rolling_252d_volatility.csv")
rolling_sharpe.to_csv(OUTPUT_DIR / "backtest_rolling_252d_sharpe.csv")

fig, ax = plt.subplots(figsize=(12, 7))
for method in plot_methods:
    ax.plot(rolling_return.index, rolling_return[method], label=short_method_name(method), linewidth=1.5)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_title("Rolling 252-Day Return")
ax.set_ylabel("Rolling Return")
ax.legend(loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "backtest_rolling_252d_return.png", dpi=160)
show_or_close(fig)

fig, ax = plt.subplots(figsize=(12, 7))
for method in plot_methods:
    ax.plot(rolling_sharpe.index, rolling_sharpe[method], label=short_method_name(method), linewidth=1.5)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_title("Rolling 252-Day Sharpe Ratio")
ax.set_ylabel("Rolling Sharpe")
ax.legend(loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "backtest_rolling_252d_sharpe.png", dpi=160)
show_or_close(fig)

print("Saved additional backtest diagnostic charts.")
