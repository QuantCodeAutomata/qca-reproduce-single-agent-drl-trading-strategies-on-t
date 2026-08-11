'''
Benchmark strategies for comparison with DRL agents.

Benchmarks implemented:
1. DJIA buy-and-hold: rebased DJIA index returns
2. Minimum-variance portfolio: quarterly rebalanced, long-only, PyPortfolioOpt

Implementation choices (paper under-specified):
- Min-variance uses expanding-history covariance from 2009 to rebalance date
- Quarterly rebalancing aligned with ensemble cadence
- LedoitWolf shrinkage for numerical stability
- No transaction costs for DJIA benchmark (pure index return)
- 0.1% transaction cost applied at each rebalance for min-variance
- Long-only weights (no shorts), sum to 1
- Risk-free rate = 0 for Sharpe (paper unspecified)
'''
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models
from typing import Optional, List, Tuple
from src.config import INITIAL_CAPITAL, TRANSACTION_COST_PCT, ANNUALIZATION_FACTOR


def build_djia_benchmark(
    djia_series: pd.Series,
    trade_start: str,
    trade_end: str,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.Series:
    """
    Create portfolio value series from DJIA index levels, rebased to initial_capital.

    V_t = initial_capital * (DJIA_t / DJIA_0), where DJIA_0 is the first level
    in the trading window.

    Args:
        djia_series: Daily Series indexed by date with DJIA index levels.
        trade_start: Start date of trading period (inclusive).
        trade_end: End date of trading period (inclusive).
        initial_capital: Initial portfolio value to rebase to.

    Returns:
        pd.Series of portfolio values indexed by trading dates, named 'DJIA'.

    Raises:
        ValueError: If no DJIA data exists in the specified date range.
    """
    series = djia_series.loc[trade_start:trade_end].dropna()
    if series.empty:
        raise ValueError(f"No DJIA data found between {trade_start} and {trade_end}")
    base_level = series.iloc[0]
    portfolio = initial_capital * (series / base_level)
    portfolio.name = 'DJIA'
    return portfolio


def compute_min_variance_rebalance_dates(windows: list) -> list:
    """
    Extract rebalance dates from quarterly rolling windows.

    Each window dict must have a 'deploy_start' key indicating when that
    quarter's portfolio goes live, which is also the rebalance trigger date.

    Args:
        windows: List of window dicts, each with at least a 'deploy_start' key.

    Returns:
        Sorted list of unique rebalance date strings.
    """
    dates = [w['deploy_start'] for w in windows]
    return sorted(set(dates))


def build_min_variance_benchmark(
    returns_df: pd.DataFrame,
    rebalance_dates: list,
    trade_start: str,
    trade_end: str,
    all_prices_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Build minimum-variance portfolio with quarterly rebalancing and transaction costs.

    For each rebalance date:
      1. Use all prices strictly before the rebalance date (expanding window).
      2. Compute LedoitWolf shrinkage covariance via risk_models.CovarianceShrinkage.
      3. Solve min_volatility with EfficientFrontier(None, S, weight_bounds=(0, 1)).
      4. Apply weights until the next rebalance date.
      5. Deduct one-way turnover cost (transaction_cost_pct) at each rebalance.

    Turnover is computed as 0.5 * sum(|new_weights - old_weights|), representing
    the fraction of portfolio value that changes hands on one side of the trade.

    Args:
        returns_df: Daily returns DataFrame (dates × tickers). Used for date index
            and determining the trading calendar.
        rebalance_dates: List of dates at which to rebalance (from
            compute_min_variance_rebalance_dates).
        trade_start: Start of trading period (inclusive).
        trade_end: End of trading period (inclusive).
        all_prices_df: Daily price DataFrame (dates × tickers) covering both
            training history and trading period, for covariance estimation and
            portfolio valuation.
        initial_capital: Starting capital in dollars.
        transaction_cost_pct: One-way transaction cost rate (default 0.1%).

    Returns:
        Tuple of:
          - portfolio_value_series: pd.Series of daily portfolio values, named
            'MinVariance'.
          - weights_log_df: pd.DataFrame of weights at each rebalance date,
            index named 'rebalance_date', columns are ticker symbols.

    Raises:
        ValueError: If no trading dates exist in the specified period.
    """
    trade_dates = returns_df.loc[trade_start:trade_end].index
    if len(trade_dates) == 0:
        raise ValueError(f"No trading dates found between {trade_start} and {trade_end}")

    tickers = list(returns_df.columns)
    n_stocks = len(tickers)

    # Sort and queue rebalance trigger dates
    rebal_queue = sorted([pd.Timestamp(d) for d in rebalance_dates])

    # Portfolio state
    holdings = np.zeros(n_stocks)   # shares held for each ticker
    cash = float(initial_capital)
    current_weights = np.zeros(n_stocks)

    portfolio_values: dict = {}
    weights_log: dict = {}

    rebal_idx = 0

    for date in trade_dates:
        prices_today = all_prices_df.loc[date, tickers].values.astype(float)
        port_value = cash + float(np.dot(holdings, prices_today))

        # Trigger rebalance if this date has passed any scheduled date
        triggered = False
        while rebal_idx < len(rebal_queue) and date >= rebal_queue[rebal_idx]:
            rebal_idx += 1
            triggered = True

        if triggered:
            # Expanding window: all prices strictly before this rebalance date
            prices_hist = all_prices_df.loc[all_prices_df.index < date, tickers]

            new_weights = _solve_min_variance_weights(prices_hist, tickers, n_stocks)
            if new_weights is None:
                # Fallback: equal weights if optimisation fails or insufficient history
                new_weights = (
                    current_weights if current_weights.sum() > 0
                    else np.ones(n_stocks) / n_stocks
                )

            # One-way turnover cost: half the sum of absolute weight changes
            turnover = 0.5 * float(np.sum(np.abs(new_weights - current_weights)))
            cost = transaction_cost_pct * turnover * port_value

            # Fully invest remaining capital at new target weights
            net_value = port_value - cost
            safe_prices = np.where(prices_today > 0, prices_today, 1e-10)
            holdings = (new_weights * net_value) / safe_prices
            cash = max(0.0, net_value - float(np.dot(holdings, prices_today)))

            current_weights = new_weights.copy()
            weights_log[date] = dict(zip(tickers, new_weights))

            # Recompute after rebalance
            port_value = cash + float(np.dot(holdings, prices_today))

        portfolio_values[date] = port_value

    portfolio_series = pd.Series(portfolio_values, name='MinVariance')
    weights_log_df = pd.DataFrame(weights_log).T
    weights_log_df.index.name = 'rebalance_date'

    return portfolio_series, weights_log_df


def _solve_min_variance_weights(
    prices_hist: pd.DataFrame,
    tickers: list,
    n_stocks: int,
) -> Optional[np.ndarray]:
    """
    Solve minimum-variance optimisation with LedoitWolf shrinkage covariance.

    Requires at least max(n_stocks + 5, 60) price observations for a
    well-conditioned covariance estimate.

    Args:
        prices_hist: Historical price DataFrame (dates × tickers).
        tickers: Ordered list of ticker symbols matching DataFrame columns.
        n_stocks: Number of stocks (length of tickers).

    Returns:
        Array of portfolio weights summing to 1 (long-only), or None if
        optimisation fails or insufficient history exists.
    """
    min_obs = max(n_stocks + 5, 60)
    if len(prices_hist) < min_obs:
        return None

    try:
        S = risk_models.CovarianceShrinkage(prices_hist).ledoit_wolf()
        ef = EfficientFrontier(None, S, weight_bounds=(0, 1))
        ef.min_volatility()
        cleaned = ef.clean_weights()
        weights = np.array([cleaned.get(t, 0.0) for t in tickers], dtype=float)
        return weights
    except Exception:
        return None
