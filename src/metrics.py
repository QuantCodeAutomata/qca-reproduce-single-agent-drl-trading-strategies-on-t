"""
Performance metrics for evaluating DRL trading strategies.

All metrics follow standard quantitative finance conventions.
Risk-free rate is set to 0 (paper unspecified; documented choice).
"""

import numpy as np
import pandas as pd

from src.config import ANNUALIZATION_FACTOR, RISK_FREE_RATE


def compute_cumulative_return(portfolio_values: pd.Series) -> float:
    """Compute cumulative return from portfolio value series."""
    return portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1


def compute_annualized_return(
    portfolio_values: pd.Series,
    n_trading_days: int = ANNUALIZATION_FACTOR,
) -> float:
    """Compute annualized return using geometric compounding."""
    n_days = len(portfolio_values) - 1
    if n_days <= 0:
        return 0.0
    cum_return = compute_cumulative_return(portfolio_values)
    return (1.0 + cum_return) ** (n_trading_days / n_days) - 1.0


def compute_annualized_volatility(
    portfolio_values: pd.Series,
    n_trading_days: int = ANNUALIZATION_FACTOR,
) -> float:
    """Compute annualized volatility from daily returns."""
    daily_returns = portfolio_values.pct_change().dropna()
    if daily_returns.empty:
        return 0.0
    return float(daily_returns.std() * np.sqrt(n_trading_days))


def compute_sharpe_ratio(
    portfolio_values: pd.Series,
    rf: float = RISK_FREE_RATE,
    n_trading_days: int = ANNUALIZATION_FACTOR,
) -> float:
    """Compute Sharpe ratio. rf=0 (paper-unspecified, documented choice)."""
    ann_vol = compute_annualized_volatility(portfolio_values, n_trading_days)
    if ann_vol == 0.0:
        return 0.0
    ann_return = compute_annualized_return(portfolio_values, n_trading_days)
    return (ann_return - rf) / ann_vol


def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """Compute maximum drawdown as a fraction (negative value)."""
    running_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - running_max) / running_max
    return float(drawdowns.min())


def compute_all_metrics(portfolio_values: pd.Series, name: str = '') -> dict:
    """Compute all metrics and return as dict."""
    return {
        'name': name,
        'cumulative_return': compute_cumulative_return(portfolio_values),
        'annualized_return': compute_annualized_return(portfolio_values),
        'annualized_volatility': compute_annualized_volatility(portfolio_values),
        'sharpe_ratio': compute_sharpe_ratio(portfolio_values),
        'max_drawdown': compute_max_drawdown(portfolio_values),
    }


def metrics_to_dataframe(metrics_list: list) -> pd.DataFrame:
    """Convert list of metrics dicts to a formatted DataFrame."""
    df = pd.DataFrame(metrics_list)
    if 'name' in df.columns:
        df = df.set_index('name')
    df = df.rename(columns={
        'cumulative_return': 'Cumul. Return',
        'annualized_return': 'Annual Return',
        'annualized_volatility': 'Annual Vol',
        'sharpe_ratio': 'Sharpe',
        'max_drawdown': 'Max DD',
    })
    pct_cols = ['Cumul. Return', 'Annual Return', 'Annual Vol', 'Max DD']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].map('{:.1%}'.format)
    if 'Sharpe' in df.columns:
        df['Sharpe'] = df['Sharpe'].map('{:.2f}'.format)
    return df
