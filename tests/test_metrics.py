import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import (
    compute_cumulative_return,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_all_metrics,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def doubling_series() -> pd.Series:
    """Portfolio that doubles in value: cumulative return = 1.0."""
    idx = pd.date_range('2016-01-04', periods=252, freq='B')
    values = np.linspace(1_000_000, 2_000_000, 252)
    return pd.Series(values, index=idx)


@pytest.fixture
def halving_series() -> pd.Series:
    """Portfolio that halves in value: cumulative return = -0.5."""
    idx = pd.date_range('2016-01-04', periods=252, freq='B')
    values = np.linspace(1_000_000, 500_000, 252)
    return pd.Series(values, index=idx)


@pytest.fixture
def flat_series() -> pd.Series:
    """Portfolio with constant value: zero returns, zero variance."""
    idx = pd.date_range('2016-01-04', periods=252, freq='B')
    return pd.Series(np.ones(252) * 1_000_000, index=idx)


@pytest.fixture
def monotonic_up_series() -> pd.Series:
    """Strictly increasing portfolio: no drawdown."""
    idx = pd.date_range('2016-01-04', periods=252, freq='B')
    values = np.exp(np.linspace(0, 0.2, 252)) * 1_000_000
    return pd.Series(values, index=idx)


@pytest.fixture
def declining_then_recover() -> pd.Series:
    """Drops 40% then partially recovers: max drawdown should be negative."""
    idx = pd.date_range('2016-01-04', periods=252, freq='B')
    values = np.concatenate([
        np.linspace(1_000_000, 600_000, 126),   # 40% drop
        np.linspace(600_000, 800_000, 126),      # partial recovery
    ])
    return pd.Series(values, index=idx)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_cumulative_return_positive(doubling_series):
    """A portfolio that doubles should have cumulative return = 1.0."""
    result = compute_cumulative_return(doubling_series)
    assert abs(result - 1.0) < 1e-6, f'Expected 1.0, got {result}'


def test_cumulative_return_negative(halving_series):
    """A portfolio that halves should have cumulative return = -0.5."""
    result = compute_cumulative_return(halving_series)
    assert abs(result - (-0.5)) < 1e-6, f'Expected -0.5, got {result}'


def test_max_drawdown_negative(declining_then_recover):
    """A series with a 40% decline should have max_drawdown < 0."""
    dd = compute_max_drawdown(declining_then_recover)
    assert dd < 0, f'Expected max_drawdown < 0, got {dd}'
    # The peak is 1_000_000 and trough is 600_000 -> -40%
    assert abs(dd - (-0.4)) < 0.01, f'Expected max_drawdown ~ -0.40, got {dd}'


def test_max_drawdown_no_drawdown(monotonic_up_series):
    """A strictly monotonic increasing series should have max_drawdown = 0."""
    dd = compute_max_drawdown(monotonic_up_series)
    assert dd == 0.0 or abs(dd) < 1e-9, f'Expected drawdown = 0, got {dd}'


def test_sharpe_zero_vol(flat_series):
    """A flat portfolio (zero return, zero variance) must not raise an exception."""
    # Sharpe should be 0 or NaN when volatility is zero, but must not crash.
    result = compute_sharpe_ratio(flat_series)
    is_zero = result == 0.0
    is_nan = result != result  # NaN self-inequality
    assert is_zero or is_nan, (
        f'Expected 0 or NaN for zero-vol series, got {result}'
    )


def test_annualized_return_single_year(doubling_series):
    """
    A portfolio that doubles over ~252 trading days should have an annualized
    return of approximately 100% (2^(252/252) - 1 = 1.0).
    """
    ann_ret = compute_annualized_return(doubling_series, n_trading_days=252)
    assert abs(ann_ret - 1.0) < 0.05, (
        f'Expected annualized return ~ 1.0 for a doubling series, got {ann_ret}'
    )


def test_all_metrics_keys(doubling_series):
    """compute_all_metrics must return all required metric keys."""
    required_keys = {
        'cumulative_return',
        'annualized_return',
        'annualized_volatility',
        'sharpe_ratio',
        'max_drawdown',
    }
    result = compute_all_metrics(doubling_series)
    missing = required_keys - set(result.keys())
    assert not missing, f'Missing keys: {missing}'


def test_metrics_types(doubling_series):
    """All numeric metric values returned by compute_all_metrics must be float/int."""
    result = compute_all_metrics(doubling_series)
    # 'name' is an optional string label key, skip it
    numeric_keys = {k for k in result if k != 'name'}
    for key in numeric_keys:
        val = result[key]
        assert isinstance(val, (float, int, np.floating, np.integer)), (
            f'Expected numeric type for key {key!r}, got {type(val)}'
        )
