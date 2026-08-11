import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.benchmarks import build_djia_benchmark
from src.config import INITIAL_CAPITAL

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def djia_series() -> pd.Series:
    """
    Synthetic DJIA index series spanning 2016-01-04 to 2020-05-08 with a
    known linear growth pattern for deterministic testing.
    """
    dates = pd.date_range('2016-01-04', '2020-05-08', freq='B')
    # Linearly increase from 17,000 to 24,000 (approximate DJIA levels)
    levels = np.linspace(17_000, 24_000, len(dates))
    return pd.Series(levels, index=dates, name='DJIA')


@pytest.fixture
def djia_with_wider_history(djia_series) -> pd.Series:
    """DJIA series that extends before and after the trade window."""
    pre = pd.date_range('2009-01-02', '2015-12-31', freq='B')
    pre_levels = np.linspace(8_000, 17_000, len(pre))
    pre_series = pd.Series(pre_levels, index=pre, name='DJIA')
    return pd.concat([pre_series, djia_series]).sort_index()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_djia_benchmark_initial_value(djia_series):
    """The first value of the benchmark must equal INITIAL_CAPITAL exactly."""
    portfolio = build_djia_benchmark(djia_series, '2016-01-04', '2020-05-08')
    assert abs(portfolio.iloc[0] - INITIAL_CAPITAL) < 1e-6, (
        f'Expected initial value {INITIAL_CAPITAL}, got {portfolio.iloc[0]}'
    )


def test_djia_benchmark_length(djia_series):
    """
    Series must cover the full specified date range.
    The length should equal the number of business days between the two dates.
    """
    trade_start = '2016-01-04'
    trade_end = '2020-05-08'
    portfolio = build_djia_benchmark(djia_series, trade_start, trade_end)

    # Count expected business days directly from the fixture index
    expected = djia_series.loc[trade_start:trade_end].shape[0]
    assert len(portfolio) == expected, (
        f'Expected {expected} dates, got {len(portfolio)}'
    )


def test_djia_benchmark_positive(djia_series):
    """All portfolio values must be strictly positive (DJIA never went to 0)."""
    portfolio = build_djia_benchmark(djia_series, '2016-01-04', '2020-05-08')
    assert (portfolio > 0).all(), (
        f'Found non-positive portfolio values:\n{portfolio[portfolio <= 0]}'
    )


def test_djia_benchmark_scaling(djia_series):
    """
    Rebasing formula: V_t = INITIAL_CAPITAL * (DJIA_t / DJIA_0).

    Verify at every date that the portfolio value matches the formula exactly.
    """
    trade_start = '2016-01-04'
    trade_end = '2020-05-08'
    portfolio = build_djia_benchmark(djia_series, trade_start, trade_end)

    raw = djia_series.loc[trade_start:trade_end].dropna()
    expected = INITIAL_CAPITAL * (raw / raw.iloc[0])

    np.testing.assert_allclose(
        portfolio.values,
        expected.values,
        rtol=1e-6,
        atol=1e-6,
        err_msg='Portfolio values do not match rebasing formula',
    )


def test_djia_benchmark_uses_correct_window(djia_with_wider_history):
    """
    When DJIA data extends beyond the trade window, only dates within
    [trade_start, trade_end] should be returned, and rebasing uses the
    first date in that window (not the beginning of the full series).
    """
    trade_start = '2016-01-04'
    trade_end = '2020-05-08'
    portfolio = build_djia_benchmark(
        djia_with_wider_history, trade_start, trade_end
    )
    assert portfolio.index[0] >= pd.Timestamp(trade_start), (
        'Portfolio contains dates before trade_start'
    )
    assert portfolio.index[-1] <= pd.Timestamp(trade_end), (
        'Portfolio contains dates after trade_end'
    )
    assert abs(portfolio.iloc[0] - INITIAL_CAPITAL) < 1e-6, (
        'Rebasing must use the first date in the trade window, not the series start'
    )
