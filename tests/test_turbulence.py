import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.turbulence import compute_turbulence_series, compute_turbulence_threshold

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_returns() -> pd.DataFrame:
    """
    Small returns DataFrame: 5 stocks, 200 trading days of IID N(0, 0.01) returns.
    Seeded for reproducibility.
    """
    np.random.seed(0)
    idx = pd.date_range('2009-01-02', periods=200, freq='B')
    data = np.random.randn(200, 5) * 0.01
    return pd.DataFrame(data, index=idx, columns=[f'S{i}' for i in range(5)])


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_turbulence_nonnegative(simple_returns):
    """Turbulence is a squared Mahalanobis distance and must be >= 0."""
    turb = compute_turbulence_series(simple_returns, min_history=30)
    assert (turb.dropna() >= 0).all(), (
        'Turbulence values must be non-negative (Mahalanobis distance squared)'
    )


def test_turbulence_early_zeros(simple_returns):
    """
    Dates before the min_history threshold should have turbulence = 0
    (insufficient data to estimate the covariance matrix reliably).
    """
    min_history = 50
    turb = compute_turbulence_series(simple_returns, min_history=min_history)
    early = turb.iloc[:min_history]
    assert (early == 0).all(), (
        f'Expected turbulence = 0 for first {min_history} dates, got:\n{early[early != 0]}'
    )


def test_turbulence_shape(simple_returns):
    """Output Series must have the same length as the input DataFrame."""
    turb = compute_turbulence_series(simple_returns, min_history=30)
    assert len(turb) == len(simple_returns), (
        f'Expected length {len(simple_returns)}, got {len(turb)}'
    )


def test_threshold_within_series(simple_returns):
    """
    The turbulence threshold should lie within [min, max] of the
    training-period turbulence values (non-zero entries only).
    """
    min_history = 30
    turb = compute_turbulence_series(simple_returns, min_history=min_history)

    # Use the first 100 dates as training period
    train_end = str(turb.index[99].date())
    train_turb = turb.iloc[:100]
    active = train_turb[train_turb > 0]
    if active.empty:
        pytest.skip('No non-zero turbulence values in training slice')

    # Pass the full series; function filters internally using train_end
    threshold = compute_turbulence_threshold(turb, train_end=train_end, quantile=0.90)
    assert active.min() <= threshold <= active.max(), (
        f'Threshold {threshold:.4f} outside [{active.min():.4f}, {active.max():.4f}]'
    )


def test_turbulence_known_case():
    """
    With identity covariance (Sigma ~ I) and zero mean (mu ~ 0), turbulence
    for a return vector r approximates r^T * I * r = sum(r_i^2).

    We use 500 demeaned IID N(0,1) observations so sample estimates converge,
    then check that the final date (return = [1, 0, 0, 0, 0]) gives turbulence
    approximately 1.0.
    """
    n_stocks = 5
    min_history = 30

    np.random.seed(999)
    idx = pd.date_range('2000-01-03', periods=501, freq='B')
    hist_data = np.random.randn(500, n_stocks)
    # Demean so sample mean is exactly 0 for the history
    hist_data -= hist_data.mean(axis=0)
    test_row = np.array([[1.0] + [0.0] * (n_stocks - 1)])
    data = np.vstack([hist_data, test_row])

    returns = pd.DataFrame(data, index=idx, columns=[f'S{i}' for i in range(n_stocks)])
    turb = compute_turbulence_series(returns, min_history=min_history)

    last_val = float(turb.iloc[-1])
    # With Sigma ~ I and r = [1, 0, ..., 0]: turbulence ~ 1.0
    assert abs(last_val - 1.0) < 0.5, (
        f'Expected turbulence ~ 1.0 for unit-vector return with Sigma ~ I, got {last_val:.4f}'
    )
