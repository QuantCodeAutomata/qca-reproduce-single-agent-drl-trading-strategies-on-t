import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env_trading import StockTradingEnv
from src.config import STATE_DIM, N_STOCKS, INITIAL_CAPITAL, H_MAX

# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_env() -> StockTradingEnv:
    """
    Minimal StockTradingEnv with 50 trading days and 30 synthetic stocks.

    DataFrame columns are a MultiIndex of (ticker, field) with 5 fields:
    adj_close, macd, rsi, cci, adx.  Prices are centred near 100 so that
    the initial INITIAL_CAPITAL budget allows meaningful share purchases.
    """
    np.random.seed(42)
    tickers = [f'STK{i:02d}' for i in range(30)]
    dates = pd.date_range('2020-01-01', periods=50, freq='B')

    arrays = []
    for ticker in tickers:
        for field in ['adj_close', 'macd', 'rsi', 'cci', 'adx']:
            arrays.append((ticker, field))
    cols = pd.MultiIndex.from_tuples(arrays)

    data = np.random.randn(50, len(arrays)) * 0.1 + 100
    # RSI must be in [0, 100]
    rsi_mask = [i for i, (t, f) in enumerate(arrays) if f == 'rsi']
    data[:, rsi_mask] = np.abs(data[:, rsi_mask]) % 100

    df = pd.DataFrame(data, index=dates, columns=cols)
    return StockTradingEnv(df=df, tickers=tickers)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_holdings(env: StockTradingEnv) -> np.ndarray:
    """Return current holdings array from env, trying common attribute names."""
    for attr in ('holdings', 'shares_held', 'shares', 'inventory'):
        if hasattr(env, attr):
            return np.asarray(getattr(env, attr))
    raise AttributeError(
        'StockTradingEnv does not expose holdings via any of: '
        'holdings, shares_held, shares, inventory'
    )


def _get_cash(env: StockTradingEnv) -> float:
    """Return current cash balance from env, trying common attribute names."""
    for attr in ('cash', 'balance', 'available_cash', 'account_balance'):
        if hasattr(env, attr):
            return float(getattr(env, attr))
    raise AttributeError(
        'StockTradingEnv does not expose cash via any of: '
        'cash, balance, available_cash, account_balance'
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_state_dimension(mock_env):
    """Observation after reset must have shape (STATE_DIM,) = (181,)."""
    obs, _ = mock_env.reset()
    obs = np.asarray(obs)
    assert obs.shape == (STATE_DIM,), (
        f'Expected observation shape ({STATE_DIM},), got {obs.shape}'
    )


def test_initial_portfolio_value(mock_env):
    """After reset, cash must equal INITIAL_CAPITAL and all holdings must be 0."""
    mock_env.reset()
    cash = _get_cash(mock_env)
    holdings = _get_holdings(mock_env)

    assert abs(cash - INITIAL_CAPITAL) < 1.0, (
        f'Expected cash = {INITIAL_CAPITAL}, got {cash}'
    )
    assert np.sum(np.abs(holdings)) == 0, (
        f'Expected zero holdings after reset, got {holdings}'
    )


def test_no_negative_holdings(mock_env):
    """Holdings must never become negative after any sequence of random actions."""
    mock_env.reset()
    rng = np.random.default_rng(0)
    done = False
    step = 0
    while not done and step < 200:
        action = rng.uniform(-1, 1, size=(N_STOCKS,))
        _, _, terminated, truncated, _ = mock_env.step(action)
        done = terminated or truncated

        holdings = _get_holdings(mock_env)
        assert (holdings >= -1e-9).all(), (
            f'Negative holdings detected at step {step}: {holdings[holdings < 0]}'
        )
        step += 1


def test_no_negative_cash(mock_env):
    """Cash must never become negative after any sequence of random actions."""
    mock_env.reset()
    rng = np.random.default_rng(1)
    done = False
    step = 0
    while not done and step < 200:
        action = rng.uniform(-1, 1, size=(N_STOCKS,))
        _, _, terminated, truncated, _ = mock_env.step(action)
        done = terminated or truncated

        cash = _get_cash(mock_env)
        assert cash >= -1.0, (
            f'Negative cash detected at step {step}: {cash:.2f}'
        )
        step += 1


def test_episode_terminates(mock_env):
    """Env must terminate (done=True) after exhausting all available dates."""
    mock_env.reset()
    rng = np.random.default_rng(2)
    done = False
    steps = 0
    max_steps = 100  # generous upper bound (fixture has only 50 dates)
    while not done and steps < max_steps:
        action = rng.uniform(-1, 1, size=(N_STOCKS,))
        _, _, terminated, truncated, _ = mock_env.step(action)
        done = terminated or truncated
        steps += 1
    assert done, (
        f'Environment did not terminate within {max_steps} steps '
        f'(50-date episode, stepped {steps} times)'
    )


def test_action_space_shape(mock_env):
    """Action space must be continuous with shape (N_STOCKS,) = (30,)."""
    mock_env.reset()
    assert mock_env.action_space.shape == (N_STOCKS,), (
        f'Expected action_space.shape = ({N_STOCKS},), '
        f'got {mock_env.action_space.shape}'
    )


def test_observation_space_shape(mock_env):
    """Observation space must have shape (STATE_DIM,) = (181,)."""
    mock_env.reset()
    assert mock_env.observation_space.shape == (STATE_DIM,), (
        f'Expected observation_space.shape = ({STATE_DIM},), '
        f'got {mock_env.observation_space.shape}'
    )
