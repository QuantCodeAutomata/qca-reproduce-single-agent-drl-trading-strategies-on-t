"""
Custom multi-stock trading environment for the DRL trading experiment.

Reproduces the environment specification from:
    'Practical Deep Reinforcement Learning Approach for Stock Trading'

State  : Box(181,) = [cash(1), adj_close(30), holdings(30),
                       macd(30), rsi(30), cci(30), adx(30)]
Action : Box(-1, 1, shape=(30,)) continuous
Reward : Δ portfolio value (transaction costs embedded in cash)
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    H_MAX,
    INITIAL_CAPITAL,
    N_STOCKS,
    SEED,
    STATE_DIM,
    TRANSACTION_COST_PCT,
)


# ---------------------------------------------------------------------------
# TurbulenceMonitor
# ---------------------------------------------------------------------------

class TurbulenceMonitor:
    """
    Helper class to manage turbulence override state across quarters.

    Behaviour (latch semantics):
      - Override activates the moment turbulence exceeds `threshold`.
      - Override deactivates once turbulence falls back at or below `threshold`.

    This prevents rapid on/off toggling caused by turbulence oscillating
    around the threshold boundary.

    Parameters
    ----------
    turbulence_series : pd.Series
        Daily turbulence values indexed by date.
    threshold : float
        Turbulence level above which the override triggers.
    """

    def __init__(self, turbulence_series: pd.Series, threshold: float) -> None:
        self.turbulence_series = turbulence_series
        self.threshold = threshold
        self._override_active: bool = False

    def reset(self) -> None:
        """Reset latch to inactive; call at the start of each episode."""
        self._override_active = False

    def is_override_active(self, date: Any) -> bool:
        """
        Return whether the turbulence override is active for *date*.

        If *date* is not in the turbulence series the current latch state is
        returned unchanged (conservative: keeps prior override decision).

        Parameters
        ----------
        date :
            The date to query (must be compatible with the series index).

        Returns
        -------
        bool
            True if the override is currently active.
        """
        if date not in self.turbulence_series.index:
            return self._override_active

        turb = float(self.turbulence_series[date])
        if turb > self.threshold:
            self._override_active = True
        elif self._override_active and turb <= self.threshold:
            self._override_active = False

        return self._override_active


# ---------------------------------------------------------------------------
# StockTradingEnv
# ---------------------------------------------------------------------------

class StockTradingEnv(gym.Env):
    """
    Custom multi-stock trading environment for 30 DJIA stocks.

    Implements the paper's environment specification:

    * 181-dimensional state:
        [cash, 30×adj_close, 30×holdings, 30×macd, 30×rsi, 30×cci, 30×adx]
    * 30-dimensional continuous action in [-1, 1].
    * Long-only, integer share holdings, non-negative cash constraint.
    * 0.1 % transaction cost on absolute traded notional.
    * Turbulence-based override: when turbulence > threshold, liquidate all
      positions and halt buys for that step.

    Action mapping (implementation choice, not stated in the paper):
        trade_shares_i = round(action_i × H_MAX)
        - trade_shares_i > 0 → buy (constrained by available cash)
        - trade_shares_i < 0 → sell (constrained by current holdings)

    Trade execution order (implementation choice):
        1. Process all sells, sorted by absolute sell quantity descending.
        2. Process all buys, sorted by action magnitude descending,
           until cash is exhausted.

    Reward definition:
        r_t = portfolio_value(t+1) − portfolio_value(t)
        Transaction costs are already embedded in cash during execution,
        so no separate cost term is subtracted from the reward.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format daily panel indexed by date.  Columns are 2-tuples
        (ticker, field) where *field* ∈ {'adj_close', 'macd', 'rsi',
        'cci', 'adx'}.
    price_col : str
        Column field name used for prices (default ``'adj_close'``).
    hmax : int
        Maximum shares traded per stock per step.
    initial_capital : float
        Starting cash balance.
    transaction_cost_pct : float
        Fractional transaction cost applied to the absolute traded notional.
    turbulence_series : pd.Series, optional
        Daily turbulence index values, indexed by date.
    turbulence_threshold : float, optional
        Override trigger level.  Required when *turbulence_series* is given.
    n_stocks : int
        Number of stocks (must equal 30 to match STATE_DIM=181).
    tickers : list of str, optional
        Ordered list of ticker symbols that defines the column ordering in
        the state vector.  If ``None``, the order is taken from the first
        level of ``df.columns``.
    """

    metadata: Dict[str, List] = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        price_col: str = "adj_close",
        hmax: int = H_MAX,
        initial_capital: float = INITIAL_CAPITAL,
        transaction_cost_pct: float = TRANSACTION_COST_PCT,
        turbulence_series: Optional[pd.Series] = None,
        turbulence_threshold: Optional[float] = None,
        n_stocks: int = N_STOCKS,
        tickers: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        # -- Parameters ------------------------------------------------------
        self.price_col = price_col
        self.hmax = hmax
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.turbulence_threshold = turbulence_threshold
        self.n_stocks = n_stocks

        # -- Tickers ---------------------------------------------------------
        if tickers is not None:
            self.tickers: List[str] = list(tickers)
        else:
            # Preserve column order, deduplicate
            self.tickers = list(df.columns.get_level_values(0).unique())

        assert len(self.tickers) == n_stocks, (
            f"Expected {n_stocks} tickers, got {len(self.tickers)}"
        )

        # -- Dates -----------------------------------------------------------
        self.dates: List = sorted(df.index.unique().tolist())
        self.n_steps: int = len(self.dates)

        # -- Extract data arrays: shape (n_dates, n_stocks) ------------------
        def _extract(field: str) -> np.ndarray:
            arr = np.array(
                [df[(t, field)].values for t in self.tickers], dtype=np.float32
            ).T  # (n_dates, n_stocks)
            return arr

        self.prices: np.ndarray = _extract(price_col)          # raw prices
        self.macd: np.ndarray = np.nan_to_num(_extract("macd"), nan=0.0)
        self.rsi: np.ndarray = np.nan_to_num(_extract("rsi"),  nan=0.0)
        self.cci: np.ndarray = np.nan_to_num(_extract("cci"),  nan=0.0)
        self.adx: np.ndarray = np.nan_to_num(_extract("adx"),  nan=0.0)

        # -- Turbulence ------------------------------------------------------
        self.turbulence_series: Optional[pd.Series] = turbulence_series
        self._turb_monitor: Optional[TurbulenceMonitor] = None
        if turbulence_series is not None and turbulence_threshold is not None:
            self._turb_monitor = TurbulenceMonitor(turbulence_series, turbulence_threshold)

        # -- Spaces ----------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_stocks,), dtype=np.float32
        )

        # -- Episode state (initialised by reset) ----------------------------
        self.current_step: int = 0
        self.cash: float = initial_capital
        self.holdings: np.ndarray = np.zeros(n_stocks, dtype=np.int64)
        self.daily_log: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Gymnasium interface
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to day 0 with full initial capital and no positions.

        Parameters
        ----------
        seed : int, optional
            Random seed forwarded to the parent class.
        options : dict, optional
            Unused; present for API compatibility.

        Returns
        -------
        observation : np.ndarray
            Initial 181-dimensional state vector.
        info : dict
            Dictionary with ``portfolio_value``, ``cash``, and ``holdings``.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = float(self.initial_capital)
        self.holdings = np.zeros(self.n_stocks, dtype=np.int64)
        self.daily_log = []

        if self._turb_monitor is not None:
            self._turb_monitor.reset()

        obs = self._get_state()
        info: Dict[str, Any] = {
            "portfolio_value": self._get_portfolio_value(),
            "cash": self.cash,
            "holdings": self.holdings.copy(),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step.

        Execution order
        ~~~~~~~~~~~~~~~
        1. Record pre-trade portfolio value at ``current_step``.
        2. Compute desired share trades: ``round(action × hmax)``.
        3. Apply turbulence override if active (liquidate all, zero buys).
        4. Execute sells (abs qty descending), then buys (magnitude descending).
        5. Advance ``current_step`` by one.
        6. Compute reward as change in portfolio value.
        7. Build and return next state.

        Parameters
        ----------
        action : np.ndarray
            Continuous action vector in [-1, 1] of shape ``(n_stocks,)``.

        Returns
        -------
        obs : np.ndarray
            Next 181-dimensional state vector.
        reward : float
            Change in total portfolio value over the step.
        terminated : bool
            True when the final trading day has been processed.
        truncated : bool
            Always False (no time limit beyond the data horizon).
        info : dict
            Contains ``portfolio_value``, ``cash``, ``holdings``,
            ``turbulence``, and ``turbulence_override``.
        """
        # 1. Pre-trade portfolio value
        old_value = self._get_portfolio_value()

        # 2. Desired integer trades
        desired_trades = np.round(np.asarray(action, dtype=np.float32) * self.hmax).astype(int)

        # 3. Turbulence check
        turbulence_override = self._check_turbulence()

        # 4. Execute trades (costs embedded in cash)
        self._execute_trades(desired_trades, turbulence_override)

        # 5. Advance to next day
        self.current_step += 1

        # 6. New portfolio value and reward
        new_value = self._get_portfolio_value()
        reward = float(new_value - old_value)

        # 7. Next state
        obs = self._get_state()

        # 8. Termination
        terminated: bool = self.current_step >= self.n_steps - 1
        truncated: bool = False

        # 9. Turbulence value for logging / info
        current_date = self.dates[self.current_step]
        turbulence_val: Optional[float] = None
        if self.turbulence_series is not None and current_date in self.turbulence_series.index:
            turbulence_val = float(self.turbulence_series[current_date])

        # 10. Log
        self.daily_log.append(
            {
                "date": current_date,
                "portfolio_value": new_value,
                "cash": self.cash,
                "holdings": self.holdings.copy(),
                "returns": reward / old_value if old_value != 0.0 else 0.0,
                "turbulence": turbulence_val,
            }
        )

        info: Dict[str, Any] = {
            "portfolio_value": new_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "turbulence": turbulence_val,
            "turbulence_override": turbulence_override,
        }

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Core helpers
    # -----------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """
        Build the 181-dimensional state vector for ``current_step``.

        Layout
        ------
        [cash(1), adj_close(30), holdings(30), macd(30), rsi(30), cci(30), adx(30)]

        Values are raw / unscaled; NaN in indicators was replaced with 0 at
        construction time.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(STATE_DIM,)`` = ``(181,)``.
        """
        t = self.current_step
        state = np.concatenate(
            [
                np.array([self.cash], dtype=np.float32),
                self.prices[t].astype(np.float32),
                self.holdings.astype(np.float32),
                self.macd[t].astype(np.float32),
                self.rsi[t].astype(np.float32),
                self.cci[t].astype(np.float32),
                self.adx[t].astype(np.float32),
            ]
        )
        assert len(state) == STATE_DIM, (
            f"State dimension mismatch: got {len(state)}, expected {STATE_DIM}"
        )
        return state

    def _execute_trades(
        self, desired_trades: np.ndarray, turbulence_override: bool
    ) -> float:
        """
        Execute trades against ``self.cash`` and ``self.holdings``.

        Transaction costs are deducted from / added to ``self.cash`` inline so
        that the reward (Δ portfolio value) automatically accounts for them.

        Parameters
        ----------
        desired_trades : np.ndarray
            Integer trade quantities (positive = buy, negative = sell).
        turbulence_override : bool
            When True, liquidate all holdings and skip buys.

        Returns
        -------
        float
            Total transaction cost incurred this step.
        """
        prices = self.prices[self.current_step]
        total_cost = 0.0

        if turbulence_override:
            # Liquidate every position at current prices
            for i in range(self.n_stocks):
                shares = int(self.holdings[i])
                if shares > 0 and prices[i] > 0:
                    notional = shares * prices[i]
                    cost = notional * self.transaction_cost_pct
                    self.cash += notional - cost
                    self.holdings[i] = 0
                    total_cost += cost
            return total_cost

        # -- Sells -----------------------------------------------------------
        # Process in order of absolute sell quantity (largest first)
        sell_mask = desired_trades < 0
        if sell_mask.any():
            sell_idx = np.where(sell_mask)[0]
            order = sell_idx[np.argsort(-np.abs(desired_trades[sell_idx]))]
            for i in order:
                shares = min(int(-desired_trades[i]), int(self.holdings[i]))
                if shares > 0 and prices[i] > 0:
                    notional = shares * prices[i]
                    cost = notional * self.transaction_cost_pct
                    self.cash += notional - cost
                    self.holdings[i] -= shares
                    total_cost += cost

        # -- Buys ------------------------------------------------------------
        # Process in order of action magnitude (largest first) until cash runs out
        buy_mask = desired_trades > 0
        if buy_mask.any():
            buy_idx = np.where(buy_mask)[0]
            order = buy_idx[np.argsort(-desired_trades[buy_idx])]
            for i in order:
                if self.cash <= 0 or prices[i] <= 0:
                    break
                cost_per_share = prices[i] * (1.0 + self.transaction_cost_pct)
                max_affordable = int(self.cash / cost_per_share)
                shares = min(int(desired_trades[i]), max_affordable)
                if shares > 0:
                    notional = shares * prices[i]
                    cost = notional * self.transaction_cost_pct
                    self.cash -= notional + cost
                    self.holdings[i] += shares
                    total_cost += cost

        return total_cost

    def _get_portfolio_value(self) -> float:
        """
        Compute total portfolio value at ``current_step``.

        Returns
        -------
        float
            ``cash + sum(holdings × current_prices)``
        """
        return float(self.cash + np.dot(self.holdings, self.prices[self.current_step]))

    def _check_turbulence(self) -> bool:
        """
        Check whether the turbulence override is active on the current date.

        Returns
        -------
        bool
            True if turbulence exceeds the threshold; False if no turbulence
            data is configured or the date is missing from the series.
        """
        if self._turb_monitor is None:
            return False
        return self._turb_monitor.is_override_active(self.dates[self.current_step])

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def get_daily_log(self) -> pd.DataFrame:
        """
        Return the per-step log accumulated during the episode.

        Columns
        -------
        date, portfolio_value, cash, holdings, returns, turbulence

        Returns
        -------
        pd.DataFrame
            One row per completed step.  ``holdings`` column contains integer
            arrays.  Returns an empty DataFrame if no steps have been taken.
        """
        if not self.daily_log:
            return pd.DataFrame(
                columns=["date", "portfolio_value", "cash", "holdings", "returns", "turbulence"]
            )
        return pd.DataFrame(self.daily_log)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=== StockTradingEnv smoke test ===\n")

    # -- Build a synthetic wide-format DataFrame ----------------------------
    N_DAYS = 60
    N_TEST_STOCKS = 30
    rng = np.random.default_rng(SEED)

    test_dates = pd.date_range("2020-01-02", periods=N_DAYS, freq="B")
    test_tickers = [f"STK{i:02d}" for i in range(N_TEST_STOCKS)]
    fields = ["adj_close", "macd", "rsi", "cci", "adx"]

    columns = pd.MultiIndex.from_product([test_tickers, fields])
    raw_data: Dict[Tuple[str, str], np.ndarray] = {}
    for t in test_tickers:
        raw_data[(t, "adj_close")] = rng.uniform(50.0, 300.0, N_DAYS)
        raw_data[(t, "macd")]      = rng.uniform(-5.0,   5.0, N_DAYS)
        raw_data[(t, "rsi")]       = rng.uniform(20.0,  80.0, N_DAYS)
        raw_data[(t, "cci")]       = rng.uniform(-200.0, 200.0, N_DAYS)
        raw_data[(t, "adx")]       = rng.uniform(10.0,  50.0, N_DAYS)

    mock_df = pd.DataFrame(raw_data, index=test_dates)

    # Inject a NaN in one indicator to verify nan-handling
    mock_df[(test_tickers[0], "macd")].iloc[0] = float("nan")

    # -- Turbulence series --------------------------------------------------
    turb_values = rng.uniform(0.0, 5.0, N_DAYS)
    turb_values[10] = 999.0          # force an override on day 10
    turb_series = pd.Series(turb_values, index=test_dates)
    turb_threshold = 50.0

    # -- Create env and run -------------------------------------------------
    env = StockTradingEnv(
        df=mock_df,
        tickers=test_tickers,
        turbulence_series=turb_series,
        turbulence_threshold=turb_threshold,
    )

    obs, info = env.reset(seed=SEED)
    print(f"[reset] obs.shape={obs.shape}, portfolio_value={info['portfolio_value']:,.0f}")
    assert obs.shape == (STATE_DIM,), f"Bad obs shape: {obs.shape}"

    total_reward = 0.0
    n_steps_run = 0
    for step_idx in range(N_DAYS - 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n_steps_run += 1

        if info["turbulence_override"]:
            print(
                f"  step {step_idx:3d} | TURBULENCE OVERRIDE | "
                f"portfolio={info['portfolio_value']:>14,.2f}"
            )

        if terminated:
            print(f"  Episode terminated at step {step_idx}.")
            break

    print(f"\n[done] steps={n_steps_run}, total_reward={total_reward:,.2f}")
    print(f"       final portfolio_value={info['portfolio_value']:,.2f}")

    log_df = env.get_daily_log()
    assert len(log_df) == n_steps_run, "Daily log length mismatch"
    assert set(log_df.columns) >= {"date", "portfolio_value", "cash", "holdings", "returns", "turbulence"}
    print(f"       daily_log rows={len(log_df)}, cols={list(log_df.columns)}")
    print(f"\nPortfolio value stats:\n{log_df['portfolio_value'].describe()}")

    # -- Verify TurbulenceMonitor standalone --------------------------------
    monitor = TurbulenceMonitor(turb_series, turb_threshold)
    assert monitor.is_override_active(test_dates[10]) is True,  "Override should be active on day 10"
    assert monitor.is_override_active(test_dates[11]) is False, "Override should deactivate on day 11"
    print("\nTurbulenceMonitor checks passed.")

    print("\n=== All smoke-test assertions passed ===")
