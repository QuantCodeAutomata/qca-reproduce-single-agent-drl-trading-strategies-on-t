"""
Feature engineering module for the DRL stock-trading experiment.

Implements the four technical indicators used in:
  'Practical Deep Reinforcement Learning Approach for Stock Trading'
  (FinRL / Ensemble DRL paper):

    * MACD  — momentum / trend
    * RSI   — relative strength (overbought / oversold)
    * CCI   — commodity channel index (cycle turns)
    * ADX   — average directional index (trend strength)

All indicators are computed via the ``ta`` library to ensure numerical
consistency with established implementations.
"""

import logging

import pandas as pd
import ta
import ta.momentum
import ta.trend

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """Compute the MACD line (fast EMA − slow EMA).

    Uses ``ta.trend.MACD`` which applies exponential moving averages with the
    standard Wilder smoothing convention.

    Parameters
    ----------
    close:
        Adjusted close-price series for a single ticker, indexed by date,
        sorted in ascending chronological order.
    fast:
        Fast EMA window (default 12).
    slow:
        Slow EMA window (default 26).
    signal:
        Signal-line EMA window (default 9).  Required by the constructor but
        only the MACD line itself is returned.

    Returns
    -------
    pd.Series
        MACD line values with the same index as ``close``.  NaN values appear
        during the warm-up period (first ``slow - 1`` bars).
    """
    return ta.trend.MACD(
        close,
        window_fast=fast,
        window_slow=slow,
        window_sign=signal,
    ).macd()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Parameters
    ----------
    close:
        Adjusted close-price series for a single ticker.
    period:
        Lookback window in trading days (default 14).

    Returns
    -------
    pd.Series
        RSI values in [0, 100] with the same index as ``close``.  NaN values
        appear during the warm-up period (first ``period`` bars).
    """
    return ta.momentum.RSIIndicator(close, window=period).rsi()


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Compute the Commodity Channel Index (CCI).

    Parameters
    ----------
    high:
        High-price series for a single ticker.
    low:
        Low-price series for a single ticker.
    close:
        Close-price series for a single ticker.
    period:
        Lookback window in trading days (default 20).

    Returns
    -------
    pd.Series
        CCI values with the same index as ``close``.  NaN values appear during
        the warm-up period.
    """
    return ta.trend.CCIIndicator(high, low, close, window=period).cci()


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute the Average Directional Index (ADX).

    Parameters
    ----------
    high:
        High-price series for a single ticker.
    low:
        Low-price series for a single ticker.
    close:
        Close-price series for a single ticker.
    period:
        Lookback window in trading days (default 14).

    Returns
    -------
    pd.Series
        ADX values in [0, 100] with the same index as ``close``.  NaN values
        appear during the warm-up period.
    """
    return ta.trend.ADXIndicator(high, low, close, window=period).adx()


# ---------------------------------------------------------------------------
# Panel-level feature construction
# ---------------------------------------------------------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD, RSI, CCI and ADX for every ticker in the panel.

    The panel is processed ticker-by-ticker so that each indicator is
    computed on a contiguous, single-stock price series.  Rows that fall
    within the warm-up window of any indicator (i.e. rows where at least one
    indicator is NaN) are dropped to avoid feeding undefined values into the
    environment.

    Parameters
    ----------
    df:
        Long-format stock panel with columns:
        [date, ticker, open, high, low, close, adj_close, volume].
        The ``date`` column must be a ``pd.Timestamp``.

    Returns
    -------
    pd.DataFrame
        Same schema as ``df`` plus four new columns:
        [macd, rsi, cci, adx].
        Sorted by [ticker, date] with the index reset.
    """
    from src.config import MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD, CCI_PERIOD, ADX_PERIOD

    enriched_frames = []

    for ticker, group in df.groupby('ticker'):
        g = group.sort_values('date').copy()

        g['macd'] = compute_macd(g['adj_close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        g['rsi'] = compute_rsi(g['adj_close'], period=RSI_PERIOD)
        g['cci'] = compute_cci(g['high'], g['low'], g['close'], period=CCI_PERIOD)
        g['adx'] = compute_adx(g['high'], g['low'], g['close'], period=ADX_PERIOD)

        enriched_frames.append(g)

    result = pd.concat(enriched_frames, ignore_index=True)
    result.sort_values(['ticker', 'date'], inplace=True)
    result.reset_index(drop=True, inplace=True)

    # Drop warm-up rows where any indicator is undefined
    indicator_cols = ['macd', 'rsi', 'cci', 'adx']
    before = len(result)
    result.dropna(subset=indicator_cols, inplace=True)
    result.reset_index(drop=True, inplace=True)
    logger.info(
        "add_technical_indicators: dropped %d warm-up rows, %d rows remain",
        before - len(result),
        len(result),
    )
    return result


def build_state_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the feature-engineered panel to wide format indexed by date.

    The resulting DataFrame has one row per trading date and a
    ``pd.MultiIndex`` on the columns with levels ``(ticker, field)``.  The
    fields included are those consumed by ``StockTradingEnv``:
    ``adj_close``, ``macd``, ``rsi``, ``cci``, ``adx``.

    Access pattern compatible with ``StockTradingEnv``::

        panel[(ticker, 'adj_close')]   # price series for one stock

    Parameters
    ----------
    df:
        Output of :func:`add_technical_indicators`.  Must contain columns:
        [date, ticker, adj_close, macd, rsi, cci, adx].

    Returns
    -------
    pd.DataFrame
        Wide-format panel indexed by ``date`` (``pd.Timestamp``).
        Columns are a ``pd.MultiIndex`` with levels ``(ticker, field)``.
        Shape: (n_trading_days, n_tickers × n_features).
        Sorted by date in ascending order.
    """
    feature_cols = ['adj_close', 'macd', 'rsi', 'cci', 'adx']
    tickers = sorted(df['ticker'].unique())

    pieces = {}
    for col in feature_cols:
        pivoted = df.pivot(index='date', columns='ticker', values=col)
        # Enforce consistent ticker order
        pivoted = pivoted[tickers]
        pieces[col] = pivoted

    # Build MultiIndex columns: (ticker, field)
    panel = pd.concat(pieces, axis=1)   # columns: (field, ticker)
    panel.columns = pd.MultiIndex.from_tuples(
        [(tkr, fld) for fld, tkr in panel.columns],
        names=['ticker', 'field'],
    )
    panel.sort_index(inplace=True)

    logger.info(
        "build_state_panel: shape %s, date range %s to %s",
        panel.shape,
        panel.index.min().date(),
        panel.index.max().date(),
    )
    return panel


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from adjusted close prices.

    Parameters
    ----------
    df:
        Long-format panel with columns [date, ticker, adj_close, ...].

    Returns
    -------
    pd.DataFrame
        Wide-format returns DataFrame indexed by ``date`` (``pd.Timestamp``),
        with one column per ticker.  The first row is NaN (no prior price).
        Shape: (n_trading_days, n_tickers).
    """
    prices = df.pivot(index='date', columns='ticker', values='adj_close')
    prices.sort_index(inplace=True)
    returns = prices.pct_change()
    returns.columns.name = 'ticker'
    logger.info(
        "compute_daily_returns: %d days × %d tickers",
        len(returns),
        len(returns.columns),
    )
    return returns
