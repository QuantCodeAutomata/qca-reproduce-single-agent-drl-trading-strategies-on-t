"""
Data acquisition module for the DRL stock-trading experiment.

Downloads split/dividend-adjusted daily OHLCV data for the DJIA-30 universe
from the Massive market-data API and persists results as Parquet for fast
subsequent loads.

API authentication uses the environment variable ``MASSIVE_TOKEN``.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from massive import RESTClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_client() -> RESTClient:
    """Instantiate a Massive RESTClient from the environment token.

    Checks MASSIVE_API_KEY first, then MASSIVE_TOKEN as fallback.
    """
    api_key = os.getenv('MASSIVE_API_KEY') or os.getenv('MASSIVE_TOKEN')
    if not api_key:
        raise EnvironmentError(
            "Neither MASSIVE_API_KEY nor MASSIVE_TOKEN environment variable is set. "
            "Export it before running: export MASSIVE_API_KEY=<your-key>"
        )
    return RESTClient(api_key=api_key)


def _bars_to_df(ticker: str, bars) -> pd.DataFrame:
    """Convert an iterator of Massive Agg objects for one ticker to a DataFrame.

    Parameters
    ----------
    ticker:
        Ticker symbol string.
    bars:
        Iterator of ``massive.rest.models.aggs.Agg`` objects.

    Returns
    -------
    pd.DataFrame
        Columns: [date, ticker, open, high, low, close, adj_close, volume].
        ``adj_close`` mirrors ``close`` because the bars are fetched with
        ``adjusted=True`` (split-/dividend-adjusted prices).
    """
    records = []
    for bar in bars:
        # timestamp is Unix milliseconds → convert to date
        date = pd.Timestamp(bar.timestamp, unit='ms').date()
        records.append({
            'date': date,
            'ticker': ticker,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'adj_close': bar.close,   # adjusted=True → close IS adj_close
            'volume': bar.volume,
        })
    if not records:
        return pd.DataFrame(columns=['date', 'ticker', 'open', 'high', 'low',
                                     'close', 'adj_close', 'volume'])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_stock_data(
    tickers: list,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily adjusted OHLCV data for a list of tickers.

    Uses the Massive ``list_aggs`` endpoint with ``adjusted=True`` to obtain
    split- and dividend-adjusted prices.  The API's iterator handles
    pagination transparently.

    Parameters
    ----------
    tickers:
        List of ticker symbols, e.g. ``['AAPL', 'MSFT', ...]``.
    start_date:
        Inclusive start date as ``'YYYY-MM-DD'``.
    end_date:
        Inclusive end date as ``'YYYY-MM-DD'``.

    Returns
    -------
    pd.DataFrame
        Columns: [date, ticker, open, high, low, close, adj_close, volume].
        Sorted by [ticker, date].  Missing dates within a ticker's series are
        forward-filled; rows that cannot be filled are dropped.
    """
    client = _make_client()
    frames = []

    # Massive list_aggs 'to' is exclusive for daily bars; shift by one day
    to_exclusive = (
        pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ).strftime('%Y-%m-%d')

    for ticker in tickers:
        logger.info("Downloading %s  %s → %s", ticker, start_date, end_date)
        try:
            bars = client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='day',
                from_=start_date,
                to=to_exclusive,
                limit=50000,
                adjusted=True,
            )
            df_ticker = _bars_to_df(ticker, bars)
            if df_ticker.empty:
                logger.warning("No data returned for %s", ticker)
            else:
                frames.append(df_ticker)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download %s: %s", ticker, exc)

    if not frames:
        raise RuntimeError("No data was downloaded for any ticker.")

    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['ticker', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # UTX (United Technologies) merged with Raytheon on 2020-04-03 to form
    # RTX (Raytheon Technologies).  Substitute RTX prices for UTX after the
    # merger date so the full 2020-05-08 period is covered.
    # Implementation choice: RTX is the direct successor entity of UTX.
    if 'UTX' in df['ticker'].values:
        utx_last = df[df['ticker'] == 'UTX']['date'].max()
        end_dt = pd.Timestamp(end_date)
        if utx_last < end_dt:
            rtx_from = (utx_last + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            rtx_to = (end_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info(
                "UTX ended %s < %s; supplementing with RTX successor data "
                "from %s to %s",
                utx_last.date(), end_dt.date(), rtx_from, rtx_to,
            )
            try:
                rtx_bars = client.list_aggs(
                    ticker='RTX', multiplier=1, timespan='day',
                    from_=rtx_from, to=rtx_to,
                    limit=50000, adjusted=True,
                )
                rtx_df = _bars_to_df('RTX', rtx_bars)
                if not rtx_df.empty:
                    rtx_df['ticker'] = 'UTX'   # relabel as UTX continuation
                    frames.append(rtx_df)
                    df = pd.concat(frames, ignore_index=True)
                    df['date'] = pd.to_datetime(df['date'])
                    df.sort_values(['ticker', 'date'], inplace=True)
                    df.drop_duplicates(subset=['ticker', 'date'], keep='first',
                                       inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    logger.info(
                        "Appended %d RTX→UTX continuation bars", len(rtx_df)
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not fetch RTX successor data: %s", exc)

    # DD (DuPont) merged into DowDuPont (DWDP) on 2017-08-31; DWDP split and
    # DD re-listed as a new standalone entity on 2019-06-03.  Fill the gap
    # 2017-09-01 → 2019-05-31 using DWDP prices relabelled as 'DD'.
    # Implementation choice: DWDP is the direct legal successor of the original
    # DD during the merger period and is the closest available price proxy.
    if 'DD' in df['ticker'].values:
        dd_dates = df[df['ticker'] == 'DD']['date']
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        full_span_days = (min(end_dt, pd.Timestamp('2019-06-02')) -
                          max(start_dt, pd.Timestamp('2017-09-01'))).days
        if full_span_days > 0:
            # Check whether the gap exists (i.e. DWDP period not already covered)
            gap_start = max(start_dt, pd.Timestamp('2017-09-01'))
            gap_end = min(end_dt, pd.Timestamp('2019-05-31'))
            dd_in_gap = dd_dates[(dd_dates >= gap_start) & (dd_dates <= gap_end)]
            if len(dd_in_gap) == 0:
                dwdp_from = gap_start.strftime('%Y-%m-%d')
                dwdp_to = (gap_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(
                    "DD gap %s → %s; supplementing with DWDP successor data",
                    dwdp_from, gap_end.date(),
                )
                try:
                    dwdp_bars = client.list_aggs(
                        ticker='DWDP', multiplier=1, timespan='day',
                        from_=dwdp_from, to=dwdp_to,
                        limit=50000, adjusted=True,
                    )
                    dwdp_df = _bars_to_df('DWDP', dwdp_bars)
                    if not dwdp_df.empty:
                        dwdp_df['ticker'] = 'DD'   # relabel as DD continuation
                        df = pd.concat([df, dwdp_df], ignore_index=True)
                        df['date'] = pd.to_datetime(df['date'])
                        df.sort_values(['ticker', 'date'], inplace=True)
                        df.drop_duplicates(subset=['ticker', 'date'], keep='first',
                                           inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        logger.info("Appended %d DWDP→DD continuation bars", len(dwdp_df))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not fetch DWDP successor data: %s", exc)

    # Forward-fill missing dates within each ticker, then drop residuals
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    df[numeric_cols] = (
        df.groupby('ticker')[numeric_cols]
        .transform(lambda s: s.ffill())
    )
    df.dropna(subset=numeric_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Download complete: %d rows, %d tickers, %s to %s",
        len(df),
        df['ticker'].nunique(),
        df['date'].min().date(),
        df['date'].max().date(),
    )
    return df


def align_trading_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the panel to dates on which ALL tickers have a price record.

    This ensures the downstream environment always receives a complete
    cross-section of prices on every time step.

    Parameters
    ----------
    df:
        Raw panel DataFrame with columns [date, ticker, ...].

    Returns
    -------
    pd.DataFrame
        Filtered copy retaining only dates present for every unique ticker in
        ``df``.  Sorted by [ticker, date] and index is reset.
    """
    all_tickers = df['ticker'].unique()
    n_tickers = len(all_tickers)

    # Count how many tickers have a record per date
    date_counts = df.groupby('date')['ticker'].nunique()
    common_dates = date_counts[date_counts == n_tickers].index

    logger.info(
        "Trading-calendar alignment: %d dates → %d common dates",
        df['date'].nunique(),
        len(common_dates),
    )

    aligned = df[df['date'].isin(common_dates)].copy()
    aligned.sort_values(['ticker', 'date'], inplace=True)
    aligned.reset_index(drop=True, inplace=True)
    return aligned


def load_or_download(
    tickers: list,
    start_date: str,
    end_date: str,
    cache_path: str = 'data/stock_data.parquet',
) -> pd.DataFrame:
    """Return cached data if available, otherwise download and persist.

    Parameters
    ----------
    tickers:
        List of ticker symbols.
    start_date:
        Inclusive start date as ``'YYYY-MM-DD'``.
    end_date:
        Inclusive end date as ``'YYYY-MM-DD'``.
    cache_path:
        Path to the Parquet cache file.  Intermediate directories are created
        automatically.

    Returns
    -------
    pd.DataFrame
        Aligned panel with columns [date, ticker, open, high, low, close,
        adj_close, volume].
    """
    cache = Path(cache_path)

    if cache.exists():
        logger.info("Loading from cache: %s", cache)
        df = pd.read_parquet(cache)
        df['date'] = pd.to_datetime(df['date'])
        return df

    logger.info("Cache not found at %s — downloading from Massive API", cache)
    df = download_stock_data(tickers, start_date, end_date)
    df = align_trading_calendar(df)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    logger.info("Saved to %s", cache)
    return df


def download_djia_index(start_date: str, end_date: str) -> pd.Series:
    """Download daily DJIA index values via the DIA ETF (DJIA proxy).

    Uses the Massive API to fetch adjusted daily closes for the ticker
    ``'DIA'`` (iShares Dow Jones Industrial Average ETF).  If that fails the
    function retries with ``'^DJI'``.

    Parameters
    ----------
    start_date:
        Inclusive start date as ``'YYYY-MM-DD'``.
    end_date:
        Inclusive end date as ``'YYYY-MM-DD'``.

    Returns
    -------
    pd.Series
        Adjusted-close prices indexed by ``pd.Timestamp`` date, named
        ``'djia'``.
    """
    client = _make_client()

    for proxy in ('DIA', '^DJI'):
        logger.info("Downloading DJIA proxy via ticker %s", proxy)
        try:
            bars = client.list_aggs(
                ticker=proxy,
                multiplier=1,
                timespan='day',
                from_=start_date,
                to=end_date,
                limit=50000,
                adjusted=True,
            )
            df = _bars_to_df(proxy, bars)
            if df.empty:
                logger.warning("No data for %s, trying next proxy", proxy)
                continue
            df['date'] = pd.to_datetime(df['date'])
            series = df.set_index('date')['adj_close'].sort_index()
            series.name = 'djia'
            logger.info(
                "DJIA proxy downloaded: %d rows (%s to %s)",
                len(series),
                series.index.min().date(),
                series.index.max().date(),
            )
            return series
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download DJIA proxy %s: %s", proxy, exc)

    raise RuntimeError(
        "Could not download DJIA index data via either 'DIA' or '^DJI'."
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from src.config import (
        DJIA_TICKERS,
        START_DATE,
        END_DATE,
    )

    logger.info("=== Starting DJIA-30 data download ===")
    stock_df = load_or_download(
        tickers=DJIA_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        cache_path='data/stock_data.parquet',
    )
    logger.info("Stock panel shape: %s", stock_df.shape)
    logger.info("Columns: %s", stock_df.columns.tolist())
    logger.info("Date range: %s to %s",
                stock_df['date'].min().date(), stock_df['date'].max().date())

    djia = download_djia_index(START_DATE, END_DATE)
    djia.to_csv('data/djia_index.csv', header=True)
    logger.info("DJIA index saved to data/djia_index.csv (%d rows)", len(djia))
    logger.info("=== Done ===")
