"""
Rolling quarterly schedule generation for the DRL trading experiment.

Implements the expanding-window retraining schedule from the paper:
  - Training window expands by one quarter each period.
  - Validation window is always the quarter immediately before deployment.
  - Deployment window is one calendar quarter (last window truncated at
    the data end date).
"""

from typing import List

import pandas as pd


def generate_quarterly_windows(
    trade_start: str,
    trade_end: str,
    train_start: str,
    val_start: str,
    val_end: str,
) -> List[dict]:
    """Generate rolling quarterly windows following the paper's schedule.

    Each window dict contains the string date boundaries for the training,
    validation, and deployment sub-periods.

    Initial window
    --------------
    * train : ``train_start`` → ``val_start`` − 1 day
    * val   : ``val_start`` → ``val_end``
    * deploy: ``trade_start`` → end of the quarter that contains ``trade_start``

    Subsequent windows (advance by one quarter)
    -------------------------------------------
    * train_end  advances to the previous val_end  (expanding window)
    * val        becomes the previous deploy quarter (by calendar boundary)
    * deploy     advances to the next calendar quarter

    The deploy_end of the last window is clamped to ``trade_end``.

    Parameters
    ----------
    trade_start:
        First date of the trading (deployment) period, e.g. ``'2016-01-04'``.
    trade_end:
        Last date of the trading period, e.g. ``'2020-05-08'``.
    train_start:
        Absolute start of the expanding training window, e.g. ``'2009-01-01'``.
    val_start:
        Start of the first validation quarter, e.g. ``'2015-10-01'``.
    val_end:
        End of the first validation quarter, e.g. ``'2015-12-31'``.

    Returns
    -------
    list of dict
        Each dict has keys: ``train_start``, ``train_end``, ``val_start``,
        ``val_end``, ``deploy_start``, ``deploy_end`` (all ``'YYYY-MM-DD'``
        strings).
    """
    trade_end_ts = pd.Timestamp(trade_end)

    # Mutable window boundaries
    cur_train_start = pd.Timestamp(train_start)
    cur_train_end = pd.Timestamp(val_start) - pd.DateOffset(days=1)
    cur_val_start = pd.Timestamp(val_start)
    cur_val_end = pd.Timestamp(val_end)

    # First deploy period: starts at trade_start, ends at the calendar
    # quarter boundary that contains trade_start
    cur_deploy_start = pd.Timestamp(trade_start)
    cur_deploy_period = cur_deploy_start.to_period('Q')
    cur_deploy_end = cur_deploy_period.end_time.normalize()

    windows: List[dict] = []

    while cur_deploy_start <= trade_end_ts:
        actual_deploy_end = min(cur_deploy_end, trade_end_ts)

        windows.append({
            'train_start': cur_train_start.strftime('%Y-%m-%d'),
            'train_end': cur_train_end.strftime('%Y-%m-%d'),
            'val_start': cur_val_start.strftime('%Y-%m-%d'),
            'val_end': cur_val_end.strftime('%Y-%m-%d'),
            'deploy_start': cur_deploy_start.strftime('%Y-%m-%d'),
            'deploy_end': actual_deploy_end.strftime('%Y-%m-%d'),
        })

        # --- Advance by one quarter ---
        next_deploy_period = cur_deploy_period + 1

        cur_train_end = cur_val_end
        # Val becomes the previous deploy quarter (use calendar start, not
        # trade_start, so subsequent val windows are full calendar quarters)
        cur_val_start = cur_deploy_period.start_time.normalize()
        cur_val_end = cur_deploy_end
        cur_deploy_start = next_deploy_period.start_time.normalize()
        cur_deploy_end = next_deploy_period.end_time.normalize()
        cur_deploy_period = next_deploy_period

    return windows


def generate_rolling_windows(
    train_start: str,
    deploy_start: str,
    deploy_end: str,
    val_months: int = 3,
) -> List[dict]:
    """Simplified quarterly window generator for benchmark use (exp_4).

    Generates one dict per deployment quarter between ``deploy_start`` and
    ``deploy_end``.  Each dict only specifies the deployment boundaries;
    callers that need full train/val splits should use
    :func:`generate_quarterly_windows` instead.

    Parameters
    ----------
    train_start:
        Absolute start of historical data, e.g. ``'2009-01-01'``.
    deploy_start:
        First date of the first deployment quarter, e.g. ``'2016-01-04'``.
    deploy_end:
        Last date of the final deployment quarter, e.g. ``'2020-05-08'``.
    val_months:
        Number of months per validation window (default 3 = one quarter).

    Returns
    -------
    list of dict
        Each dict has keys ``train_start``, ``train_end``, ``deploy_start``,
        ``deploy_end`` (all ``'YYYY-MM-DD'`` strings).
    """
    trade_end_ts = pd.Timestamp(deploy_end)
    cur_start = pd.Timestamp(deploy_start)
    cur_period = cur_start.to_period('Q')

    windows: List[dict] = []
    while cur_start <= trade_end_ts:
        quarter_end = cur_period.end_time.normalize()
        actual_end = min(quarter_end, trade_end_ts)
        windows.append({
            'train_start': train_start,
            'train_end': (cur_start - pd.DateOffset(days=1)).strftime('%Y-%m-%d'),
            'deploy_start': cur_start.strftime('%Y-%m-%d'),
            'deploy_end': actual_end.strftime('%Y-%m-%d'),
        })
        cur_period = cur_period + 1
        cur_start = cur_period.start_time.normalize()

    return windows


def get_df_slice(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Slice a DataFrame by date range (inclusive on both ends).

    Handles DataFrames with a DatetimeIndex (from ``build_state_panel``) and
    DataFrames with a ``date`` column (long-format panels).

    Parameters
    ----------
    df:
        Panel DataFrame to slice.
    start_date:
        Inclusive start date string, ``'YYYY-MM-DD'``.
    end_date:
        Inclusive end date string, ``'YYYY-MM-DD'``.

    Returns
    -------
    pd.DataFrame
        Slice of ``df`` covering the requested date range.  The original
        index is preserved for DatetimeIndex DataFrames; it is reset for
        long-format DataFrames.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if isinstance(df.index, pd.DatetimeIndex):
        return df.loc[start_ts:end_ts]

    # Long-format: filter on a 'date' column
    dates = pd.to_datetime(df['date'])
    mask = (dates >= start_ts) & (dates <= end_ts)
    return df.loc[mask].reset_index(drop=True)
