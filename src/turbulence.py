"""
Turbulence index computation for the DRL stock-trading experiment.

Implements the market-turbulence measure from:
  'Practical Deep Reinforcement Learning Approach for Stock Trading'
  (FinRL / Ensemble DRL paper).

The turbulence index is the Mahalanobis distance squared of the current
cross-sectional return vector relative to its expanding-window historical
distribution:

    turbulence_t = (y_t - μ)ᵀ Σ⁻¹ (y_t - μ)

where μ and Σ are the expanding mean and covariance estimated on all returns
prior to date t.  High values indicate abnormal, correlated market moves;
the agent is trained to reduce position sizes when turbulence exceeds a
threshold calibrated on the training period.

No third-party library provides this exact formulation, so it is implemented
from scratch using NumPy linear algebra.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

_RIDGE_EPS = 1e-6   # diagonal regularisation to guard against near-singularity


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_turbulence_series(
    returns_df: pd.DataFrame,
    min_history: int = 252,
) -> pd.Series:
    """Compute the daily turbulence index using an expanding estimation window.

    For each date *t* that has at least ``min_history`` prior trading days of
    return data the turbulence is:

        turbulence_t = (y_t − μ)ᵀ Σ⁻¹ (y_t − μ)

    where

    * ``y_t``  — cross-sectional return vector at date *t* (one value per stock)
    * ``μ``    — expanding mean of returns from the series start through *t − 1*
    * ``Σ``    — expanding covariance of returns from the series start through *t − 1*

    Singularity / near-singularity of Σ is handled in two steps:

    1. A small ridge term ``ε·I`` (``ε = 1e-6``) is added to Σ before
       inversion as a first-pass regulariser.
    2. If the regularised matrix is still singular ``numpy.linalg.pinv`` is
       used as a fallback (minimum-norm pseudo-inverse).

    For dates with fewer than ``min_history`` prior observations turbulence is
    set to zero.

    Parameters
    ----------
    returns_df:
        Wide-format daily-return DataFrame produced by
        :func:`feature_engineering.compute_daily_returns`.
        Index: ``pd.Timestamp`` dates (ascending).
        Columns: one per ticker.
        The first row may contain NaN (artifact of ``pct_change``).
    min_history:
        Minimum number of prior trading days required before turbulence is
        computed.  Rows at positions ``0 … min_history − 1`` receive a value
        of 0.  Default is 252 (≈ one trading year).

    Returns
    -------
    pd.Series
        Turbulence values indexed by date (same index as ``returns_df``).
        Values are non-negative floats; early dates are 0.0.
    """
    # Drop leading NaN rows (e.g. first row from pct_change) and any rows
    # where ALL stocks are NaN.  Remaining per-cell NaNs are filled with 0
    # so that the cross-sectional vector always has a defined value.
    df = returns_df.dropna(how='all').copy()
    df.fillna(0.0, inplace=True)

    dates = df.index.tolist()
    n_dates = len(dates)
    values = np.zeros(n_dates, dtype=float)

    logger.info(
        "Computing turbulence series: %d dates, %d stocks, min_history=%d",
        n_dates, df.shape[1], min_history,
    )

    ret_matrix = df.to_numpy()   # shape (n_dates, n_stocks)

    for t in range(min_history, n_dates):
        history = ret_matrix[:t]          # shape (t, n_stocks) — all prior rows
        y_t = ret_matrix[t]               # shape (n_stocks,)

        mu = history.mean(axis=0)         # shape (n_stocks,)
        sigma = np.cov(history, rowvar=False)  # shape (n_stocks, n_stocks)

        dev = y_t - mu                    # deviation from historical mean

        # Ridge regularisation
        sigma_reg = sigma + _RIDGE_EPS * np.eye(sigma.shape[0])

        try:
            inv_sigma = np.linalg.inv(sigma_reg)
            # Sanity check: if result is numerically unreliable, fall back
            if not np.all(np.isfinite(inv_sigma)):
                raise np.linalg.LinAlgError("non-finite values in inverse")
        except np.linalg.LinAlgError:
            logger.debug(
                "Singular covariance at t=%d (%s); using pseudo-inverse",
                t, dates[t],
            )
            inv_sigma = np.linalg.pinv(sigma_reg)

        turb = float(dev @ inv_sigma @ dev)
        values[t] = max(turb, 0.0)   # numerical noise can produce tiny negatives

    # Re-attach to the full original index (pre-dropna) so callers can align easily
    series = pd.Series(values, index=dates, name='turbulence')

    # Reindex to the original returns_df index, filling any dropped dates with 0
    series = series.reindex(returns_df.index, fill_value=0.0)

    logger.info(
        "Turbulence series computed. Non-zero entries: %d / %d. "
        "Max: %.2f, Mean (post-history): %.2f",
        (series > 0).sum(),
        len(series),
        series.max(),
        series.iloc[min_history:].mean() if len(series) > min_history else float('nan'),
    )
    return series


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def compute_turbulence_threshold(
    turbulence_series: pd.Series,
    train_end: str,
    quantile: float = 0.90,
) -> float:
    """Determine the turbulence threshold from the training-period distribution.

    The threshold is the ``quantile``-th percentile of turbulence values
    observed up to and including ``train_end``.  Only values from the
    training period are used so that the threshold does not incorporate
    look-ahead information.

    Parameters
    ----------
    turbulence_series:
        Full turbulence series as returned by :func:`compute_turbulence_series`.
    train_end:
        Last date of the training period as ``'YYYY-MM-DD'``.  Values on or
        before this date are used for calibration.
    quantile:
        Percentile to use as the threshold (default 0.90 → 90th percentile).

    Returns
    -------
    float
        Turbulence threshold value.
    """
    train_end_ts = pd.Timestamp(train_end)
    train_turbulence = turbulence_series[turbulence_series.index <= train_end_ts]

    # Exclude zero values (warm-up period) from the quantile calculation
    nonzero = train_turbulence[train_turbulence > 0]
    if nonzero.empty:
        logger.warning(
            "No non-zero turbulence values in training period; returning 0."
        )
        return 0.0

    threshold = float(nonzero.quantile(quantile))
    logger.info(
        "Turbulence threshold (%.0f%%ile, training through %s): %.4f",
        quantile * 100,
        train_end,
        threshold,
    )
    return threshold


# ---------------------------------------------------------------------------
# Point query helper
# ---------------------------------------------------------------------------

def get_turbulence_at_date(
    date,
    returns_df: pd.DataFrame,
    turbulence_series: pd.Series,
) -> float:
    """Return the turbulence value for a specific date.

    Parameters
    ----------
    date:
        Date to query.  Accepts any type accepted by ``pd.Timestamp``.
    returns_df:
        Returns panel (used only for reference; not recomputed here).
    turbulence_series:
        Pre-computed turbulence series from :func:`compute_turbulence_series`.

    Returns
    -------
    float
        Turbulence value for ``date``, or 0.0 if the date is not present in
        the series.
    """
    ts = pd.Timestamp(date)
    if ts not in turbulence_series.index:
        logger.debug("Date %s not found in turbulence series; returning 0.0", ts.date())
        return 0.0
    return float(turbulence_series.loc[ts])
