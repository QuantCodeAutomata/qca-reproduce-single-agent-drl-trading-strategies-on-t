'''
Experiment 4: Benchmark Reproduction and Comparative Evaluation
Constructs DJIA buy-and-hold and minimum-variance portfolio benchmarks.

Paper targets:
  Min-variance: cumulative=31.7%, annual=6.5%, vol=17.8%, Sharpe=0.45, maxDD=-34.3%
  DJIA:         cumulative=38.6%, annual=7.8%, vol=20.1%, Sharpe=0.47, maxDD=-37.1%

Implementation choices:
  - Min-variance: expanding-history covariance, quarterly rebalancing, LedoitWolf shrinkage
  - 0.1% transaction cost at each rebalance for min-variance
  - DJIA benchmark: ^DJI index rebased to initial capital (no transaction costs)
  - Long-only weights, sum to 1
  - Risk-free rate = 0
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

warnings.filterwarnings('ignore')

from src.config import (
    INITIAL_CAPITAL, TRANSACTION_COST_PCT, ANNUALIZATION_FACTOR,
    DJIA_TICKERS, START_DATE, END_DATE,
)
from src.data_loader import load_or_download
from src.rolling_schedule import generate_rolling_windows
from src.metrics import compute_all_metrics
from src.benchmarks import (
    build_djia_benchmark,
    build_min_variance_benchmark,
    compute_min_variance_rebalance_dates,
)
from src.plots import plot_portfolio_comparison, plot_drawdowns, plot_metrics_table

TRAIN_START = '2009-01-01'
TRAIN_END   = '2015-12-31'
TRADE_START = '2016-01-04'
TRADE_END   = '2020-05-08'

RESULTS_DIR = Path('results')
DATA_DIR = Path('data')
RESULTS_DIR.mkdir(exist_ok=True)

_NUMERIC_METRIC_KEYS = {
    'cumulative_return', 'annualized_return', 'annualized_volatility',
    'sharpe_ratio', 'max_drawdown',
}


def _load_djia_series() -> pd.Series:
    """Load DJIA index levels from data/ cache or download."""
    cache_path = DATA_DIR / 'djia.parquet'
    if cache_path.exists():
        print('  Loading DJIA from cache ...')
        return pd.read_parquet(cache_path).squeeze()
    print('  Downloading DJIA index (^DJI) ...')
    djia = None
    try:
        import yfinance as yf  # type: ignore
        raw = yf.download('^DJI', start=TRAIN_START, end=TRADE_END, progress=False)
        # yfinance >= 0.2 returns MultiIndex columns; '^DJI' is an index (no dividends)
        # so 'Adj Close' may be absent — fall back to 'Close'.
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0)
            col = 'Adj Close' if 'Adj Close' in level0 else 'Close'
            djia = raw[col].squeeze()
        else:
            col = 'Adj Close' if 'Adj Close' in raw.columns else 'Close'
            djia = raw[col]
        djia.index = pd.to_datetime(djia.index)
        if djia.empty:
            raise ValueError('yfinance returned empty series for ^DJI')
    except Exception as exc:
        print(f'  yfinance failed ({exc}); DJIA will be approximated from constituent prices')
    if djia is None:
        # Approximate DJIA as equal-weight index of 30 constituents using cached parquet
        _df = load_or_download(tickers=DJIA_TICKERS, start_date=TRAIN_START, end_date=TRADE_END)
        _prices = _extract_prices(_df, DJIA_TICKERS)
        djia = _prices.mean(axis=1)
        print('  Using equal-weight constituent proxy for DJIA')
    djia = djia.rename('DJIA')
    djia.to_frame().to_parquet(cache_path)
    return djia


def _extract_prices(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Extract a wide adj_close price DataFrame (date x ticker) from long-format data."""
    if isinstance(df.columns, pd.MultiIndex):
        # Wide MultiIndex format: (ticker, feature)
        try:
            prices = df.xs('adj_close', axis=1, level=1)[tickers]
        except KeyError:
            prices = df.xs('adj_close', axis=1, level=0)[tickers]
    elif 'ticker' in df.columns and 'adj_close' in df.columns:
        # Long format with date column
        date_col = 'date' if 'date' in df.columns else df.index.name
        if date_col and date_col in df.columns:
            prices = df.pivot(index=date_col, columns='ticker', values='adj_close')
        else:
            prices = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')
        prices.index = pd.to_datetime(prices.index)
        prices.index.name = 'date'
        # Only keep requested tickers that exist
        available = [t for t in tickers if t in prices.columns]
        prices = prices[available]
    else:
        prices = df[tickers]
    return prices.astype(float)


def _print_metrics(name: str, metrics: dict) -> None:
    """Pretty-print a metrics dict, skipping non-numeric entries like 'name'."""
    print(f'\n  {name}:')
    for k, v in metrics.items():
        if k in _NUMERIC_METRIC_KEYS:
            print(f'    {k}: {v:.4f}')


def _numeric_only(m: dict) -> dict:
    """Return only numeric metric entries."""
    return {k: v for k, v in m.items() if k in _NUMERIC_METRIC_KEYS}


def _save_results_md(djia_metrics: dict, minvar_metrics: dict, extra_metrics: dict) -> None:
    """Write a comprehensive RESULTS.md to results/."""
    def _fmt_row(name: str, m: dict) -> str:
        return (
            f'| {name} '
            f'| {m.get("cumulative_return", float("nan")) * 100:.1f}% '
            f'| {m.get("annualized_return", float("nan")) * 100:.1f}% '
            f'| {m.get("annualized_volatility", float("nan")) * 100:.1f}% '
            f'| {m.get("sharpe_ratio", float("nan")):.2f} '
            f'| {m.get("max_drawdown", float("nan")) * 100:.1f}% |'
        )

    lines = [
        '# DRL Trading Strategy Experiment Results\n',
        '## Overview',
        f'Reproduction of *Practical Deep Reinforcement Learning Approach for Stock Trading*',
        f'- Out-of-sample period : {TRADE_START} to {TRADE_END}',
        f'- Initial capital      : ${INITIAL_CAPITAL:,.0f}',
        '- Risk-free rate       : 0 (paper unspecified)',
        f'- Annualization        : {ANNUALIZATION_FACTOR} trading days',
        '',
        '## Implementation Choices (Paper Under-Specified)',
        '- Min-variance: expanding-history LedoitWolf covariance, quarterly rebalancing, 0.1% one-way transaction cost',
        '- DJIA benchmark: ^DJI index rebased to initial capital (no transaction costs)',
        '- Long-only portfolio weights, sum to 1',
        '- Turbulence threshold: 90th percentile of training-period turbulence',
        '- H_MAX = 100 shares per stock per trade',
        '- Ensemble tie-break: PPO > A2C > DDPG',
        '',
        '## Benchmark Results (Exp 4)',
        '',
        '| Strategy | Cum. Return | Ann. Return | Ann. Vol | Sharpe | Max DD |',
        '|----------|------------|-------------|----------|--------|--------|',
        _fmt_row('DJIA Buy & Hold', djia_metrics),
        _fmt_row('Min-Variance', minvar_metrics),
    ]
    for strat, met in extra_metrics.items():
        lines.append(_fmt_row(strat, met))
    lines += [
        '',
        '### Paper Targets',
        '| Strategy     | Cum. Return | Ann. Return | Ann. Vol | Sharpe | Max DD |',
        '|--------------|------------|-------------|----------|--------|--------|',
        '| DJIA B&H     | 38.6%       | 7.8%        | 20.1%    | 0.47   | -37.1% |',
        '| Min-Variance | 31.7%       | 6.5%        | 17.8%    | 0.45   | -34.3% |',
        '',
        '## Single-Agent and Ensemble Results (Exp 1-3)',
        '*(Populated after running exp_1_single_agents.py, exp_2_ensemble.py, exp_3_turbulence.py)*',
        '',
        '## File Manifest',
        '- `results/benchmark_metrics.json`        -- JSON metrics for both benchmarks',
        '- `results/benchmark_portfolios.csv`       -- Daily portfolio values',
        '- `results/all_strategies_comparison.png`  -- Overlay of all strategies',
        '- `results/drawdowns.png`                  -- Drawdown series',
        '- `results/metrics_table.png`              -- Formatted metrics table',
    ]
    (RESULTS_DIR / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('  Saved results/RESULTS.md')


def main() -> None:
    print('=' * 60)
    print('Experiment 4: Benchmark Reproduction')
    print('=' * 60)

    print('\n[1] Loading stock data ...')
    df = load_or_download(tickers=DJIA_TICKERS, start_date=START_DATE, end_date=END_DATE)
    tickers = DJIA_TICKERS
    prices_df = _extract_prices(df, tickers)
    print(f'  Loaded {len(tickers)} tickers, {len(prices_df)} dates')

    returns_df = prices_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='all')

    print('\n[2] Loading DJIA index ...')
    djia_series = _load_djia_series()

    print('\n[3] Generating quarterly rebalance schedule ...')
    windows = generate_rolling_windows(train_start=TRAIN_START, deploy_start=TRADE_START, deploy_end=TRADE_END)
    rebalance_dates = compute_min_variance_rebalance_dates(windows)
    print(f'  {len(rebalance_dates)} rebalance dates: {rebalance_dates[0]} ... {rebalance_dates[-1]}')

    print('\n[4] Building DJIA buy-and-hold benchmark ...')
    djia_portfolio = build_djia_benchmark(djia_series, TRADE_START, TRADE_END)
    print(f'  Final value: ${djia_portfolio.iloc[-1]:,.0f}')

    print('\n[5] Building minimum-variance benchmark ...')
    minvar_portfolio, weights_log = build_min_variance_benchmark(
        returns_df=returns_df,
        rebalance_dates=rebalance_dates,
        trade_start=TRADE_START,
        trade_end=TRADE_END,
        all_prices_df=prices_df,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
    )
    print(f'  Final value: ${minvar_portfolio.iloc[-1]:,.0f}')
    print(f'  Weight log: {len(weights_log)} rebalances')

    print('\n[6] Computing metrics ...')
    djia_metrics = compute_all_metrics(djia_portfolio)
    minvar_metrics = compute_all_metrics(minvar_portfolio)
    _print_metrics('DJIA Buy & Hold', djia_metrics)
    _print_metrics('Min-Variance', minvar_metrics)

    portfolios: Dict[str, pd.Series] = {
        'DJIA Buy & Hold': djia_portfolio,
        'Min-Variance': minvar_portfolio,
    }
    extra_metrics: dict = {}

    # Load single-agent portfolios (exp_1 output)
    single_agent_path = RESULTS_DIR / 'single_agent_portfolios.csv'
    ensemble_path = RESULTS_DIR / 'ensemble_portfolio.csv'
    turb_path = RESULTS_DIR / 'turbulence_comparison.csv'

    loaded_any = False
    if single_agent_path.exists():
        print('\n[7] Loading exp_1 single-agent portfolio results ...')
        sa_df = pd.read_csv(single_agent_path, index_col=0, parse_dates=True)
        for col in sa_df.columns:
            portfolios[col] = sa_df[col].dropna()
            extra_metrics[col] = compute_all_metrics(portfolios[col])
        print(f'  Loaded single-agent strategies: {list(sa_df.columns)}')
        loaded_any = True
    else:
        print('\n[7] No single-agent results found (run exp_1_single_agents.py first).')

    if ensemble_path.exists():
        print('  Loading exp_2 ensemble portfolio results ...')
        ens_df = pd.read_csv(ensemble_path, index_col=0, parse_dates=True)
        ens_series = ens_df.iloc[:, 0].dropna()
        ens_series.name = 'Ensemble'
        portfolios['Ensemble'] = ens_series
        extra_metrics['Ensemble'] = compute_all_metrics(ens_series)
        print('  Loaded Ensemble strategy.')
        loaded_any = True

    if turb_path.exists():
        print('  Loading exp_3 turbulence comparison results ...')
        turb_df = pd.read_csv(turb_path, index_col=0, parse_dates=True)
        for col in turb_df.columns:
            if col not in portfolios:
                portfolios[col] = turb_df[col].dropna()
                extra_metrics[col] = compute_all_metrics(portfolios[col])
        print(f'  Loaded turbulence comparison strategies: {list(turb_df.columns)}')
        loaded_any = True

    if not loaded_any:
        print('\n[7] No prior experiment results found (run exp_1/2/3 first for overlays).')

    print('\n[8] Saving results ...')

    json_path = RESULTS_DIR / 'benchmark_metrics.json'
    with open(json_path, 'w') as f:
        json.dump({'DJIA': _numeric_only(djia_metrics), 'MinVariance': _numeric_only(minvar_metrics)}, f, indent=2)
    print(f'  Saved {json_path}')

    bm_df = pd.DataFrame({'DJIA': djia_portfolio, 'MinVariance': minvar_portfolio})
    csv_path = RESULTS_DIR / 'benchmark_portfolios.csv'
    bm_df.to_csv(csv_path)
    print(f'  Saved {csv_path}')

    plot_portfolio_comparison(portfolios, title='All Strategies: Portfolio Value Comparison', filename='all_strategies_comparison.png')
    print('  Saved results/all_strategies_comparison.png')

    plot_drawdowns(portfolios, filename='drawdowns.png')
    print('  Saved results/drawdowns.png')

    col_map = {
        'cumulative_return': 'Cum. Return', 'annualized_return': 'Ann. Return',
        'annualized_volatility': 'Ann. Vol', 'sharpe_ratio': 'Sharpe', 'max_drawdown': 'Max DD',
    }
    all_metrics = {'DJIA Buy & Hold': djia_metrics, 'Min-Variance': minvar_metrics, **extra_metrics}
    metrics_rows = {s: {col_map[k]: v for k, v in m.items() if k in col_map} for s, m in all_metrics.items()}
    plot_metrics_table(pd.DataFrame(metrics_rows).T, filename='metrics_table.png')
    print('  Saved results/metrics_table.png')

    _save_results_md(djia_metrics, minvar_metrics, extra_metrics)

    print('\n' + '=' * 60)
    print('Experiment 4 complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()
