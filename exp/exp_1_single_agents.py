"""
Experiment 1: Single-Agent DRL Trading Strategies on DJIA-30
Reproduces standalone PPO, A2C, DDPG results from the paper.

Paper targets (not expected to match exactly due to omitted hyperparams):
  PPO:  cumulative=83.0%, annual=15.0%, vol=13.6%, Sharpe=1.10, maxDD=-23.7%
  A2C:  cumulative=60.0%, annual=11.4%, vol=10.4%, Sharpe=1.12, maxDD=-10.2%
  DDPG: cumulative=54.8%, annual=10.5%, vol=12.3%, Sharpe=0.87, maxDD=-14.8%

Implementation choices (not in paper):
  - H_MAX=100 (max shares per trade)
  - Turbulence threshold: 90th percentile of training-period turbulence
  - Transaction cost: 0.1% of absolute notional
  - Risk-free rate: 0 (paper unspecified)
  - Covariance estimation: expanding window with pseudoinverse for singularity
  - Retrain from scratch each quarter
  - Trade order: sells first, then buys by descending action magnitude
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DJIA_TICKERS,
    SEED,
    START_DATE,
    END_DATE,
    TRADE_END,
    TRADE_START,
    TRAIN_END,
    TRAIN_START,
    TURBULENCE_THRESHOLD_QUANTILE,
    VAL_END,
    VAL_START,
)
from src.data_loader import load_or_download
from src.feature_engineering import (
    add_technical_indicators,
    build_state_panel,
    compute_daily_returns,
)
from src.metrics import compute_all_metrics, metrics_to_dataframe
from src.rolling_schedule import generate_quarterly_windows, get_df_slice
from src.train_agent import (
    build_env,
    evaluate_agent,
    train_a2c,
    train_ddpg,
    train_ppo,
)
from src.turbulence import compute_turbulence_series, compute_turbulence_threshold

RESULTS_DIR = Path(__file__).parent.parent / 'results'

_TRAIN_FN = {
    'PPO': train_ppo,
    'A2C': train_a2c,
    'DDPG': train_ddpg,
}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print('Loading data...')
    df_raw = load_or_download(
        tickers=DJIA_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # ------------------------------------------------------------------
    # 2. Add technical indicators
    # ------------------------------------------------------------------
    print('Computing technical indicators...')
    df_ti = add_technical_indicators(df_raw)

    # ------------------------------------------------------------------
    # 3. Build state panel (wide format, DatetimeIndex)
    # ------------------------------------------------------------------
    print('Building state panel...')
    df_panel = build_state_panel(df_ti)

    # ------------------------------------------------------------------
    # 4. Compute daily returns and turbulence series
    # ------------------------------------------------------------------
    print('Computing turbulence series...')
    daily_returns = compute_daily_returns(df_ti)
    turbulence_series = compute_turbulence_series(daily_returns)

    # ------------------------------------------------------------------
    # 5. Compute turbulence threshold from training period
    # ------------------------------------------------------------------
    turbulence_threshold = compute_turbulence_threshold(
        turbulence_series,
        train_end=TRAIN_END,
        quantile=TURBULENCE_THRESHOLD_QUANTILE,
    )
    print(f'Turbulence threshold (90th pct, training period): {turbulence_threshold:.4f}')

    # ------------------------------------------------------------------
    # 6. Generate rolling quarterly windows
    # ------------------------------------------------------------------
    windows = generate_quarterly_windows(
        trade_start=TRADE_START,
        trade_end=TRADE_END,
        train_start=TRAIN_START,
        val_start=VAL_START,
        val_end=VAL_END,
    )
    print(f'Generated {len(windows)} quarterly windows '
          f'({windows[0]["deploy_start"]} → {windows[-1]["deploy_end"]})')

    # ------------------------------------------------------------------
    # 7. Train & deploy each algorithm independently
    # ------------------------------------------------------------------
    portfolio_values_dict = {}

    for algo in ['PPO', 'A2C', 'DDPG']:
        print(f'\n=== {algo} ===')
        train_fn = _TRAIN_FN[algo]
        chain_value = 1.0
        all_pv: list = []

        for i, window in enumerate(windows):
            deploy_label = f"{window['deploy_start'][:7]}"
            print(f'  [{i+1:02d}/{len(windows)}] {deploy_label}  '
                  f'train: {window["train_start"]} → {window["train_end"]}  '
                  f'deploy: {window["deploy_start"]} → {window["deploy_end"]}')

            train_slice = get_df_slice(df_panel, window['train_start'], window['train_end'])
            deploy_slice = get_df_slice(df_panel, window['deploy_start'], window['deploy_end'])

            if train_slice.empty or deploy_slice.empty:
                print(f'    Skipping: empty slice.')
                continue

            # Train
            env_train = build_env(
                train_slice, DJIA_TICKERS, turbulence_series, turbulence_threshold
            )
            model = train_fn(env_train, seed=SEED)

            # Deploy
            env_deploy = build_env(
                deploy_slice, DJIA_TICKERS, turbulence_series, turbulence_threshold
            )
            pv_q, _ = evaluate_agent(model, env_deploy)

            if pv_q.empty:
                print(f'    Skipping: empty portfolio values.')
                continue

            # Chain portfolio values across quarters
            pv_scaled = pv_q / pv_q.iloc[0] * chain_value
            chain_value = float(pv_scaled.iloc[-1])
            all_pv.append(pv_scaled)

        if all_pv:
            portfolio_values_dict[algo] = pd.concat(all_pv)
            print(f'  {algo} final portfolio value: {chain_value:.4f}')
        else:
            print(f'  {algo}: no portfolio data collected.')

    # ------------------------------------------------------------------
    # 8. Compute metrics
    # ------------------------------------------------------------------
    metrics_list = []
    for algo, pv in portfolio_values_dict.items():
        m = compute_all_metrics(pv, name=algo)
        metrics_list.append(m)
        print(f'\n{algo} metrics:')
        for k, v in m.items():
            if k != 'name':
                print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    # Portfolio values CSV
    pv_df = pd.DataFrame(portfolio_values_dict)
    pv_df.index.name = 'date'
    pv_path = RESULTS_DIR / 'single_agent_portfolios.csv'
    pv_df.to_csv(pv_path)
    print(f'\nPortfolio values saved to {pv_path}')

    # Metrics JSON (convert numpy floats for JSON serialisation)
    metrics_dict = {}
    for m in metrics_list:
        name = m['name']
        metrics_dict[name] = {
            k: float(v) for k, v in m.items() if k != 'name'
        }
    metrics_path = RESULTS_DIR / 'single_agent_metrics.json'
    with open(metrics_path, 'w') as fh:
        json.dump(metrics_dict, fh, indent=2)
    print(f'Metrics saved to {metrics_path}')

    # Print formatted metrics table
    if metrics_list:
        print('\n--- Metrics Summary ---')
        print(metrics_to_dataframe(metrics_list).to_string())

    return portfolio_values_dict, metrics_dict


if __name__ == '__main__':
    main()
