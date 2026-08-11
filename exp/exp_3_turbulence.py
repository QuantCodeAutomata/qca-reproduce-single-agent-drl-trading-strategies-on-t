"""
Experiment 3: Turbulence-Based Crash-Risk Control
Compares PPO with vs. without the turbulence override mechanism.

Variant A (no turbulence): standard PPO, no position override.
Variant B (with turbulence): PPO + turbulence liquidation override.

Same PPO model is trained each quarter; each trained model is evaluated
twice — once per variant — to isolate the effect of the turbulence
mechanism, not the training.

Focuses on 2020 crash period analysis (2020-01-01 to 2020-05-08).
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
from src.train_agent import build_env, evaluate_agent, train_ppo
from src.turbulence import compute_turbulence_series, compute_turbulence_threshold

RESULTS_DIR = Path(__file__).parent.parent / 'results'
CRASH_START = '2020-01-01'
CRASH_END = '2020-05-08'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    print('Loading data...')
    df_raw = load_or_download(
        tickers=DJIA_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print('Computing technical indicators...')
    df_ti = add_technical_indicators(df_raw)

    print('Building state panel...')
    df_panel = build_state_panel(df_ti)

    print('Computing turbulence series...')
    daily_returns = compute_daily_returns(df_ti)
    turbulence_series = compute_turbulence_series(daily_returns)

    turbulence_threshold = compute_turbulence_threshold(
        turbulence_series,
        train_end=TRAIN_END,
        quantile=TURBULENCE_THRESHOLD_QUANTILE,
    )
    print(f'Turbulence threshold: {turbulence_threshold:.4f}')

    # ------------------------------------------------------------------
    # 2. Generate rolling quarterly windows (PPO only)
    # ------------------------------------------------------------------
    windows = generate_quarterly_windows(
        trade_start=TRADE_START,
        trade_end=TRADE_END,
        train_start=TRAIN_START,
        val_start=VAL_START,
        val_end=VAL_END,
    )
    print(f'Generated {len(windows)} quarterly windows.')

    # ------------------------------------------------------------------
    # 3. Train PPO each quarter; evaluate both variants
    # ------------------------------------------------------------------
    chain_with = 1.0        # Variant B: with turbulence
    chain_without = 1.0     # Variant A: without turbulence

    pv_with_list: list = []
    pv_without_list: list = []

    for i, window in enumerate(windows):
        deploy_label = f"{window['deploy_start'][:7]}"
        print(f'\n[{i+1:02d}/{len(windows)}] {deploy_label}')

        train_slice = get_df_slice(df_panel, window['train_start'], window['train_end'])
        deploy_slice = get_df_slice(df_panel, window['deploy_start'], window['deploy_end'])

        if train_slice.empty or deploy_slice.empty:
            print('  Skipping: empty slice.')
            continue

        # --- Train PPO once per window ---
        print('  Training PPO...')
        env_train = build_env(
            train_slice, DJIA_TICKERS, turbulence_series, turbulence_threshold
        )
        model = train_ppo(env_train, seed=SEED)

        # --- Variant B: deploy WITH turbulence override ---
        env_with = build_env(
            deploy_slice, DJIA_TICKERS, turbulence_series, turbulence_threshold
        )
        pv_with_q, _ = evaluate_agent(model, env_with)

        # --- Variant A: deploy WITHOUT turbulence override ---
        env_without = build_env(
            deploy_slice,
            DJIA_TICKERS,
            turbulence_series=None,
            turbulence_threshold=None,
        )
        pv_without_q, _ = evaluate_agent(model, env_without)

        if pv_with_q.empty or pv_without_q.empty:
            print('  Skipping: empty portfolio values.')
            continue

        # Chain Variant B
        pv_with_scaled = pv_with_q / pv_with_q.iloc[0] * chain_with
        chain_with = float(pv_with_scaled.iloc[-1])
        pv_with_list.append(pv_with_scaled)

        # Chain Variant A
        pv_without_scaled = pv_without_q / pv_without_q.iloc[0] * chain_without
        chain_without = float(pv_without_scaled.iloc[-1])
        pv_without_list.append(pv_without_scaled)

        print(f'  With turb: {chain_with:.4f}  |  Without turb: {chain_without:.4f}')

    if not pv_with_list or not pv_without_list:
        raise RuntimeError('No portfolio data collected.')

    pv_with = pd.concat(pv_with_list)
    pv_without = pd.concat(pv_without_list)
    pv_with.name = 'PPO_with_turbulence'
    pv_without.name = 'PPO_no_turbulence'

    # ------------------------------------------------------------------
    # 5. Compute full-period metrics for both variants
    # ------------------------------------------------------------------
    metrics_with = compute_all_metrics(pv_with, name='PPO (with turbulence)')
    metrics_without = compute_all_metrics(pv_without, name='PPO (no turbulence)')

    # ------------------------------------------------------------------
    # 6. Compute crash-period metrics (2020-01-01 to 2020-05-08)
    # ------------------------------------------------------------------
    crash_with = pv_with.loc[CRASH_START:CRASH_END]
    crash_without = pv_without.loc[CRASH_START:CRASH_END]

    crash_metrics_with = {}
    crash_metrics_without = {}
    if len(crash_with) > 1:
        crash_metrics_with = compute_all_metrics(
            crash_with, name='PPO (with turbulence) – crash'
        )
    if len(crash_without) > 1:
        crash_metrics_without = compute_all_metrics(
            crash_without, name='PPO (no turbulence) – crash'
        )

    # Print summaries
    full_period_list = [metrics_with, metrics_without]
    print('\n--- Full-Period Metrics ---')
    print(metrics_to_dataframe(full_period_list).to_string())

    crash_list = [m for m in [crash_metrics_with, crash_metrics_without] if m]
    if crash_list:
        print('\n--- Crash-Period Metrics (2020-01-01 to 2020-05-08) ---')
        print(metrics_to_dataframe(crash_list).to_string())

    # ------------------------------------------------------------------
    # 7. Generate turbulence event log (when override was triggered)
    # ------------------------------------------------------------------
    trade_start_ts = pd.Timestamp(TRADE_START)
    trade_end_ts = pd.Timestamp(TRADE_END)
    trade_turb = turbulence_series[
        (turbulence_series.index >= trade_start_ts) &
        (turbulence_series.index <= trade_end_ts)
    ]
    events = trade_turb[trade_turb > turbulence_threshold].reset_index()
    events.columns = ['date', 'turbulence_value']
    events['threshold'] = turbulence_threshold
    events['excess'] = events['turbulence_value'] - turbulence_threshold
    print(f'\nTurbulence override triggered on {len(events)} trading days.')

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    # Both portfolio series
    comparison_df = pd.DataFrame({
        'PPO_with_turbulence': pv_with,
        'PPO_no_turbulence': pv_without,
    })
    comparison_df.index.name = 'date'
    comp_path = RESULTS_DIR / 'turbulence_comparison.csv'
    comparison_df.to_csv(comp_path)
    print(f'Portfolio comparison saved to {comp_path}')

    # Event log
    event_path = RESULTS_DIR / 'turbulence_events.csv'
    events.to_csv(event_path, index=False)
    print(f'Turbulence event log saved to {event_path}')

    # Metrics comparison JSON
    def _metrics_to_dict(m: dict) -> dict:
        return {k: float(v) for k, v in m.items() if k != 'name'}

    metrics_dict = {
        'full_period': {
            'PPO_with_turbulence': _metrics_to_dict(metrics_with),
            'PPO_no_turbulence': _metrics_to_dict(metrics_without),
        },
        'crash_period': {},
    }
    if crash_metrics_with:
        metrics_dict['crash_period']['PPO_with_turbulence'] = _metrics_to_dict(crash_metrics_with)
    if crash_metrics_without:
        metrics_dict['crash_period']['PPO_no_turbulence'] = _metrics_to_dict(crash_metrics_without)

    metrics_path = RESULTS_DIR / 'turbulence_metrics.json'
    with open(metrics_path, 'w') as fh:
        json.dump(metrics_dict, fh, indent=2)
    print(f'Metrics saved to {metrics_path}')

    return (pv_with, pv_without), metrics_dict


if __name__ == '__main__':
    main()
