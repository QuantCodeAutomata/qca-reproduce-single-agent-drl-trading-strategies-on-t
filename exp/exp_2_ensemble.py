"""
Experiment 2: Rolling Ensemble Selection Strategy
Reproduces the paper's ensemble: PPO+A2C+DDPG with quarterly selection by
validation Sharpe.

Paper targets:
  Ensemble: cumulative=70.4%, annual=13.0%, vol=9.7%, Sharpe=1.30, maxDD=-9.7%

Paper's Table I selection sequence:
  2016Q1:PPO,  2016Q2:DDPG, 2016Q3:DDPG, 2016Q4:PPO,
  2017Q1:PPO,  2017Q2:A2C,  2017Q3:PPO,  2017Q4:DDPG,
  2018Q1:PPO,  2018Q2:DDPG, 2018Q3:A2C,  2018Q4:A2C,
  2019Q1:DDPG, 2019Q2:PPO,  2019Q3:PPO,  2019Q4:A2C,
  2020Q1:A2C,  2020Q2:A2C

Implementation choices:
  - Retrain from scratch each quarter
  - Portfolio continuity preserved across quarters
  - Tie-break: PPO > A2C > DDPG
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
    select_best_model,
    train_all_agents,
    validate_agents,
)
from src.turbulence import compute_turbulence_series, compute_turbulence_threshold

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and prepare data  (same pipeline as Experiment 1)
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
    # 2. Generate rolling quarterly windows
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
    # 3. Rolling ensemble: train all, validate, select, deploy
    # ------------------------------------------------------------------
    chain_value = 1.0
    all_pv: list = []
    selection_log: list = []

    for i, window in enumerate(windows):
        deploy_label = f"{window['deploy_start'][:7]}"
        print(f'\n[{i+1:02d}/{len(windows)}] {deploy_label}  '
              f'train: {window["train_start"]} → {window["train_end"]}  '
              f'val: {window["val_start"]} → {window["val_end"]}  '
              f'deploy: {window["deploy_start"]} → {window["deploy_end"]}')

        train_slice = get_df_slice(df_panel, window['train_start'], window['train_end'])
        val_slice = get_df_slice(df_panel, window['val_start'], window['val_end'])
        deploy_slice = get_df_slice(df_panel, window['deploy_start'], window['deploy_end'])

        if train_slice.empty or val_slice.empty or deploy_slice.empty:
            print('  Skipping: empty slice.')
            continue

        # --- 3a. Train all three agents ---
        print('  Training PPO, A2C, DDPG...')
        models = train_all_agents(
            train_df=train_slice,
            tickers=DJIA_TICKERS,
            turbulence_series=turbulence_series,
            turbulence_threshold=turbulence_threshold,
            seed=SEED,
        )

        # --- 3b. Validate: Sharpe on validation slice ---
        print('  Validating...')
        val_sharpes = validate_agents(
            models=models,
            val_df=val_slice,
            tickers=DJIA_TICKERS,
            turbulence_series=turbulence_series,
            turbulence_threshold=turbulence_threshold,
        )
        print(f'  Val Sharpes — PPO: {val_sharpes.get("PPO", float("nan")):.3f}  '
              f'A2C: {val_sharpes.get("A2C", float("nan")):.3f}  '
              f'DDPG: {val_sharpes.get("DDPG", float("nan")):.3f}')

        # --- 3c. Select best model ---
        best_name = select_best_model(val_sharpes)
        print(f'  Selected: {best_name}')

        # --- 3d. Deploy best model ---
        env_deploy = build_env(
            deploy_slice, DJIA_TICKERS, turbulence_series, turbulence_threshold
        )
        pv_q, _ = evaluate_agent(models[best_name], env_deploy)

        if pv_q.empty:
            print('  Skipping: empty portfolio values.')
            continue

        # Chain portfolio values (preserve continuity across quarters)
        pv_scaled = pv_q / pv_q.iloc[0] * chain_value
        chain_value = float(pv_scaled.iloc[-1])
        all_pv.append(pv_scaled)

        selection_log.append({
            'window': deploy_label,
            'PPO_sharpe': float(val_sharpes.get('PPO', np.nan)),
            'A2C_sharpe': float(val_sharpes.get('A2C', np.nan)),
            'DDPG_sharpe': float(val_sharpes.get('DDPG', np.nan)),
            'selected': best_name,
        })

    # ------------------------------------------------------------------
    # 5. Aggregate portfolio values
    # ------------------------------------------------------------------
    if not all_pv:
        raise RuntimeError('No portfolio data collected across any window.')

    ensemble_pv = pd.concat(all_pv)
    ensemble_pv.name = 'Ensemble'

    # ------------------------------------------------------------------
    # 6. Compute metrics
    # ------------------------------------------------------------------
    metrics = compute_all_metrics(ensemble_pv, name='Ensemble')
    print('\n--- Ensemble Metrics ---')
    for k, v in metrics.items():
        if k != 'name':
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    metrics_table = metrics_to_dataframe([metrics])
    print('\n' + metrics_table.to_string())

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    # Ensemble portfolio CSV
    pv_path = RESULTS_DIR / 'ensemble_portfolio.csv'
    ensemble_pv.to_csv(pv_path, header=True)
    print(f'\nEnsemble portfolio saved to {pv_path}')

    # Selection log CSV
    if selection_log:
        sel_df = pd.DataFrame(selection_log)
        sel_path = RESULTS_DIR / 'ensemble_selection_log.csv'
        sel_df.to_csv(sel_path, index=False)
        print(f'Selection log saved to {sel_path}')
        print('\n--- Selection Log ---')
        print(sel_df.to_string(index=False))

    # Metrics JSON
    metrics_dict = {'Ensemble': {k: float(v) for k, v in metrics.items() if k != 'name'}}
    metrics_path = RESULTS_DIR / 'ensemble_metrics.json'
    with open(metrics_path, 'w') as fh:
        json.dump(metrics_dict, fh, indent=2)
    print(f'Metrics saved to {metrics_path}')

    return ensemble_pv, metrics_dict


if __name__ == '__main__':
    main()
