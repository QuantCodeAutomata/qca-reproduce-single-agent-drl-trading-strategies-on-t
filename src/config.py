"""
Configuration constants for the DRL stock-trading experiment.

Reproduces the setup from:
  'Practical Deep Reinforcement Learning Approach for Stock Trading'
  (FinRL / Ensemble DRL paper).

All design choices that are unspecified in the paper are documented inline.
"""

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# DJIA-30 membership as of 2016-01-01 (fixed, survivorship-aware).
# Using the 2016 composition avoids look-ahead bias relative to the
# 2016-01-04 trading-period start date.
DJIA_TICKERS = [
    'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
    'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
    'NKE', 'PFE', 'PG', 'T', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT'
]

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------

START_DATE = '2009-01-01'
END_DATE = '2020-05-08'

TRAIN_START = '2009-01-01'
TRAIN_END = '2015-09-30'

VAL_START = '2015-10-01'
VAL_END = '2015-12-31'

TRADE_START = '2016-01-04'
TRADE_END = '2020-05-08'

# ---------------------------------------------------------------------------
# Environment parameters
# ---------------------------------------------------------------------------

INITIAL_CAPITAL = 1_000_000.0
TRANSACTION_COST_PCT = 0.001            # 0.1 % per trade
H_MAX = 100                             # max shares per stock per action
                                        # (implementation choice, not in paper)

# State vector: [balance(1), shares(30), close(30), MACD(30),
#                RSI(30), CCI(30), ADX(30)] = 181
STATE_DIM = 181
N_STOCKS = 30

# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

ANNUALIZATION_FACTOR = 252
RISK_FREE_RATE = 0.0    # paper unspecified; set to 0 for Sharpe computation

# ---------------------------------------------------------------------------
# Turbulence
# ---------------------------------------------------------------------------

# Paper does not specify the exact threshold; use the 90th percentile of
# training-period turbulence values as a robust, data-driven cutoff.
TURBULENCE_THRESHOLD_QUANTILE = 0.90

# ---------------------------------------------------------------------------
# Technical indicator parameters  (paper: MACD, RSI, CCI, ADX)
# ---------------------------------------------------------------------------

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
CCI_PERIOD = 20
ADX_PERIOD = 14

# ---------------------------------------------------------------------------
# RL hyperparameters
# Paper leaves most hyperparameters unspecified; practical defaults below.
# ---------------------------------------------------------------------------

SEED = 42

PPO_PARAMS = {
    'learning_rate': 1e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'policy': 'MlpPolicy',
    'total_timesteps': 5000,    # paper unspecified; reduced for tractable CI run
}

A2C_PARAMS = {
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'policy': 'MlpPolicy',
    'total_timesteps': 5000,    # paper unspecified; reduced for tractable CI run
}

DDPG_PARAMS = {
    'learning_rate': 1e-4,
    'buffer_size': 100000,
    'batch_size': 64,
    'gamma': 0.99,
    'tau': 0.005,
    'action_noise_sigma': 0.1,
    # train_freq: use SB3 DDPG default (1, "episode") — one gradient update per
    # episode, which is the most efficient option for long episodes (~63 steps/quarter)
    'policy': 'MlpPolicy',
    'total_timesteps': 5000,    # paper unspecified; reduced for tractable CI run
}
