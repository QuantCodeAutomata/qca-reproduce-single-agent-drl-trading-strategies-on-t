"""
Agent training and evaluation utilities for the DRL stock-trading experiment.

Supports PPO, A2C, and DDPG via stable-baselines3.  Each training function
wraps the raw gymnasium environment in a DummyVecEnv as required by SB3.

DataFrame format contract
--------------------------
build_env() accepts the wide-format panel produced by build_state_panel()
(columns: ``close_<TICKER>``, ``macd_<TICKER>``, ``rsi_<TICKER>``,
``cci_<TICKER>``, ``adx_<TICKER>``; index: DatetimeIndex) and converts it
to the MultiIndex ``(ticker, field)`` format expected by StockTradingEnv,
mapping the ``close`` field to ``adj_close`` (equivalent in this dataset
because download uses ``adjusted=True``).
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import A2C_PARAMS, DDPG_PARAMS, N_STOCKS, PPO_PARAMS, SEED
from src.env_trading import StockTradingEnv
from src.metrics import compute_sharpe_ratio

# Fields produced by build_state_panel → mapped to env field names
_PANEL_TO_ENV_FIELD = {
    'close': 'adj_close',
    'macd': 'macd',
    'rsi': 'rsi',
    'cci': 'cci',
    'adx': 'adx',
}

# Keys to pop from params before forwarding to model constructors
_NON_CONSTRUCTOR_KEYS = {'policy', 'total_timesteps', 'action_noise_sigma'}


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def _panel_to_multiindex(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Convert flat-column panel (``field_TICKER``) to MultiIndex ``(TICKER, field)``."""
    data = {}
    for src_field, dst_field in _PANEL_TO_ENV_FIELD.items():
        for ticker in tickers:
            flat_col = f'{src_field}_{ticker}'
            if flat_col in df.columns:
                data[(ticker, dst_field)] = df[flat_col].values

    mi_df = pd.DataFrame(data, index=df.index)
    mi_df.columns = pd.MultiIndex.from_tuples(mi_df.columns)
    return mi_df


def build_env(
    df_slice: pd.DataFrame,
    tickers: list,
    turbulence_series: Optional[pd.Series] = None,
    turbulence_threshold: Optional[float] = None,
) -> StockTradingEnv:
    """Build the trading environment for a given date slice.

    Accepts either the flat-column panel from ``build_state_panel()`` or a
    DataFrame already carrying a MultiIndex column structure.  The turbulence
    series is passed through unfiltered; ``TurbulenceMonitor`` handles missing
    dates gracefully.

    Parameters
    ----------
    df_slice:
        Date-sliced stock feature panel.
    tickers:
        Ordered list of ticker symbols (must match columns).
    turbulence_series:
        Full turbulence series (indexed by date); ``None`` disables the
        turbulence override.
    turbulence_threshold:
        Activation threshold for the turbulence override.

    Returns
    -------
    StockTradingEnv
    """
    if isinstance(df_slice.columns, pd.MultiIndex):
        env_df = df_slice
    else:
        env_df = _panel_to_multiindex(df_slice, tickers)

    return StockTradingEnv(
        df=env_df,
        tickers=tickers,
        turbulence_series=turbulence_series,
        turbulence_threshold=turbulence_threshold,
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_vec_env(env: StockTradingEnv) -> DummyVecEnv:
    """Wrap a single environment in DummyVecEnv, required by SB3."""
    return DummyVecEnv([lambda: env])  # noqa: B023


def _model_kwargs(params: dict) -> dict:
    """Strip non-constructor keys from a params dict."""
    return {k: v for k, v in params.items() if k not in _NON_CONSTRUCTOR_KEYS}


def train_ppo(
    env: StockTradingEnv,
    params: dict = PPO_PARAMS,
    seed: int = SEED,
) -> PPO:
    """Train a PPO agent.

    Parameters
    ----------
    env:
        Training environment (will be wrapped in DummyVecEnv internally).
    params:
        Hyper-parameters including ``policy`` and ``total_timesteps``.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    PPO
        Trained model.
    """
    policy = params.get('policy', 'MlpPolicy')
    total_timesteps = params.get('total_timesteps', 50_000)
    vec_env = _make_vec_env(env)
    model = PPO(policy, vec_env, seed=seed, **_model_kwargs(params))
    model.learn(total_timesteps=total_timesteps)
    return model


def train_a2c(
    env: StockTradingEnv,
    params: dict = A2C_PARAMS,
    seed: int = SEED,
) -> A2C:
    """Train an A2C agent.

    Parameters
    ----------
    env:
        Training environment.
    params:
        Hyper-parameters including ``policy`` and ``total_timesteps``.
    seed:
        Random seed.

    Returns
    -------
    A2C
        Trained model.
    """
    policy = params.get('policy', 'MlpPolicy')
    total_timesteps = params.get('total_timesteps', 50_000)
    vec_env = _make_vec_env(env)
    model = A2C(policy, vec_env, seed=seed, **_model_kwargs(params))
    model.learn(total_timesteps=total_timesteps)
    return model


def train_ddpg(
    env: StockTradingEnv,
    params: dict = DDPG_PARAMS,
    seed: int = SEED,
) -> DDPG:
    """Train a DDPG agent.

    Action noise is ``NormalActionNoise(mean=0, sigma=0.1)`` applied to each
    action dimension independently.

    Parameters
    ----------
    env:
        Training environment.
    params:
        Hyper-parameters including ``policy``, ``total_timesteps``, and
        optionally ``action_noise_sigma`` (default 0.1).
    seed:
        Random seed.

    Returns
    -------
    DDPG
        Trained model.
    """
    policy = params.get('policy', 'MlpPolicy')
    total_timesteps = params.get('total_timesteps', 50_000)
    sigma = params.get('action_noise_sigma', 0.1)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=sigma * np.ones(n_actions),
    )

    vec_env = _make_vec_env(env)
    model = DDPG(
        policy,
        vec_env,
        action_noise=action_noise,
        seed=seed,
        **_model_kwargs(params),
    )
    model.learn(total_timesteps=total_timesteps)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    model,
    env: StockTradingEnv,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Run the agent through the environment deterministically.

    The agent steps through the full episode without exploration noise.
    Portfolio values and the daily log are retrieved from the environment's
    internal memory after the episode completes.

    Parameters
    ----------
    model:
        Trained SB3 model (PPO, A2C, or DDPG).
    env:
        Deployment environment (raw, not vectorised).

    Returns
    -------
    portfolio_values : pd.Series
        Portfolio value at each trading step, indexed by date.
    daily_log : pd.DataFrame
        Per-step log from ``env.get_daily_log()``.
    """
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

    daily_log = env.get_daily_log()

    pv_series = daily_log.set_index('date')['portfolio_value']
    pv_series.index = pd.to_datetime(pv_series.index)
    pv_series.name = 'portfolio_value'

    return pv_series, daily_log


# ---------------------------------------------------------------------------
# Batch training / selection
# ---------------------------------------------------------------------------

def train_all_agents(
    train_df: pd.DataFrame,
    tickers: list,
    turbulence_series: pd.Series,
    turbulence_threshold: float,
    seed: int = SEED,
) -> dict:
    """Train PPO, A2C, and DDPG on *train_df*.

    A fresh environment is created for each algorithm to avoid any shared
    internal state.

    Parameters
    ----------
    train_df:
        Training date slice (panel format from ``build_state_panel``).
    tickers:
        Ticker list.
    turbulence_series:
        Full turbulence series (used during training to learn turbulence
        response).
    turbulence_threshold:
        Turbulence override activation level.
    seed:
        Random seed (applied uniformly to all three agents).

    Returns
    -------
    dict
        ``{'PPO': model, 'A2C': model, 'DDPG': model}``
    """
    models = {}
    for algo_name, train_fn in [
        ('PPO', train_ppo),
        ('A2C', train_a2c),
        ('DDPG', train_ddpg),
    ]:
        env = build_env(train_df, tickers, turbulence_series, turbulence_threshold)
        models[algo_name] = train_fn(env, seed=seed)
    return models


def validate_agents(
    models: dict,
    val_df: pd.DataFrame,
    tickers: list,
    turbulence_series: pd.Series,
    turbulence_threshold: float,
) -> dict:
    """Evaluate each model on *val_df* and return Sharpe ratios.

    Parameters
    ----------
    models:
        Dict mapping algorithm name → trained model.
    val_df:
        Validation date slice.
    tickers:
        Ticker list.
    turbulence_series:
        Turbulence series (active during validation).
    turbulence_threshold:
        Turbulence override level.

    Returns
    -------
    dict
        ``{algo_name: sharpe_ratio}``
    """
    sharpes = {}
    for name, model in models.items():
        env = build_env(val_df, tickers, turbulence_series, turbulence_threshold)
        pv, _ = evaluate_agent(model, env)
        sharpes[name] = compute_sharpe_ratio(pv) if len(pv) > 1 else 0.0
    return sharpes


def select_best_model(validation_sharpes: dict) -> str:
    """Return the name of the model with the highest Sharpe ratio.

    Tie-break order: PPO > A2C > DDPG.

    Parameters
    ----------
    validation_sharpes:
        Dict mapping algorithm name → Sharpe ratio.

    Returns
    -------
    str
        Name of the best algorithm.
    """
    tiebreak_order = ['PPO', 'A2C', 'DDPG']
    best_name = None
    best_sharpe = -np.inf

    for name in tiebreak_order:
        if name not in validation_sharpes:
            continue
        sharpe = validation_sharpes[name]
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_name = name

    if best_name is None:
        # Fallback: any key with highest value
        best_name = max(validation_sharpes, key=validation_sharpes.get)

    return best_name


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str) -> None:
    """Save a trained SB3 model to disk.

    Parameters
    ----------
    model:
        Trained SB3 model.
    path:
        Destination file path (without ``.zip`` extension; SB3 adds it).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    model.save(path)


def load_model(algo_name: str, path: str):
    """Load a trained SB3 model from disk.

    Parameters
    ----------
    algo_name:
        One of ``'PPO'``, ``'A2C'``, ``'DDPG'``.
    path:
        Path to the saved model file.

    Returns
    -------
    Loaded SB3 model.
    """
    algo_map = {'PPO': PPO, 'A2C': A2C, 'DDPG': DDPG}
    cls = algo_map[algo_name]
    return cls.load(path)
